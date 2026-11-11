"""
go2_deploy_sdk2.py - [P1.5] Run the state machine (go2_fsm.py) on the Go2, or in unitree_mujoco, through
unitree_sdk2_python's low-level interface.

!!! NOT TESTED ON HARDWARE. Written from Unitree's example/go2/low_level/go2_stand_example.py. [S2R] Run end to end
!!! against unitree_mujoco (simulate_python bridge) over DDS on the loopback interface, never on a real Go2.
!!! Go step by step: 1) unitree_mujoco, 2) robot hung on a stand, feet clear of the ground, 3) on the ground.

    python go2_deploy_sdk2.py --policy logs/go2-p15/exported --net enp3s0   # robot: interface wired to the Go2
    python go2_deploy_sdk2.py --policy logs/go2-p15/exported --sim          # unitree_mujoco (domain 1, "lo")
[S2R]
    python go2_deploy_sdk2.py --policy ... --sim --scenario walk --profile fwd_back --yes   # scripted: no gamepad
    python go2_deploy_sdk2.py --policy ... --net enp3s0 --zero-action      # first robot runs: hold the default pose
    python go2_deploy_sdk2.py --policy ... --net enp3s0 --profile slow     # L1+X in policy mode runs the profile
    python go2_deploy_sdk2.py --policy ... --net enp3s0 --allow-joint-test # robot hung: L1+B runs the joint test
    python go2_deploy_sdk2.py --policy ... --net enp3s0 --keys             # PC keyboard too: space = damping
    python go2_deploy_sdk2.py --policy ... --rehearsal --keys              # the robot path against unitree_mujoco
                                                     # (go2_mujoco_headless.py --robot-services): a dress rehearsal

What it does
- On the robot, it first releases Unitree's sport mode (MotionSwitcherClient, as go2_stand_example.py does), so
  the onboard controller stops sending its own commands. The robot lies down first (StandDown).
- Every 2 ms it reads rt/lowstate (joints, IMU, gamepad), calls Go2StateMachine.update() and writes rt/lowcmd.
- It starts in damping. Gamepad: L1+R2 stand up, L1+A policy, L1+R1 sit down, L1+Y damping (software e-stop).
  [P1.6] Policies trained with the body-height command: D-pad up / down = body height +-1 cm.
  [S2R] L1+X starts the --profile command profile (policy mode), L1+B the joint test (--allow-joint-test, standing).
- Damping if no new robot state arrived for 50 ms, and for 1 s on Ctrl+C before exiting.
  [S2R] Also on any error in the control step: a damping command (kp 0, kd 3) is written whatever happens.
- [S2R] Logs every control step and every policy step (go2_log.py) to logs/deploy/<robot|mujoco>_<date>[_tag].npz,
  for go2_analyze.py; prints a status line every second (mode, command, loop timing, torque demand).
"""

import argparse
import gc
import json
import os
import signal
import sys
import threading
import time
import traceback

import numpy as np

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_, SportModeState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

import go2_log
from go2_fsm import MODES, FsmConfig, Go2StateMachine, Mode, Policy
from go2_inputs import PROFILES, SCENARIOS, CommandProfile, Scenario, buttons_mask, parse_remote

# Motor index of each joint in rt/lowcmd and rt/lowstate (unitree_legged_const.py LegID: FR_0 = 0 ... RL_2 = 11)
SDK_JOINTS = ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
              "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"]
POS_STOP_F, VEL_STOP_F = 2.146e9, 16000.0  # unitree_legged_const.py: "no position / velocity target"
CONTROL_DT = 0.002  # [s] as go2_stand_example.py
STATE_TIMEOUT_S = 0.05
# [S2R] --keys: a key "presses" the button combo for 0.1 s, like a scripted press
KEY_COMBOS = {" ": "damping", "0": "damping", "1": "stand_up", "2": "policy", "3": "sit_down", "4": "profile",
              "5": "joint_test"}


def set_realtime(priority=50):
    """[S2R] --rt: SCHED_FIFO for the calling thread and every thread started from it afterwards. main() calls it
    before ChannelFactoryInitialize, so the DDS threads (they run the state handler, holding the Python interpreter
    lock) and the 2 ms control loop all get it: other programs cannot preempt them, and the loop never waits for a
    preempted thread that holds the lock. Needs root, or an rtprio limit (/etc/security/limits.conf:
    <user> - rtprio 98, log in again)."""
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(priority))
        print(f"real-time priority: SCHED_FIFO {priority} for the DDS threads and the control loop")
    except (AttributeError, OSError) as e:
        print(f"real-time priority NOT set ({e}): run as root or raise the rtprio limit")


def release_sport_mode(timeout_s=30.0):
    """go2_stand_example.py: lie down with the sport controller, then release it.
    [S2R] Fails with a clear message when the motion switcher does not answer (robot off, wrong --net), waits for the
    robot to be down before releasing, and gives up after timeout_s."""
    sc = SportClient()
    sc.SetTimeout(5.0)
    sc.Init()
    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()
    t_end = time.monotonic() + timeout_s
    while True:
        code, result = msc.CheckMode()
        if result is None:
            raise SystemExit(f"motion_switcher did not answer (code {code}): is the robot on, and is --net the "
                             f"interface wired to it (ping 192.168.123.161)?")
        if not result.get("name"):
            print("sport mode released: the onboard controller sends no command now")
            return
        if time.monotonic() > t_end:
            raise SystemExit(f"sport mode {result['name']!r} still active after {timeout_s:.0f} s")
        print(f"sport mode {result['name']!r} active: StandDown, then ReleaseMode")
        sc.StandDown()
        time.sleep(3.0)  # let the robot get down before its controller lets go
        msc.ReleaseMode()
        time.sleep(1.0)


class Keyboard:
    """[S2R] The PC keyboard as a second remote (--keys): 1 stand up, 2 policy, 3 sit down, 4 profile, 5 joint test,
    space or 0 damping. Each key holds its button combo for 0.1 s."""

    def __init__(self, combos):
        import termios
        import tty

        if not sys.stdin.isatty():
            raise SystemExit("--keys needs an interactive terminal")
        self.combos = combos
        self.fd = sys.stdin.fileno()
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)  # one key at a time, no echo; Ctrl+C still works
        self.lock = threading.Lock()
        self.held = {}  # combo -> monotonic time until which it is held
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        while True:
            ch = os.read(self.fd, 1).decode(errors="ignore")
            combo = KEY_COMBOS.get(ch)
            if combo is not None:
                with self.lock:
                    self.held[combo] = time.monotonic() + 0.1

    def buttons(self, now):
        with self.lock:
            return {b for combo, until in self.held.items() if now < until for b in self.combos[combo]}

    def close(self):
        import termios

        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)


class Go2Deploy:
    def __init__(self, fsm: Go2StateMachine, sdk_idx, recorder=None, scenario=None, keyboard=None):
        self.fsm, self.sdk_idx = fsm, sdk_idx  # sdk_idx[i]: motor index of policy joint i
        self.recorder, self.scenario, self.keyboard = recorder, scenario, keyboard  # [S2R]
        self.crc = CRC()
        self.cmd = unitree_go_msg_dds__LowCmd_()
        self.cmd.head[0], self.cmd.head[1] = 0xFE, 0xEF
        self.cmd.level_flag = 0xFF
        self.cmd.gpio = 0
        for m in self.cmd.motor_cmd:  # 20 slots, as InitLowCmd(): servo mode, no target, no gains
            m.mode, m.q, m.dq, m.kp, m.kd, m.tau = 0x01, POS_STOP_F, VEL_STOP_F, 0.0, 0.0, 0.0
        self.lock = threading.Lock()
        self.state, self.state_time, self.n_states = None, 0.0, 0
        self.t0 = time.monotonic() if recorder is None else recorder.t0_mono
        self.t_scenario0 = None  # [S2R] scenario clock starts at the first robot state
        self.prev_start, self.prev_state = None, None
        self.stop_requested = False
        self.error = False
        # [S2R] recent timings for the status line
        self.recent = {k: np.zeros(500) for k in ("loop_dt", "state_age", "compute", "tau_ratio")}
        self.n_steps = 0
        self.publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.publisher.Init()
        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        # [S2R] queue length 0: the handler (it only stores the message) runs in the DDS listener, one thread hop less
        # than the example's queue of 10 (whose reader thread drops new samples, not old ones, when it falls behind)
        self.subscriber.Init(self.on_state, 0)

    def on_state(self, msg: LowState_):
        with self.lock:
            self.state, self.state_time = msg, time.monotonic()
            self.n_states += 1

    def step(self):
        try:
            self._step()
        except Exception:  # [S2R] never leave the motors without a command
            if not self.error:
                print("error in the control step, damping:\n" + traceback.format_exc())
            self.error = True
            self.fsm.emergency_damping(time.monotonic() - self.t0, "error in the control step")
            self._write_damping()

    def _write_damping(self):
        """kp 0, kd = passive_kd, no position target: damping whatever the robot state."""
        for j in self.sdk_idx:
            m = self.cmd.motor_cmd[j]
            m.q, m.dq, m.kp, m.kd, m.tau = POS_STOP_F, 0.0, 0.0, float(self.fsm.cfg.passive_kd), 0.0
        self.cmd.crc = self.crc.Crc(self.cmd)
        self.publisher.Write(self.cmd)

    def _step(self):
        start = time.monotonic()
        with self.lock:
            s, state_time = self.state, self.state_time
        if s is None:  # no command before the first robot state
            return
        t = start - self.t0
        age = start - state_time
        loop_dt = 0.0 if self.prev_start is None else start - self.prev_start
        self.prev_start = start
        q = np.array([s.motor_state[j].q for j in self.sdk_idx], dtype=float)
        dq = np.array([s.motor_state[j].dq for j in self.sdk_idx], dtype=float)
        gyro = np.array(s.imu_state.gyroscope, dtype=float)  # base frame [rad/s]
        quat = np.array(s.imu_state.quaternion, dtype=float)  # (w, x, y, z)
        buttons, sticks = parse_remote(s.wireless_remote)
        flags = 0
        if self.scenario is not None:  # [S2R] scripted presses (simulators)
            if self.t_scenario0 is None:
                self.t_scenario0 = t
            scripted = self.scenario.buttons_at(t - self.t_scenario0, self.fsm.cfg.combos)
            if scripted:
                buttons |= scripted
                flags |= go2_log.FLAG_SCRIPTED
        if self.keyboard is not None:  # [S2R]
            keys = self.keyboard.buttons(start)
            if keys:
                buttons |= keys
                flags |= go2_log.FLAG_SCRIPTED
        if age > STATE_TIMEOUT_S:
            self.fsm.emergency_damping(t, f"no robot state for {1000 * age:.0f} ms")
            flags |= go2_log.FLAG_STALE
        if self.stop_requested:
            self.fsm.emergency_damping(t, "Ctrl+C")
        q_des, dq_des, kp, kd, tau = self.fsm.update(t, q, dq, gyro, quat, buttons, sticks)
        for i, j in enumerate(self.sdk_idx):
            m = self.cmd.motor_cmd[j]
            m.q, m.dq, m.kp, m.kd, m.tau = float(q_des[i]), float(dq_des[i]), float(kp[i]), float(kd[i]), float(tau[i])
        self.cmd.crc = self.crc.Crc(self.cmd)
        self.publisher.Write(self.cmd)
        compute = time.monotonic() - start

        # [S2R] bookkeeping and logs (after the command is out)
        if s is self.prev_state:  # no new rt/lowstate since the previous step
            flags |= go2_log.FLAG_REPEAT
        self.prev_state = s
        if self.fsm.last_clamped:
            flags |= go2_log.FLAG_CLAMPED
        if self.fsm.profile_t0 is not None:
            flags |= go2_log.FLAG_PROFILE
        k = self.n_steps % 500
        tau_ratio = float(np.max(np.abs(self.fsm.last_tau_cmd) / self.fsm.torque_limits))
        for name, v in (("loop_dt", loop_dt), ("state_age", age), ("compute", compute), ("tau_ratio", tau_ratio)):
            self.recent[name][k] = v
        self.n_steps += 1
        if self.recorder is None:
            return
        self.recorder.append(
            "ctrl", t=t, mode=MODES.index(self.fsm.mode), q=q, dq=dq,
            tau_est=[s.motor_state[j].tau_est for j in self.sdk_idx],
            quat=quat, gyro=gyro, acc=s.imu_state.accelerometer, foot_force=s.foot_force,
            q_des=q_des, dq_des=dq_des, kp=kp, kd=kd, tau_ff=tau, tau_cmd=self.fsm.last_tau_cmd,
            tick=s.tick, state_age=age, loop_dt=loop_dt, compute=compute,
            buttons=buttons_mask(buttons), sticks=sticks, flags=flags)
        if self.fsm.policy_stepped:
            p = self.fsm.policy
            self.recorder.append("pol", t=t, obs=p.last_obs, action=p.last_action, cmd=self.fsm.command,
                                 infer=p.last_infer_s, seg=self.fsm.profile_seg)

    def on_sport(self, msg):
        """[S2R] unitree_mujoco only: rt/sportmodestate carries the simulator's ground truth (position and world
        velocity of the IMU site); the real robot does not publish it once sport mode is released."""
        if self.recorder is not None:
            self.recorder.append("truth", t=time.monotonic() - self.t0, pos=msg.position, lin_vel=msg.velocity)

    def status(self):
        """[S2R] One line for the terminal: mode, command, loop timing and torque demand over the last second."""
        n = min(self.n_steps, 500)
        if n == 0:
            return None
        r = {k: v[:n] for k, v in self.recent.items()}
        p99 = {k: 1e3 * np.percentile(v, 99) for k, v in r.items()}  # [ms]
        fsm = self.fsm
        cmd = ""
        if fsm.mode is Mode.POLICY:
            cmd = f" | cmd vx {fsm.command[0]:+.2f} vy {fsm.command[1]:+.2f} wz {fsm.command[2]:+.2f}"
            if len(fsm.command) > 3:
                cmd += f" dh {100 * fsm.command[3]:+.0f} cm"  # [P1.6]
            if fsm.profile_t0 is not None:
                cmd += f" (profile {fsm.profile.name}, segment {fsm.profile_seg})"
        return (f"{fsm.mode.value:10s}{cmd} | loop p99 {p99['loop_dt']:.1f} ms | state age p99 {p99['state_age']:.1f} ms"
                f" | step p99 {p99['compute']:.2f} ms | max torque asked {100 * r['tau_ratio'].max():.0f} %")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True, help="folder written by go2_export.py (policy.pt, deploy_cfg.json)")
    p.add_argument("--net", type=str, default=None, help="network interface wired to the robot, e.g. enp3s0")
    p.add_argument("--domain", type=int, default=0, help="DDS domain id (robot 0, unitree_mujoco 1)")
    p.add_argument("--sim", action="store_true", help="unitree_mujoco: domain 1, interface lo, no sport mode")
    p.add_argument("--rehearsal", action="store_true",
                   help="[S2R] the robot's code path (sport-mode release, no --scenario, Enter prompt) against "
                        "go2_mujoco_headless.py --robot-services: domain 1 on lo, log marked 'rehearsal'")
    # [S2R]
    p.add_argument("--zero-action", action="store_true", help="policy mode holds the default pose (no network)")
    p.add_argument("--profile", choices=list(PROFILES), default=None, help="command profile started by L1+X")
    p.add_argument("--scenario", choices=list(SCENARIOS), default=None,
                   help="simulators only: scripted button sequence, then exit")
    p.add_argument("--keys", action="store_true", help="PC keyboard as a second remote (space / 0 = damping)")
    p.add_argument("--allow-joint-test", action="store_true", help="enable L1+B joint test: robot hung only")
    p.add_argument("--duration", type=float, default=None, help="exit after this many seconds")
    p.add_argument("--yes", action="store_true", help="simulators only: skip the 'Press Enter' prompt")
    p.add_argument("--log-dir", type=str, default="logs/deploy", help="where the run log goes")
    p.add_argument("--tag", type=str, default="", help="suffix of the log file name")
    p.add_argument("--log-max-s", type=float, default=900.0, help="a longer run keeps its last LOG_MAX_S seconds")
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--rt", action="store_true", help="real-time priority (SCHED_FIFO 50) for the DDS threads and the "
                   "control loop")
    args = p.parse_args()
    if args.rt:  # [S2R] hand the interpreter lock over every 1 ms (default 5 ms): the control thread waits less for it
        sys.setswitchinterval(0.001)
    if args.sim and args.rehearsal:
        p.error("--sim and --rehearsal exclude each other: --rehearsal is the robot path, against the simulator")
    if args.sim or args.rehearsal:
        args.domain, args.net = 1, args.net or "lo"
    if not args.sim and (args.scenario or args.yes):
        p.error("--scenario and --yes are for the simulators (--sim): on the robot you press the buttons yourself")

    import torch

    torch.set_num_threads(1)  # [S2R] one small network: thread hand-offs cost more than they save, and add jitter
    policy = Policy(args.policy, zero_action=args.zero_action)
    sdk_idx = [SDK_JOINTS.index(name) for name in policy.cfg["joint_names"]]
    fsm = Go2StateMachine(policy, FsmConfig(allow_joint_test=args.allow_joint_test))
    profile = CommandProfile(args.profile, fsm.num_commands) if args.profile else None
    fsm.load_profile(profile)
    scenario = None
    if args.scenario:
        scenario = Scenario(args.scenario, profile, fsm.joint_test_duration())
        if scenario.uses_joint_test and not args.allow_joint_test:
            p.error(f"scenario {args.scenario} runs the joint test: add --allow-joint-test (and hang the robot)")
        if args.duration is None:
            args.duration = scenario.duration + 1.0
    print_alignment(policy, sdk_idx, args)

    print("WARNING: low-level control. Robot hung on a stand for the first run, clear area, finger on L1+Y.")
    if not args.yes:
        input("Press Enter to continue...")
    if args.rt:  # [S2R] before the SDK starts its threads: they inherit the priority
        set_realtime()
    if args.net:
        ChannelFactoryInitialize(args.domain, args.net)
    else:
        ChannelFactoryInitialize(args.domain)
    if not args.sim:
        release_sport_mode()

    recorder = None
    if not args.no_log:  # [S2R]
        source = "mujoco" if args.sim else "rehearsal" if args.rehearsal else "robot"
        recorder = go2_log.Recorder(
            go2_log.default_path(args.log_dir, source, args.tag), source, max_s=args.log_max_s,
            ctrl_hz=1.0 / CONTROL_DT, pol_hz=1.0 / policy.cfg["policy_dt"], n_obs=policy.n_obs,
            n_cmd=fsm.num_commands, truth_hz=250.0 if args.sim else None,
            meta={"deploy_cfg": policy.cfg, "fsm_cfg": fsm.config_dict(), "args": vars(args), "sdk_idx": sdk_idx,
                  "policy_dir": os.path.abspath(args.policy), "zero_action": args.zero_action,
                  "control_dt": CONTROL_DT, "warmup_s": policy.warmup_s,
                  "profile": profile.describe() if profile else None, "profile_name": args.profile,
                  "scenario": args.scenario, "scenario_presses": scenario.presses if scenario else None,
                  "joint_test": fsm.joint_test.describe(), "modes": [m.value for m in MODES],
                  "truth_kind": "sportmodestate" if args.sim else None,
                  "versions": {"numpy": np.__version__, "torch": torch.__version__}})
    keyboard = Keyboard(fsm.cfg.combos) if args.keys else None
    node = Go2Deploy(fsm, sdk_idx, recorder, scenario, keyboard)
    if args.sim:  # [S2R] the simulator's ground truth, for go2_analyze.py
        sport = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        sport.Init(node.on_sport, 0)
    # [S2R] Objects created so far (torch, the SDK, the policy) are never freed: moving them out of the garbage
    # collector's reach makes its full passes short (they can otherwise stall the loop by tens of ms)
    gc.collect()
    gc.freeze()
    thread = RecurrentThread(interval=CONTROL_DT, target=node.step, name="go2_fsm")
    thread.Start()
    signal.signal(signal.SIGTERM, _raise_interrupt)  # [S2R] kill <pid> takes the Ctrl+C path (damping, log saved)
    print("Running in damping mode. L1+R2 stand up | L1+A policy | L1+R1 sit down | L1+Y damping | Ctrl+C quit"
          + (" | L1+X profile" if profile else "") + (" | L1+B joint test" if args.allow_joint_test else ""))
    if keyboard:
        print("Keys: 1 stand up | 2 policy | 3 sit down | 4 profile | 5 joint test | space or 0 damping")
    t_start = time.monotonic()
    try:
        while True:
            time.sleep(1.0)
            if node.state is None:
                print("waiting for rt/lowstate ...")
            else:
                line = node.status()
                if line:
                    print(f"[{time.monotonic() - node.t0:7.1f} s] {line}")
            if args.duration is not None and time.monotonic() - t_start > args.duration:
                raise KeyboardInterrupt
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # a second Ctrl+C must not cut the damping phase short
        node.stop_requested = True
        time.sleep(1.0)  # keep sending damping for 1 s
        thread.Wait(0.1)
        if keyboard:
            keyboard.close()
        print("stopped in damping mode (the robot is limp: restart sport mode from the app or power-cycle)")
    finally:
        if recorder is not None:  # [S2R]
            recorder.close()  # the control thread keeps sending damping: stop it writing to the log first
            recorder.events.extend([float(t), msg] for t, msg in fsm.events)
            path = recorder.save(extra_meta={"clamp_count": fsm.clamp_count, "loop_error": node.error,
                                             "n_states": node.n_states})
            print(f"log saved to {path}  (python go2_analyze.py {path})")


def _raise_interrupt(*_):
    raise KeyboardInterrupt


def print_alignment(policy, sdk_idx, args):
    """[S2R] What this run will feed the policy: compare with the training config (H4 checklist)."""
    c = policy.cfg
    src = c.get("source", {})
    kind = "zero action (no network)" if args.zero_action else ("recurrent" if c.get("recurrent") else "MLP")
    print(f"policy {src.get('exp_name', '?')} ckpt {src.get('ckpt', '?')} ({kind}), {policy.n_obs} observations, "
          f"policy every {1000 * c['policy_dt']:.0f} ms, network warm-up {1000 * policy.warmup_s:.0f} ms")
    print(f"  kp {c['kp']} kd {c['kd']} | action scale {c['action_scale']} clip {c['clip_actions']} | obs scales "
          f"{json.dumps(c['obs_scales'])} | commands scale {c['commands_scale']}")
    print(f"  command ranges {json.dumps(c['command_ranges'])} | deadzones lin {c['lin_deadzone']} ang "
          f"{c['ang_deadzone']}")
    print("  joint -> SDK motor | default [rad]: " + ", ".join(
        f"{n.replace('_joint', '')}->{j} {d:+.2f}" for n, j, d in zip(c["joint_names"], sdk_idx,
                                                                     c["default_joint_angles"])))


if __name__ == "__main__":
    main()
