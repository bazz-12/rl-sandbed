"""
go2_probe.py - [S2R] Step 1 on the robot: listen only. It subscribes to rt/lowstate (and rt/sportmodestate) and never
publishes a command, so it is safe in any state of the robot: sport mode standing, lying, damping, held in your hands.

    python go2_probe.py --net enp3s0                       # 60 s log + report
    python go2_probe.py --net enp3s0 --policy logs/go2-p15/exported   # + inference time of your policy
    python go2_probe.py --net enp3s0 --watch joints        # live: which joint moved, and which way
    python go2_probe.py --net enp3s0 --watch imu           # live: gravity from the quaternion and the accelerometer
    python go2_probe.py --net enp3s0 --watch remote        # live: buttons and sticks
    python go2_probe.py --net enp3s0 --watch feet          # live: foot force per foot
    python go2_probe.py --sim                              # unitree_mujoco (domain 1, lo)

What the report checks
- rt/lowstate: rate, gaps, tick steps, arrival jitter, transport-delay variation (arrival time against the tick clock)
- a 500 Hz loop built like go2_deploy_sdk2.py's (RecurrentThread, 2 ms) that only reads: its period jitter, the age of
  the newest state it sees, how often it sees the same state twice; with --policy, the network's inference time on the
  live observations (zero command), as in the deploy loop
- IMU: quaternion norm; gravity from the quaternion against gravity from the accelerometer (catches a w,x,y,z /
  x,y,z,w mix-up); the rpy field against the quaternion; gyro bias and noise
- joints: positions against the URDF limits and against the stand and lie poses (catches a sign convention), noise
- feet: foot_force and foot_force_est per foot; remote: buttons seen, stick ranges; battery, motor temperatures, lost
The log (logs/probe/probe_<date>.npz) keeps every message; go2_analyze.py --probe re-prints the report from it.
"""

import argparse
import json
import os
import sys
import threading
import time

import numpy as np

import go2_log
from go2_fsm import LIE_POSE_LEG, projected_gravity
from go2_inputs import KEY_BITS, buttons_mask, mask_buttons, parse_remote

SDK_JOINTS = ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
              "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"]
SHORT = [n.replace("_joint", "") for n in SDK_JOINTS]
# URDF limits (Genesis go2.urdf = unitree_ros go2_description) [rad]
Q_LIMITS = {"hip": (-1.0472, 1.0472), "front_thigh": (-1.5708, 3.4907), "rear_thigh": (-0.5236, 4.5379),
            "calf": (-2.7227, -0.83776)}
STAND_POSE_LEG = (0.0, 0.67, -1.3)  # go2_stand_example.py _targetPos_2
FEET = ("FR", "FL", "RR", "RL")  # foot_force order (same leg order as the motors)
G = 9.81


def set_realtime(priority=50):
    """--rt: SCHED_FIFO for the calling thread and the threads it starts afterwards: the DDS threads and the read loop
    (as go2_deploy_sdk2.py --rt)."""
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(priority))
        print(f"real-time priority: SCHED_FIFO {priority} for the DDS threads and the read loop")
    except (AttributeError, OSError) as e:
        print(f"real-time priority NOT set ({e}): run as root or raise the rtprio limit")


def joint_limits():
    lo, hi = [], []
    for n in SDK_JOINTS:
        leg, kind = n.split("_")[:2]
        key = kind if kind != "thigh" else ("front_thigh" if leg[0] == "F" else "rear_thigh")
        lo.append(Q_LIMITS[key][0])
        hi.append(Q_LIMITS[key][1])
    return np.array(lo), np.array(hi)


def quat_to_rpy(qwxyz):
    """(w, x, y, z) -> roll, pitch, yaw [rad] (ZYX), rows of qwxyz [N, 4]."""
    w, x, y, z = np.moveaxis(np.asarray(qwxyz, dtype=float), -1, 0)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.stack((roll, pitch, yaw), -1)


def state_fields():
    j = (12,)
    return {"t": ((), "f8"), "tick": ((), "u4"), "q": (j, "f4"), "dq": (j, "f4"), "tau_est": (j, "f4"),
            "temp": (j, "u1"), "lost": (j, "u4"), "quat": ((4,), "f4"), "gyro": ((3,), "f4"), "acc": ((3,), "f4"),
            "rpy": ((3,), "f4"), "imu_temp": ((), "i2"), "foot_force": ((4,), "i2"), "foot_force_est": ((4,), "i2"),
            "buttons": ((), "u2"), "sticks": ((4,), "f4"), "power_v": ((), "f4"), "power_a": ((), "f4"),
            "soc": ((), "u1")}


def loop_fields():
    return {"t": ((), "f8"), "loop_dt": ((), "f4"), "state_age": ((), "f4"), "tick": ((), "u4"), "infer": ((), "f4")}


def sport_fields():
    return {"t": ((), "f8"), "position": ((3,), "f4"), "velocity": ((3,), "f4"), "yaw_speed": ((), "f4"),
            "body_height": ((), "f4"), "foot_force": ((4,), "i2"), "mode": ((), "u1"), "gait_type": ((), "u1")}


class Probe:
    def __init__(self, max_s, policy=None):
        rows = max_s * 1100 + 100  # room for up to 1 kHz messages
        self.state = go2_log.Stream(state_fields(), rows)
        self.loop = go2_log.Stream(loop_fields(), max_s * 520 + 100)
        self.sport = go2_log.Stream(sport_fields(), max_s * 1100 + 100)
        self.lock = threading.Lock()
        self.last, self.last_time, self.last_tick = None, 0.0, None
        self.t0 = time.monotonic()
        self.prev_loop = None
        self.policy = policy
        self.n_loop = 0
        self.last_action = np.zeros(12)

    def on_state(self, msg):
        now = time.monotonic()
        with self.lock:
            self.last, self.last_time = msg, now
        m = msg.motor_state
        buttons, sticks = parse_remote(msg.wireless_remote)
        self.state.append(
            t=now - self.t0, tick=msg.tick, q=[m[j].q for j in range(12)], dq=[m[j].dq for j in range(12)],
            tau_est=[m[j].tau_est for j in range(12)], temp=[m[j].temperature for j in range(12)],
            lost=[m[j].lost for j in range(12)], quat=msg.imu_state.quaternion, gyro=msg.imu_state.gyroscope,
            acc=msg.imu_state.accelerometer, rpy=msg.imu_state.rpy, imu_temp=msg.imu_state.temperature,
            foot_force=msg.foot_force, foot_force_est=msg.foot_force_est, buttons=buttons_mask(buttons),
            sticks=sticks, power_v=msg.power_v, power_a=msg.power_a, soc=msg.bms_state.soc)

    def on_sport(self, msg):
        self.sport.append(t=time.monotonic() - self.t0, position=msg.position, velocity=msg.velocity,
                          yaw_speed=msg.yaw_speed, body_height=msg.body_height, foot_force=msg.foot_force,
                          mode=msg.mode, gait_type=msg.gait_type)

    def loop_step(self):
        """What go2_deploy_sdk2.py does every 2 ms, minus the command: read the newest state (and run the policy
        every 10th step)."""
        start = time.monotonic()
        with self.lock:
            s, st = self.last, self.last_time
        if s is None:
            return
        loop_dt = 0.0 if self.prev_loop is None else start - self.prev_loop
        self.prev_loop = start
        infer = 0.0
        if self.policy is not None and self.n_loop % 10 == 0:
            q = np.array([s.motor_state[j].q for j in range(12)])
            dq = np.array([s.motor_state[j].dq for j in range(12)])
            cmd = np.zeros(int(self.policy.cfg.get("num_commands", 3)))
            self.policy.step(s.imu_state.gyroscope, s.imu_state.quaternion, q, dq, cmd)
            infer = self.policy.last_infer_s
        self.n_loop += 1
        self.loop.append(t=start - self.t0, loop_dt=loop_dt, state_age=start - st, tick=s.tick, infer=infer)


# ======================================================================================================================
# Report
# ======================================================================================================================
def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def report(state, loop, sport, sim=False):
    """Prints the checks; returns a dict of the main numbers (saved in the log)."""
    out, flags = {}, []
    t = state["t"]
    n = len(t)
    print("\n=== rt/lowstate ===")
    if n < 10:
        print("fewer than 10 messages: no robot state. Check --net, the IP (192.168.123.x/24), ping 192.168.123.161.")
        return {"messages": n}
    dur = t[-1] - t[0]
    dt = np.diff(t)
    rate = (n - 1) / dur
    gaps = int(np.sum(dt > 0.010))
    out.update(rate_hz=rate, interval_p50_ms=1e3 * pct(dt, 50), interval_p99_ms=1e3 * pct(dt, 99),
               interval_max_ms=1e3 * dt.max(), gaps_10ms=gaps)
    print(f"{n} messages in {dur:.1f} s -> {rate:.1f} Hz | interval p50 {1e3 * pct(dt, 50):.2f} ms, "
          f"p99 {1e3 * pct(dt, 99):.2f} ms, max {1e3 * dt.max():.1f} ms | gaps > 10 ms: {gaps}")
    if gaps:
        flags.append(f"{gaps} gaps > 10 ms in rt/lowstate (the deploy script goes to damping after 50 ms)")
    tick = state["tick"].astype(np.int64)
    if np.any(tick != tick[0]):
        dtick = np.diff(tick)
        steps, counts = np.unique(dtick, return_counts=True)
        top = ", ".join(f"+{s}: {100 * c / len(dtick):.1f} %" for s, c in sorted(zip(steps, counts), key=lambda x: -x[1])[:4])
        tick_rate = (tick[-1] - tick[0]) / dur
        out["tick_rate_hz"] = tick_rate
        print(f"tick: {tick_rate:.0f} ticks/s; steps between messages {top}")
        # transport-delay variation: arrival time minus the tick clock (assumed tick_rate per second), minus its minimum
        lag = t - (tick - tick[0]) / tick_rate
        lag -= lag.min()
        out.update(delay_var_p50_ms=1e3 * pct(lag, 50), delay_var_p99_ms=1e3 * pct(lag, 99))
        print(f"arrival time against the tick clock (delay variation above its minimum): p50 {1e3 * pct(lag, 50):.2f} ms,"
              f" p99 {1e3 * pct(lag, 99):.2f} ms, max {1e3 * lag.max():.2f} ms")
    else:
        print("tick: constant (unitree_mujoco does not fill it)")

    print("\n=== 500 Hz read loop (as go2_deploy_sdk2.py, no command sent) ===")
    if len(loop["t"]) > 10:
        ldt = loop["loop_dt"][1:]
        age = loop["state_age"]
        over = int(np.sum(ldt > 0.004))
        same = float(np.mean(np.diff(loop["tick"].astype(np.int64)) == 0)) if np.any(tick != tick[0]) else float("nan")
        out.update(loop_p50_ms=1e3 * pct(ldt, 50), loop_p99_ms=1e3 * pct(ldt, 99), loop_max_ms=1e3 * ldt.max(),
                   loop_overruns=over, age_p50_ms=1e3 * pct(age, 50), age_p99_ms=1e3 * pct(age, 99),
                   age_max_ms=1e3 * age.max(), same_state=same)
        print(f"period p50 {1e3 * pct(ldt, 50):.2f} ms, p99 {1e3 * pct(ldt, 99):.2f} ms, max {1e3 * ldt.max():.2f} ms |"
              f" steps longer than 4 ms: {over}")
        print(f"age of the newest state: p50 {1e3 * pct(age, 50):.2f} ms, p99 {1e3 * pct(age, 99):.2f} ms, "
              f"max {1e3 * age.max():.1f} ms" + ("" if np.isnan(same) else f" | same state as the previous step: "
                                                                          f"{100 * same:.1f} %"))
        if pct(ldt, 99) > 0.003:
            flags.append(f"control loop p99 {1e3 * pct(ldt, 99):.1f} ms > 3 ms: close other programs, check CPU "
                         f"frequency scaling (performance governor)")
        infer = loop["infer"][loop["infer"] > 0]
        if len(infer):
            out.update(infer_p50_ms=1e3 * pct(infer, 50), infer_p99_ms=1e3 * pct(infer, 99), infer_max_ms=1e3 * infer.max())
            print(f"policy inference (1 thread): p50 {1e3 * pct(infer, 50):.3f} ms, p99 {1e3 * pct(infer, 99):.3f} ms,"
                  f" max {1e3 * infer.max():.2f} ms")
            if infer.max() > 0.005:
                flags.append(f"policy inference up to {1e3 * infer.max():.1f} ms")

    print("\n=== IMU ===")
    quat, acc, gyro, rpy = (state[k].astype(float) for k in ("quat", "acc", "gyro", "rpy"))
    qn = np.linalg.norm(quat, axis=1)
    g_quat = np.array([projected_gravity(qq) for qq in quat])
    an = np.linalg.norm(acc, axis=1)
    still = np.abs(an - G) < 0.5  # samples where the accelerometer sees about 1 g (robot not accelerating)
    print(f"|quat| {qn.mean():.4f} (min {qn.min():.4f}, max {qn.max():.4f}) | |acc| mean {an.mean():.2f} m/s2")
    if abs(qn.mean() - 1) > 0.01:
        flags.append(f"|quaternion| = {qn.mean():.3f}, not 1")
    if still.sum() > 10:
        g_acc = -acc[still] / an[still, None]  # at rest the accelerometer reads -gravity (specific force)
        ang = np.degrees(np.arccos(np.clip(np.sum(g_acc * g_quat[still], 1), -1, 1)))
        # what the angle would be with the components read as (x, y, z, w)
        g_alt = np.array([projected_gravity(np.roll(qq, 1)) for qq in quat[still]])
        ang_alt = np.degrees(np.arccos(np.clip(np.sum(g_acc * g_alt, 1), -1, 1)))
        out.update(gravity_angle_deg=float(np.median(ang)), gravity_angle_xyzw_deg=float(np.median(ang_alt)))
        print(f"gravity from the quaternion (w,x,y,z) vs from the accelerometer: median {np.median(ang):.1f} deg "
              f"(read as x,y,z,w instead: {np.median(ang_alt):.1f} deg) over {still.sum()} still samples")
        gm = g_quat[still].mean(0)
        print(f"projected gravity mean ({gm[0]:+.3f}, {gm[1]:+.3f}, {gm[2]:+.3f})  "
              f"[level robot: (0, 0, -1); nose down: x > 0; right side down: y < 0]")
        if np.median(ang) > 5:
            flags.append(f"gravity from the quaternion and from the accelerometer differ by {np.median(ang):.0f} deg")
    if np.any(rpy != 0):
        d = quat_to_rpy(quat) - rpy
        d = np.degrees(np.abs((d + np.pi) % (2 * np.pi) - np.pi)).max(0)
        print(f"rpy field vs rpy from the quaternion (ZYX): max diff roll {d[0]:.2f}, pitch {d[1]:.2f}, yaw {d[2]:.2f}"
              f" deg")
    still_gyro = still & (np.linalg.norm(gyro, axis=1) < 0.2)
    if still_gyro.sum() > 10:
        b, s = gyro[still_gyro].mean(0), gyro[still_gyro].std(0)
        out.update(gyro_bias=b.tolist(), gyro_std=s.tolist())
        print(f"gyro at rest: bias ({b[0]:+.4f}, {b[1]:+.4f}, {b[2]:+.4f}) rad/s, noise std ({s[0]:.4f}, {s[1]:.4f}, "
              f"{s[2]:.4f}) rad/s  [training noise: uniform +-0.2 rad/s]")

    print("\n=== joints (SDK order = policy order) ===")
    lo, hi = joint_limits()
    q, dq, tau, temp, lost = (state[k] for k in ("q", "dq", "tau_est", "temp", "lost"))
    print(f"{'joint':9s} {'q mean':>7s} {'q min':>7s} {'q max':>7s}   {'URDF range':>15s} {'dq std':>7s} "
          f"{'tau mean':>8s} {'temp max':>8s} {'lost':>6s}")
    out["joints"] = {}
    for j, name in enumerate(SHORT):
        inside = lo[j] - 0.02 <= q[:, j].min() and q[:, j].max() <= hi[j] + 0.02
        lost_n = int(lost[-1, j]) - int(lost[0, j])
        print(f"{name:9s} {q[:, j].mean():+7.3f} {q[:, j].min():+7.3f} {q[:, j].max():+7.3f}   "
              f"[{lo[j]:+.2f}, {hi[j]:+.2f}]{'' if inside else ' !'} {dq[:, j].std():7.3f} {tau[:, j].mean():+8.2f} "
              f"{int(temp[:, j].max()):8d} {lost_n:6d}")
        out["joints"][name] = {"q_mean": float(q[:, j].mean()), "inside": bool(inside), "temp_max": int(temp[:, j].max())}
        if not inside:
            flags.append(f"{name} outside its URDF range: check the joint order and sign convention")
    qm = q.mean(0)
    for pose_name, leg in (("stand", STAND_POSE_LEG), ("lie", LIE_POSE_LEG)):
        diff = np.abs(qm - np.tile(leg, 4)).max()
        print(f"distance to the {pose_name} pose {leg}: max {diff:.2f} rad")

    print("\n=== feet (foot_force: raw sensor units; FR, FL, RR, RL) ===")
    for key in ("foot_force", "foot_force_est"):
        f = state[key].astype(float)
        if np.any(f != 0):
            print(f"{key:15s} " + " | ".join(f"{name} mean {f[:, i].mean():6.1f} [{f[:, i].min():5.0f}, "
                                            f"{f[:, i].max():5.0f}]" for i, name in enumerate(FEET)))
        else:
            print(f"{key:15s} all zero" + (" (unitree_mujoco does not fill it)" if sim else ""))

    print("\n=== remote ===")
    seen = mask_buttons(np.bitwise_or.reduce(state["buttons"].astype(np.int64)))
    st = state["sticks"]
    print(f"buttons seen: {', '.join(b for b in KEY_BITS if b in seen) or 'none'}")
    print("sticks (min, max): " + " | ".join(f"{name} [{st[:, i].min():+.2f}, {st[:, i].max():+.2f}]"
                                            for i, name in enumerate(("lx", "ly", "rx", "ry"))))

    if np.any(state["power_v"] != 0):
        print(f"\n=== power ===\nbattery {state['power_v'].mean():.1f} V, {state['power_a'].mean():+.2f} A, "
              f"SOC {int(state['soc'][-1])} %")
    if len(sport["t"]) > 1:
        v = sport["velocity"]
        print(f"\n=== rt/sportmodestate ({len(sport['t'])} messages) ===\nvelocity |v| max {np.linalg.norm(v, axis=1).max():.2f}"
              f" m/s, body height {sport['body_height'].mean():.3f} m (only published while sport mode runs)")

    print("\n=== summary ===")
    print("\n".join(f"WARN {f}" for f in flags) if flags else "no warning")
    out["warnings"] = flags
    return out


# ======================================================================================================================
# Live views
# ======================================================================================================================
def watch(probe, what):
    lo, hi = joint_limits()
    q_ref = None
    print({"joints": "Move one joint at a time by hand (robot in damping). The line shows the joint that moved most.\n"
                     "Expected signs: a foot swung forward -> thigh angle DOWN; knee straightened -> calf angle UP "
                     "(towards -0.84);\nleft foot moved outward -> hip UP, right foot moved outward -> hip DOWN. "
                     "Press Enter to reset the reference.",
           "imu": "Tilt the robot. Level: gravity (0, 0, -1). Nose down: x > 0. Right side down: y < 0.\n"
                  "Turning left (counter-clockwise seen from above): gyro z > 0. Nose dipping: gyro y > 0.",
           "remote": "Press buttons, move the sticks: forward = ly > 0, right = lx > 0.",
           "feet": "Lift the robot or press each foot: the matching column must change (FR, FL, RR, RL)."}[what])
    reset = threading.Event()
    if what == "joints" and sys.stdin.isatty():
        threading.Thread(target=lambda: [reset.set() for _ in iter(sys.stdin.readline, "")], daemon=True).start()
    while True:
        time.sleep(0.1)
        with probe.lock:
            s = probe.last
        if s is None:
            print("\rwaiting for rt/lowstate ...", end="", flush=True)
            continue
        if what == "joints":
            q = np.array([s.motor_state[j].q for j in range(12)])
            if q_ref is None or reset.is_set():
                q_ref, _ = q.copy(), reset.clear()
            d = q - q_ref
            j = int(np.argmax(np.abs(d)))
            line = (f"largest change: {SHORT[j]:9s} {d[j]:+.3f} rad (now {q[j]:+.3f}) | q: "
                    + " ".join(f"{v:+.2f}" for v in q))
        elif what == "imu":
            quat = np.array(s.imu_state.quaternion, dtype=float)
            g = projected_gravity(quat)
            a = np.array(s.imu_state.accelerometer, dtype=float)
            ga = -a / max(np.linalg.norm(a), 1e-6)
            w = np.array(s.imu_state.gyroscope)
            line = (f"gravity quat ({g[0]:+.2f} {g[1]:+.2f} {g[2]:+.2f}) acc ({ga[0]:+.2f} {ga[1]:+.2f} {ga[2]:+.2f}) | "
                    f"gyro ({w[0]:+.2f} {w[1]:+.2f} {w[2]:+.2f}) rad/s | rpy from quat "
                    + " ".join(f"{np.degrees(v):+6.1f}" for v in quat_to_rpy(quat)) + " deg")
        elif what == "remote":
            buttons, (lx, ly, rx, ry) = parse_remote(s.wireless_remote)
            line = f"buttons: {' '.join(b for b in KEY_BITS if b in buttons):30s} | lx {lx:+.2f} ly {ly:+.2f} rx {rx:+.2f} ry {ry:+.2f}"
        else:
            line = ("foot_force " + " ".join(f"{n} {v:5d}" for n, v in zip(FEET, s.foot_force))
                    + " | foot_force_est " + " ".join(f"{n} {v:5d}" for n, v in zip(FEET, s.foot_force_est)))
        print("\r" + line + " " * 4, end="", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--net", type=str, default=None, help="network interface wired to the robot, e.g. enp3s0")
    p.add_argument("--domain", type=int, default=0)
    p.add_argument("--sim", action="store_true", help="unitree_mujoco: domain 1, interface lo")
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--watch", choices=("joints", "imu", "remote", "feet"), default=None)
    p.add_argument("--policy", type=str, default=None, help="exported policy folder: also time its inference")
    p.add_argument("--out", type=str, default=None, help="default: logs/probe/probe_<date>.npz")
    p.add_argument("--rt", action="store_true", help="real-time priority, as go2_deploy_sdk2.py --rt")
    args = p.parse_args()
    if args.rt:
        sys.setswitchinterval(0.001)
    if args.sim:
        args.domain, args.net = 1, args.net or "lo"

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_
    from unitree_sdk2py.utils.thread import RecurrentThread

    policy = None
    if args.policy:
        import torch

        from go2_fsm import Policy

        torch.set_num_threads(1)
        policy = Policy(args.policy)
        if policy.cfg["joint_names"] != SDK_JOINTS:
            print("note: the policy joint order differs from the SDK order; inference timing only")
    if args.rt:  # before the SDK starts its threads: they inherit the priority
        set_realtime()
    if args.net:
        ChannelFactoryInitialize(args.domain, args.net)
    else:
        ChannelFactoryInitialize(args.domain)
    probe = Probe(args.seconds + 5, policy)
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(probe.on_state, 0)
    sport_sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    sport_sub.Init(probe.on_sport, 0)
    if args.watch:
        try:
            watch(probe, args.watch)
        except KeyboardInterrupt:
            print()
            os._exit(0)
    loop = RecurrentThread(interval=0.002, target=probe.loop_step, name="probe_loop")
    loop.Start()
    print(f"logging rt/lowstate for {args.seconds:.0f} s (read only, no command is sent) ...")
    try:
        t_end = time.monotonic() + args.seconds
        while time.monotonic() < t_end:
            time.sleep(1.0)
            print(f"\r  {probe.state.n} states, {probe.sport.n} sport states", end="", flush=True)
    except KeyboardInterrupt:
        pass
    print()
    loop.Wait(0.1)
    state, lp, sport = probe.state.arrays(), probe.loop.arrays(), probe.sport.arrays()
    numbers = report(state, lp, sport, sim=args.sim)
    out = args.out or os.path.join("logs", "probe", time.strftime("probe_%Y%m%d_%H%M%S.npz"))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    meta = {"format": go2_log.FORMAT, "source": "probe", "sim": args.sim, "net": args.net, "domain": args.domain,
            "seconds": args.seconds, "joint_names": SDK_JOINTS, "report": numbers,
            "t0_mono": probe.t0, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    arrays = {"meta": np.array(json.dumps(meta, default=go2_log._json_default))}
    for name, stream in (("state", state), ("loop", lp), ("sport", sport)):
        arrays.update({f"{name}.{k}": v for k, v in stream.items()})
    np.savez(out, **arrays)
    print(f"\nlog saved to {out}")
    os._exit(0)


if __name__ == "__main__":
    main()
