"""
go2_mujoco_headless.py - [S2R] unitree_mujoco without its window, for scripted runs, plus what the stock simulator
cannot do: robot-like imperfections, injected faults, the robot's start-up services, a video.

    python go2_mujoco_headless.py --um ~/unitree_mujoco                          # unitree_mujoco.py, minus the window
    python go2_mujoco_headless.py --um ~/unitree_mujoco --truth logs/deploy/mj_truth.npz   # + ground truth log
    python go2_mujoco_headless.py --um ~/unitree_mujoco --hang 0.6             # base held 0.6 m up: the stand
    python go2_mujoco_headless.py --um ~/unitree_mujoco --remote walk --profile fwd_back   # scripted remote bytes
    python go2_mujoco_headless.py --um ~/unitree_mujoco --noise 1 --cmd-delay-ms 10 --payload 2 --push 0.5
    python go2_mujoco_headless.py --um ~/unitree_mujoco --blackout 12:0.2      # no rt/lowstate for 200 ms at 12 s
    python go2_mujoco_headless.py --um ~/unitree_mujoco --robot-services       # for go2_deploy_sdk2.py --rehearsal
    python go2_mujoco_headless.py --um ~/unitree_mujoco --video run.mp4        # or --viewer, with a display

By default it is unitree_mujoco.py as shipped: the model file, the physics step (config.SIMULATE_DT), the official DDS
bridge (simulate_python/unitree_sdk2py_bridge.py: the torque kp (q* - q) + kd (dq* - dq) + tau of a command is computed
when it arrives and held until the next one; rt/lowstate and rt/sportmodestate go out every physics step), domain 1 on
"lo", the start pose (legs straight: the robot folds onto the floor in the first second). The physics loop keeps an
absolute schedule and prints its real-time factor. Times below are seconds since the first rt/lowcmd.

Physics    --dt S            physics step (unitree_mujoco default 0.005; 0.002 recommended)
           --foot-friction MU  unitree_mujoco's feet have 0.4, below the training range (0.5 - 1.25)
           --payload KG      extra mass on the base, at its centre of mass (training: -1 to +2 kg)
           --com-shift X,Y,Z  base centre of mass moved [m] (training: up to 0.03 per axis)
           --push V          every --push-every s from --push-start s, the base gets a horizontal velocity kick of
                             V m/s in a random direction (training: up to 1 m/s every 15 s)
           --hang Z          base pinned at height Z (pose and velocity reset every step), legs free: the stand
Motors     --pd-per-step     re-evaluate each command's PD law at every physics step, as a motor driver does, instead
                             of holding the torque computed at arrival (the bridge)
           --cmd-delay-ms D  commands take effect D ms after they arrive (training: 0 - 20 ms action latency)
           --motor-strength S  torques and torque limits x S (training: 0.9 - 1.1)
Sensors    --noise K         the training's observation noise x K on rt/lowstate (uniform: gyro 0.2 rad/s, joint
                             angles 0.01 rad, joint speeds 1.5 rad/s, attitude 0.05 rad)
           --state-delay-ms D  rt/lowstate shows the robot as it was D ms earlier
           --drop P          a fraction P of the rt/lowstate messages is lost
Faults     --blackout T:D[,T:D]  no rt/lowstate from T to T + D s (cable, robot or PC stall): the deploy script must
                             go to damping after 50 ms
           --swap A,B        motors A and B wired to each other's joint (e.g. FR_thigh,FR_calf)
           --flip J          motor J turns the wrong way (sign error)
           --offset J:RAD    motor J's encoder reads RAD too much (calibration error)
Operator   --remote SCENARIO  the scripted presses of go2_inputs.SCENARIOS written into rt/lowstate's remote bytes,
                             as the real remote would (needs --profile for "walk")
           --joystick TYPE   a gamepad on this PC as the robot's remote (xbox or switch layout), with unitree_mujoco's
                             own gamepad code: the L1 combos as on the robot
           --robot-services  fake "motion_switcher" and "sport" services: CheckMode answers "normal" until
                             ReleaseMode, StandDown answers, and rt/lowcmd is ignored until the release, as on the
                             robot. go2_deploy_sdk2.py --rehearsal then runs its robot path against this simulator
Output     --truth FILE      ground truth (go2_log.py "truth" stream) and the injected events, for go2_analyze.py
           --video FILE.mp4  a video following the robot, rendered after the run so it never slows the simulation
                             (offscreen OpenGL: MUJOCO_GL=egl on a GPU machine without a display; --video-fps 30)
           --viewer          the MuJoCo viewer window (needs a display)
Run        --duration S, --exit-idle S (stop once rt/lowcmd has stopped for S s), --seed N
"""

import argparse
import collections
import json
import math
import os
import signal
import sys
import threading
import time

import numpy as np

import go2_log
from go2_fsm import COMBOS, FsmConfig
from go2_inputs import PROFILES, SCENARIOS, CommandProfile, JointTest, Scenario, encode_remote

SDK_JOINTS = ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
              "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"]
FOOT_GEOMS = ("FR", "FL", "RR", "RL")  # foot spheres in go2.xml, policy leg order
TORQUE_LIMITS = np.array([23.7, 23.7, 45.43] * 4)  # go2.xml motor ctrlrange, SDK order
# --noise 1: go2_train.py obs_cfg["noise_scales"] (half-widths of uniform noise), applied to what the robot sends
NOISE = {"gyro": 0.2, "q": 0.01, "dq": 1.5, "attitude": 0.05}


# ======================================================================================================================
# Pure helpers (no MuJoCo, no DDS): test_go2_deploy.py checks them
# ======================================================================================================================
def motor_index(token):
    """'FR_calf', 'FR_calf_joint' or '2' -> SDK motor index 0..11."""
    token = token.strip()
    if token.isdigit():
        i = int(token)
    else:
        name = token if token.endswith("_joint") else token + "_joint"
        if name not in SDK_JOINTS:
            raise ValueError(f"unknown joint {token!r}: FR_hip, FR_thigh, FR_calf, FL_hip ... RL_calf or 0..11")
        i = SDK_JOINTS.index(name)
    if not 0 <= i < 12:
        raise ValueError(f"motor index {i} outside 0..11")
    return i


def parse_windows(text):
    """'12:0.2,20:1' -> [(12.0, 12.2), (20.0, 21.0)] (start, end) [s]."""
    out = []
    for part in filter(None, (p.strip() for p in (text or "").split(","))):
        t, d = (float(v) for v in part.split(":"))
        if d <= 0:
            raise ValueError(f"blackout {part!r}: the duration must be positive")
        out.append((t, t + d))
    return out


class MotorMap:
    """Wiring and calibration of the 12 motors: motor m reads and drives joint perm[m], turning with sign[m] (+1 or -1),
    and its encoder reads offset[m] rad too much. The identity on a healthy robot; --swap, --flip and --offset inject
    wiring and calibration faults, to check that the safety layer catches them."""

    def __init__(self, swap=None, flip=(), offset=None):
        self.perm = np.arange(12)
        self.sign = np.ones(12)
        self.offset = np.zeros(12)
        if swap:
            a, b = swap
            self.perm[[a, b]] = self.perm[[b, a]]
        for j in flip:
            self.sign[j] = -1.0
        for j, v in (offset or {}).items():
            self.offset[j] = v

    @property
    def identity(self):
        return bool(np.all(self.perm == np.arange(12)) and np.all(self.sign == 1.0) and np.all(self.offset == 0.0))

    def to_motor(self, q_j, dq_j, tau_j):
        """Joint readings (joint order) -> what the motors report (motor order)."""
        q_j, dq_j, tau_j = (np.asarray(x, dtype=float) for x in (q_j, dq_j, tau_j))
        return (self.sign * q_j[self.perm] + self.offset, self.sign * dq_j[self.perm], self.sign * tau_j[self.perm])

    def to_joint(self, tau_m):
        """Motor torques (motor order) -> torques on the joints (joint order)."""
        out = np.zeros(12)
        out[self.perm] = self.sign * np.asarray(tau_m, dtype=float)
        return out

    def describe(self):
        faults = []
        for m in range(12):
            if self.perm[m] != m:
                faults.append(f"motor {SDK_JOINTS[m][:-6]} drives {SDK_JOINTS[self.perm[m]][:-6]}")
            if self.sign[m] < 0:
                faults.append(f"motor {SDK_JOINTS[m][:-6]} turns the wrong way")
            if self.offset[m]:
                faults.append(f"motor {SDK_JOINTS[m][:-6]} reads {self.offset[m]:+.3f} rad off")
        return faults


class Actuators:
    """What an rt/lowcmd does to the joints.
    hold (unitree_mujoco's bridge): the torque kp (q* - q) + kd (dq* - dq) + tau is computed once, when the command takes
        effect, and held until the next one.
    per_step (--pd-per-step): the same law re-evaluated at every physics step from the latest command, as the motor
        driver of a real joint does (its loop runs much faster than 500 Hz).
    Commands take effect delay_s after they arrive; torques and their limits are scaled by strength. Motor order in,
    joint order out (MotorMap)."""

    def __init__(self, motor_map, limits=TORQUE_LIMITS, per_step=False, delay_s=0.0, strength=1.0):
        self.map, self.per_step, self.delay, self.strength = motor_map, per_step, float(delay_s), float(strength)
        self.limits = np.asarray(limits, dtype=float) * self.strength
        self.lock = threading.Lock()
        self.pending = collections.deque()  # (t_due, 5 x 12 array: q, dq, kp, kd, tau in motor order)
        self.cmd = None
        self.tau = np.zeros(12)

    def receive(self, t_sim, cmd):
        """Any thread: a command (5 x 12, motor order) that takes effect at simulation time t_sim + delay."""
        with self.lock:
            self.pending.append((t_sim + self.delay, np.array(cmd, dtype=float)))

    def torques(self, t_sim, q_j, dq_j):
        """Physics thread, before each step: joint torques (joint order), None before the first command."""
        new = None
        with self.lock:
            while self.pending and self.pending[0][0] <= t_sim + 1e-9:
                new = self.pending.popleft()[1]
        if new is not None:
            self.cmd = new
        if self.cmd is None:
            return None
        if new is not None or self.per_step:
            q_m, dq_m, _ = self.map.to_motor(q_j, dq_j, np.zeros(12))
            q_t, dq_t, kp, kd, tau = self.cmd
            with np.errstate(invalid="ignore", over="ignore"):  # "no target" (2.146e9) times kp = 0
                tau_m = (np.where(kp != 0.0, kp * (q_t - q_m), 0.0) + np.where(kd != 0.0, kd * (dq_t - dq_m), 0.0)
                         + tau)
            self.tau = np.clip(self.map.to_joint(tau_m) * self.strength, -self.limits, self.limits)
        return self.tau


class DelayLine:
    """Time-stamped history: get(t) returns the newest value recorded at or before t (the oldest one before that)."""

    def __init__(self, keep_s):
        self.keep = float(keep_s)
        self.items = collections.deque()

    def put(self, t, value):
        self.items.append((t, value))
        while len(self.items) > 1 and self.items[1][0] <= t - self.keep:
            self.items.popleft()

    def get(self, t):
        best = self.items[0][1] if self.items else None
        for ti, v in self.items:
            if ti <= t + 1e-12:
                best = v
            else:
                break
        return best


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def small_rotation(rng, max_angle):
    """Quaternion (w, x, y, z) of a rotation by an angle uniform in [-max_angle, max_angle] about a random axis."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-12
    a = rng.uniform(-max_angle, max_angle)
    return np.concatenate(([math.cos(a / 2)], math.sin(a / 2) * axis))


class StateTransport:
    """Sits between unitree_mujoco's bridge and its rt/lowstate publisher: what sensors and network do to the robot
    state on its way to the PC. Delay, loss, blackouts, wiring faults, noise; the remote bytes go through untouched."""

    def __init__(self, publisher, motor_map, clock, delay_s=0.0, noise=0.0, drop=0.0, blackouts=(), seed=0,
                 on_event=None):
        self.pub, self.map, self.clock = publisher, motor_map, clock  # clock() -> (sim time, s since first command)
        self.delay, self.noise, self.drop = float(delay_s), float(noise), float(drop)
        self.blackouts = list(blackouts)
        self.rng = np.random.default_rng(seed)
        self.history = DelayLine(self.delay + 0.1)
        self.on_event = on_event or (lambda msg: None)
        self.in_blackout = False
        self.n_sent = self.n_dropped = self.n_blacked = 0

    def Write(self, msg):  # the bridge's thread, every physics step
        t_sim, t_cmd = self.clock()
        n = 12
        ms = msg.motor_state
        now = (np.array([ms[i].q for i in range(n)]), np.array([ms[i].dq for i in range(n)]),
               np.array([ms[i].tau_est for i in range(n)]), np.array(msg.imu_state.quaternion, dtype=float),
               np.array(msg.imu_state.gyroscope, dtype=float), np.array(msg.imu_state.accelerometer, dtype=float))
        self.history.put(t_sim, now)
        blacked = t_cmd is not None and any(a <= t_cmd < b for a, b in self.blackouts)
        if blacked != self.in_blackout:
            self.in_blackout = blacked
            self.on_event(f"blackout {'starts' if blacked else 'ends'}: rt/lowstate {'stopped' if blacked else 'back'}")
        if blacked:
            self.n_blacked += 1
            return True
        if self.drop > 0 and self.rng.random() < self.drop:
            self.n_dropped += 1
            return True
        q, dq, tau, quat, gyro, acc = self.history.get(t_sim - self.delay) if self.delay > 0 else now
        q, dq, tau = self.map.to_motor(q, dq, tau)
        if self.noise > 0:
            k, r = self.noise, self.rng
            q = q + r.uniform(-1, 1, n) * NOISE["q"] * k
            dq = dq + r.uniform(-1, 1, n) * NOISE["dq"] * k
            gyro = gyro + r.uniform(-1, 1, 3) * NOISE["gyro"] * k
            quat = quat_mul(quat, small_rotation(r, NOISE["attitude"] * k))
        for i in range(n):
            ms[i].q, ms[i].dq, ms[i].tau_est = float(q[i]), float(dq[i]), float(tau[i])
        msg.imu_state.quaternion = [float(v) for v in quat]
        msg.imu_state.gyroscope = [float(v) for v in gyro]
        msg.imu_state.accelerometer = [float(v) for v in acc]
        self.n_sent += 1
        return self.pub.Write(msg)


def start_robot_services(on_event):
    """Fake versions of the two robot services go2_deploy_sdk2.py talks to before taking over the motors. They answer
    as go2_stand_example.py expects: CheckMode {"name": "normal"} until ReleaseMode, then {"name": ""}."""
    from unitree_sdk2py.comm.motion_switcher import motion_switcher_api as msa
    from unitree_sdk2py.go2.sport import sport_api as spa
    from unitree_sdk2py.rpc.server import Server

    class MotionSwitcher(Server):
        def __init__(self):
            super().__init__(msa.MOTION_SWITCHER_SERVICE_NAME)
            self.mode = "normal"
            self.released = False

        def Init(self):
            self._SetApiVersion(msa.MOTION_SWITCHER_API_VERSION)
            self._RegistHandler(msa.MOTION_SWITCHER_API_ID_CHECK_MODE, self.check_mode, False)
            self._RegistHandler(msa.MOTION_SWITCHER_API_ID_SELECT_MODE, self.select_mode, False)
            self._RegistHandler(msa.MOTION_SWITCHER_API_ID_RELEASE_MODE, self.release_mode, False)

        def check_mode(self, parameter):
            on_event(f"motion_switcher: CheckMode -> {self.mode!r}")
            return 0, json.dumps({"form": "0", "name": self.mode})

        def select_mode(self, parameter):
            self.mode = json.loads(parameter).get("name", "normal")
            self.released = not self.mode
            on_event(f"motion_switcher: SelectMode {self.mode!r}")
            return 0, ""

        def release_mode(self, parameter):
            self.mode, self.released = "", True
            on_event("motion_switcher: ReleaseMode, low-level commands accepted from now on")
            return 0, ""

    class Sport(Server):
        def __init__(self):
            super().__init__(spa.SPORT_SERVICE_NAME)

        def Init(self):
            self._SetApiVersion(spa.SPORT_API_VERSION)
            for api, name in ((spa.SPORT_API_ID_STANDDOWN, "StandDown"), (spa.SPORT_API_ID_DAMP, "Damp")):
                self._RegistHandler(api, lambda parameter, name=name: self.answer(name), False)

        def answer(self, name):
            on_event(f"sport: {name}")
            return 0, ""

    switcher, sport = MotionSwitcher(), Sport()
    for server in (switcher, sport):
        server.Init()
        server.Start(False)
    return switcher


class Video:
    """An mp4 following the robot. While the simulation runs, only the pose is kept (fps per simulated second); the
    frames are rendered after the run, so rendering never slows the real-time loop."""

    def __init__(self, path, model, size=(640, 360), fps=30):
        import cv2
        import mujoco

        self.cv2, self.mujoco, self.model = cv2, mujoco, model
        self.w, self.h = size
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, self.w)
        model.vis.global_.offheight = max(model.vis.global_.offheight, self.h)
        self.renderer = mujoco.Renderer(model, self.h, self.w)  # fails here, before the run, without OpenGL
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = model.body("base_link").id
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 1.8, -20.0, 135.0
        self.path, self.fps, self.next_t, self.poses = path, fps, 0.0, []

    def capture(self, data, label):
        if data.time + 1e-9 >= self.next_t:
            self.next_t += 1.0 / self.fps
            self.poses.append((data.qpos.copy(), label))

    def render(self):
        cv2, mujoco = self.cv2, self.mujoco
        writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (self.w, self.h))
        d = mujoco.MjData(self.model)
        for qpos, label in self.poses:
            d.qpos[:] = qpos
            mujoco.mj_forward(self.model, d)
            self.renderer.update_scene(d, camera=self.cam)
            img = cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)
            cv2.putText(img, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(img)
        writer.release()
        return len(self.poses)


# ======================================================================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--um", type=str, required=True, help="unitree_mujoco checkout (has simulate_python/)")
    p.add_argument("--scene", type=str, default=None, help="default: <um>/unitree_robots/go2/scene.xml")
    p.add_argument("--dt", type=float, default=None, help="physics step [s] (default: config.SIMULATE_DT)")
    p.add_argument("--domain", type=int, default=1)
    p.add_argument("--net", type=str, default="lo")
    p.add_argument("--hang", type=float, default=None, metavar="Z", help="pin the base at height Z [m]")
    p.add_argument("--foot-friction", type=float, default=None, metavar="MU")
    p.add_argument("--payload", type=float, default=0.0, metavar="KG")
    p.add_argument("--com-shift", type=str, default=None, metavar="X,Y,Z")
    p.add_argument("--push", type=float, default=0.0, metavar="V", help="velocity kick [m/s]")
    p.add_argument("--push-every", type=float, default=4.0, metavar="S")
    p.add_argument("--push-start", type=float, default=8.0, metavar="S")
    p.add_argument("--pd-per-step", action="store_true")
    p.add_argument("--cmd-delay-ms", type=float, default=0.0, metavar="D")
    p.add_argument("--motor-strength", type=float, default=1.0, metavar="S")
    p.add_argument("--noise", type=float, default=0.0, metavar="K")
    p.add_argument("--state-delay-ms", type=float, default=0.0, metavar="D")
    p.add_argument("--drop", type=float, default=0.0, metavar="P")
    p.add_argument("--blackout", type=str, default=None, metavar="T:D[,T:D]")
    p.add_argument("--swap", type=str, default=None, metavar="A,B")
    p.add_argument("--flip", type=str, default=None, metavar="J[,J]")
    p.add_argument("--offset", type=str, default=None, metavar="J:RAD[,J:RAD]")
    p.add_argument("--remote", choices=list(SCENARIOS), default=None, help="scripted remote presses")
    p.add_argument("--profile", choices=list(PROFILES), default=None, help="profile duration for --remote")
    p.add_argument("--num-commands", type=int, default=3, help="policy commands (4 with --height-cmd), --remote only")
    p.add_argument("--joystick", choices=("xbox", "switch"), default=None, help="gamepad as the robot's remote")
    p.add_argument("--robot-services", action="store_true")
    p.add_argument("--truth", type=str, default=None, help="ground-truth log file (.npz)")
    p.add_argument("--video", type=str, default=None, metavar="FILE.mp4")
    p.add_argument("--video-size", type=str, default="640x360", metavar="WxH")
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--duration", type=float, default=None, help="exit after this many seconds")
    p.add_argument("--exit-idle", type=float, default=None, metavar="S",
                   help="exit once rt/lowcmd has stopped for S seconds (after it started): scripted runs")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # faults and perturbations, checked before anything starts
    try:
        blackouts = parse_windows(args.blackout)
        swap = tuple(motor_index(t) for t in args.swap.split(",")) if args.swap else None
        if swap is not None and (len(swap) != 2 or swap[0] == swap[1]):
            raise ValueError("--swap takes two different joints, e.g. FR_thigh,FR_calf")
        flip = [motor_index(t) for t in args.flip.split(",")] if args.flip else []
        offset = {}
        for part in filter(None, (args.offset or "").split(",")):
            j, v = part.split(":")
            offset[motor_index(j)] = float(v)
        com_shift = np.array([float(v) for v in args.com_shift.split(",")]) if args.com_shift else np.zeros(3)
        if com_shift.shape != (3,):
            raise ValueError("--com-shift takes X,Y,Z")
        if not 0.0 <= args.drop < 1.0:
            raise ValueError("--drop takes a fraction in [0, 1)")
        if args.motor_strength <= 0:
            raise ValueError("--motor-strength must be positive")
        if args.joystick and args.remote:
            raise ValueError("--joystick and --remote both write the remote bytes: choose one")
    except ValueError as e:
        p.error(str(e))
    motor_map = MotorMap(swap, flip, offset)

    sys.path.insert(0, os.path.join(os.path.abspath(args.um), "simulate_python"))
    import config  # unitree_mujoco's simulate_python/config.py: the bridge reads it at import

    config.ROBOT, config.USE_JOYSTICK, config.ENABLE_ELASTIC_BAND = "go2", 0, False
    import mujoco
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py_bridge import UnitreeSdk2Bridge

    scene = args.scene or os.path.join(args.um, "unitree_robots", "go2", "scene.xml")
    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    model.opt.timestep = args.dt or config.SIMULATE_DT
    dt = model.opt.timestep
    foot_ids = [model.geom(name).id for name in FOOT_GEOMS]
    if args.foot_friction is not None:
        for g in foot_ids:
            model.geom_friction[g, 0] = args.foot_friction
    mu = float(model.geom_friction[foot_ids[0], 0])
    base = model.body("base_link").id
    base_mass = float(model.body_mass[base])
    if args.payload or np.any(com_shift):
        model.body_mass[base] += args.payload
        model.body_ipos[base] += com_shift
        mujoco.mj_setConst(model, data)
    qadr = np.array([model.joint(n).qposadr[0] for n in SDK_JOINTS])
    vadr = np.array([model.joint(n).dofadr[0] for n in SDK_JOINTS])
    act = np.array([model.actuator(n.replace("_joint", "")).id for n in SDK_JOINTS])
    if args.motor_strength > 1.0:  # stronger motors: MuJoCo's own torque clipping (ctrlrange) must not undo it
        model.actuator_ctrlrange[act] *= args.motor_strength
    if args.hang is not None:
        data.qpos[:7] = (0.0, 0.0, args.hang, 1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    pin = data.qpos[:7].copy()

    video = None  # [before any DDS thread starts: fail fast]
    if args.video:
        w, h = (int(v) for v in args.video_size.lower().split("x"))
        try:
            video = Video(args.video, model, (w, h), args.video_fps)
        except Exception as e:  # no OpenGL context
            p.error(f"--video could not render ({type(e).__name__}: {e}). It needs OpenGL: a desktop session, "
                    f"MUJOCO_GL=egl with a GPU, or xvfb-run -a python go2_mujoco_headless.py ...")
    viewer = None
    if args.viewer:
        import mujoco.viewer

        try:
            viewer = mujoco.viewer.launch_passive(model, data)
        except Exception as e:  # no display
            p.error(f"--viewer could not open a window ({type(e).__name__}: {e}): use --video, or unitree_mujoco.py "
                    f"on a desktop")

    custom_motors = args.pd_per_step or args.cmd_delay_ms > 0 or args.motor_strength != 1.0 or not motor_map.identity
    actuators = Actuators(motor_map, TORQUE_LIMITS, args.pd_per_step, args.cmd_delay_ms / 1000.0,
                          args.motor_strength) if custom_motors else None
    transport_needed = (args.noise > 0 or args.state_delay_ms > 0 or args.drop > 0 or blackouts
                        or not motor_map.identity)

    scenario = None
    if args.remote:
        profile = CommandProfile(args.profile, args.num_commands) if args.profile else None
        fsm_cfg = FsmConfig()
        joint_test_s = 2.0 * fsm_cfg.handover_s + JointTest(SDK_JOINTS, fsm_cfg.joint_test).duration
        scenario = Scenario(args.remote, profile, joint_test_s)
        print(f"scripted remote {args.remote!r}: {scenario.duration:.1f} s from the first rt/lowcmd")

    ChannelFactoryInitialize(args.domain, args.net)
    t_start = time.monotonic()
    recorder = None
    if args.truth:
        recorder = go2_log.Recorder(args.truth, "mujoco_truth", max_s=900.0, ctrl_hz=1.0, truth_hz=1.0 / dt,
                                    t0_mono=t_start,
                                    meta={"scene": os.path.abspath(scene), "dt": dt, "foot_friction": mu,
                                          "hang": args.hang, "remote": args.remote, "profile": args.profile,
                                          "joint_names": SDK_JOINTS, "base_mass": base_mass, "payload": args.payload,
                                          "com_shift": com_shift.tolist(), "push": args.push,
                                          "push_every": args.push_every, "push_start": args.push_start,
                                          "pd_per_step": args.pd_per_step, "cmd_delay_ms": args.cmd_delay_ms,
                                          "motor_strength": args.motor_strength, "noise": args.noise,
                                          "state_delay_ms": args.state_delay_ms, "drop": args.drop,
                                          "blackouts": blackouts, "faults": motor_map.describe(),
                                          "robot_services": args.robot_services, "seed": args.seed})

    def event(msg):
        """Something the simulator did on purpose: printed, and kept in the truth file's events."""
        t = time.monotonic() - t_start
        if recorder is not None:
            recorder.event(t, msg)
        print(f"[sim {t:6.1f} s] {msg}")

    state = {"t_first_cmd": None, "n_cmd": 0, "ignored": 0}
    services = start_robot_services(event) if args.robot_services else None

    class Bridge(UnitreeSdk2Bridge):
        """The official bridge, plus the command path (Actuators), the fake sport mode and the scripted remote."""

        def LowCmdHandler(self, msg):
            if state["t_first_cmd"] is None and (services is None or services.released):
                state["t_first_cmd"] = time.monotonic()
            state["n_cmd"] += 1
            if services is not None and not services.released:  # the onboard controller still owns the motors
                state["ignored"] += 1
                if state["ignored"] == 1:
                    event("rt/lowcmd before ReleaseMode: ignored, as the robot's sport mode would")
                return
            if actuators is None:
                super().LowCmdHandler(msg)  # unitree_mujoco's own: torque computed now, held
            else:
                m = msg.motor_cmd
                actuators.receive(data.time, [[m[i].q for i in range(12)], [m[i].dq for i in range(12)],
                                              [m[i].kp for i in range(12)], [m[i].kd for i in range(12)],
                                              [m[i].tau for i in range(12)]])

        def PublishLowState(self):
            if scenario is not None and state["t_first_cmd"] is not None:
                buttons = scenario.buttons_at(time.monotonic() - state["t_first_cmd"], COMBOS)
                self.low_state.wireless_remote = encode_remote(buttons)
            super().PublishLowState()

    def clock():
        t1 = state["t_first_cmd"]
        return data.time, (None if t1 is None else time.monotonic() - t1)

    bridge = Bridge(model, data)
    if args.joystick:
        bridge.SetupJoystick(device_id=0, js_type=args.joystick)  # unitree_mujoco's own code; exits without a gamepad
    transport = None
    if transport_needed:
        transport = StateTransport(bridge.low_state_puber, motor_map, clock, args.state_delay_ms / 1000.0, args.noise,
                                   args.drop, blackouts, args.seed, event)
        bridge.low_state_puber = transport


    settings = [f"physics {1000 * dt:.1f} ms", f"foot friction {mu:g}"]
    settings += [f"base pinned at {args.hang} m"] if args.hang is not None else []
    settings += [f"payload {args.payload:+g} kg"] if args.payload else []
    settings += [f"CoM shift {com_shift.tolist()} m"] if np.any(com_shift) else []
    settings += [f"push {args.push:g} m/s every {args.push_every:g} s from {args.push_start:g} s"] if args.push else []
    settings += ["PD per physics step"] if args.pd_per_step else []
    settings += [f"commands {args.cmd_delay_ms:g} ms late"] if args.cmd_delay_ms else []
    settings += [f"motor strength x{args.motor_strength:g}"] if args.motor_strength != 1.0 else []
    settings += [f"noise x{args.noise:g}"] if args.noise else []
    settings += [f"state {args.state_delay_ms:g} ms late"] if args.state_delay_ms else []
    settings += [f"{100 * args.drop:g} % of states lost"] if args.drop else []
    settings += [f"blackout {a:g}-{b:g} s" for a, b in blackouts]
    settings += motor_map.describe()
    settings += ["fake robot services"] if args.robot_services else []
    print(f"unitree_mujoco headless: {os.path.basename(scene)}, " + ", ".join(settings)
          + f", domain {args.domain} on {args.net}", flush=True)

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    rng = np.random.default_rng(args.seed + 1)
    t_loop = time.monotonic()
    next_t, steps, last_print, last_cmds = t_loop, 0, t_loop, 0
    last_cmd_seen, cmd_count = t_loop, 0
    next_push, push_label_until = args.push_start, -1.0
    next_view = t_loop
    contact_force = np.zeros(6)
    try:
        while args.duration is None or time.monotonic() - t_loop < args.duration:
            _, t_cmd = clock()
            if args.push and t_cmd is not None and t_cmd >= next_push and args.hang is None:
                heading = rng.uniform(0.0, 2.0 * math.pi)
                data.qvel[0] += args.push * math.cos(heading)
                data.qvel[1] += args.push * math.sin(heading)
                event(f"push: {args.push:g} m/s towards {math.degrees(heading):.0f} deg")
                next_push += args.push_every
                push_label_until = t_cmd + 0.5
            if args.hang is not None:
                data.qpos[:7] = pin
                data.qvel[:6] = 0.0
            if actuators is not None:
                tau = actuators.torques(data.time, data.qpos[qadr], data.qvel[vadr])
                data.ctrl[act] = 0.0 if tau is None else tau
            mujoco.mj_step(model, data)
            steps += 1
            if recorder is not None:
                foot_force = np.zeros((4, 3))
                for i in range(data.ncon):
                    c = data.contact[i]
                    for k, g in enumerate(foot_ids):
                        if g in (c.geom1, c.geom2):
                            mujoco.mj_contactForce(model, data, i, contact_force)
                            f_world = c.frame.reshape(3, 3).T @ contact_force[:3]  # contact frame -> world
                            foot_force[k] += f_world if g == c.geom2 else -f_world  # force ON the foot
                recorder.append("truth", t=time.monotonic() - recorder.t0_mono, pos=data.qpos[:3],
                                quat=data.qpos[3:7], lin_vel=data.qvel[:3], ang_vel=data.qvel[3:6],
                                q=data.qpos[qadr], dq=data.qvel[vadr], tau=data.actuator_force[act],
                                foot_force=foot_force, foot_pos=data.geom_xpos[foot_ids])
            if video is not None:
                label = "" if t_cmd is None else (
                    f"t {t_cmd:5.1f} s" + ("  push" if t_cmd < push_label_until else "")
                    + ("  STATE BLACKOUT" if transport is not None and transport.in_blackout else ""))
                video.capture(data, label)
            now = time.monotonic()
            if viewer is not None and now >= next_view:
                if not viewer.is_running():
                    print("viewer closed: done")
                    break
                viewer.sync()
                next_view = now + 1.0 / 60.0
            next_t += dt
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.1:  # more than 100 ms behind: drop the backlog instead of racing to catch up
                next_t = time.monotonic()
            now = time.monotonic()
            if state["n_cmd"] != cmd_count:
                cmd_count, last_cmd_seen = state["n_cmd"], now
            elif args.exit_idle is not None and cmd_count > 0 and now - last_cmd_seen > args.exit_idle:
                print(f"no rt/lowcmd for {args.exit_idle:.0f} s: done")
                break
            if now - last_print >= 2.0:
                rtf = steps * dt / (now - t_loop)
                rate = (state["n_cmd"] - last_cmds) / (now - last_print)
                print(f"[{now - t_start:6.1f} s] real-time factor {rtf:.2f} | base z {data.qpos[2]:.3f} m | "
                      f"rt/lowcmd {rate:.0f} Hz", flush=True)
                last_print, last_cmds = now, state["n_cmd"]
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        rtf = steps * dt / max(time.monotonic() - t_loop, 1e-9)
        print(f"stopped: {steps * dt:.1f} s simulated, real-time factor {rtf:.3f}")
        extra = {"real_time_factor": rtf, "lowcmd_received": state["n_cmd"], "lowcmd_ignored": state["ignored"]}
        if transport is not None:
            extra.update(lowstate_sent=transport.n_sent, lowstate_dropped=transport.n_dropped,
                         lowstate_blacked_out=transport.n_blacked)
            print(f"rt/lowstate: {transport.n_sent} sent, {transport.n_dropped} lost, "
                  f"{transport.n_blacked} withheld by blackouts")
        if video is not None:
            print(f"rendering the video ({len(video.poses)} frames) ...", flush=True)
            print(f"video: {video.render()} frames in {video.path}")
        if recorder is not None:
            print(f"ground truth saved to {recorder.save(extra_meta=extra)}", flush=True)
        os._exit(0)  # the bridge's DDS threads are not daemonic everywhere


if __name__ == "__main__":
    main()
