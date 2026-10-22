"""
go2_inputs.py - [S2R] What drives the state machine (go2_fsm.py) besides the robot state. Shared by the three worlds
(Genesis bench go2_fsm_sim.py, unitree_mujoco, robot) so that they all run exactly the same test.

    parse_remote / encode_remote  the 40-byte wireless_remote field of rt/lowstate (Unitree remote on the robot, a
                                  gamepad in unitree_mujoco): pressed buttons and sticks
    PROFILES, CommandProfile      velocity-command timelines. In policy mode L1+X starts the profile given with
                                  --profile; it replaces the sticks until it ends (then zero command) or until a stick
                                  is pushed past STICK_ABORT (the operator takes over)
    JointTest                     target offsets around the stand pose for the actuator check on the stand (L1+B,
                                  robot hung, feet clear of the ground)
    SCENARIOS, Scenario           scripted button sequences for the simulators (stand up, policy, profile, sit down,
                                  e-stop, joint test), so a sim test needs no gamepad and runs the same way every time

numpy only: copy it to the robot PC with go2_fsm.py and go2_commands.py.
"""

from __future__ import annotations

import math
import struct

import numpy as np

# ======================================================================================================================
# Remote (rt/lowstate wireless_remote)
# ======================================================================================================================
# Bit i of the uint16 at bytes 2-3 (unitree_rl_gym deploy_real/common/remote_controller.py KeyMap; the same order as
# unitree_mujoco's gamepad bridge, simulate_python/unitree_sdk2py_bridge.py)
KEY_BITS = ("R1", "L1", "start", "select", "R2", "L2", "F1", "F2", "A", "B", "X", "Y", "up", "right", "down", "left")
STICK_ABORT = 0.5  # a stick pushed past this (of full scale) stops a running command profile


def parse_remote(data):
    """wireless_remote (40 bytes) -> (pressed buttons, (lx, ly, rx, ry)).

    Layout as unitree_rl_gym remote_controller.py: key bits in the uint16 at bytes 2-3, then floats lx (4-7), rx (8-11),
    ry (12-15), ly (20-23), little-endian. Sticks in [-1, 1]: lx and rx > 0 pushed right, ly and ry > 0 pushed forward.
    """
    raw = bytes(data)
    keys = struct.unpack_from("<H", raw, 2)[0]
    buttons = {name for i, name in enumerate(KEY_BITS) if keys >> i & 1}
    lx, rx, ry = struct.unpack_from("<3f", raw, 4)
    (ly,) = struct.unpack_from("<f", raw, 20)
    return buttons, (lx, ly, rx, ry)


def encode_remote(buttons, sticks=(0.0, 0.0, 0.0, 0.0)):
    """Inverse of parse_remote: -> list of 40 ints (go2_mujoco_headless.py --remote writes it into rt/lowstate)."""
    raw = bytearray(40)
    struct.pack_into("<H", raw, 2, buttons_mask(buttons))
    lx, ly, rx, ry = sticks
    struct.pack_into("<3f", raw, 4, lx, rx, ry)
    struct.pack_into("<f", raw, 20, ly)
    return list(raw)


def buttons_mask(buttons):
    """Set of button names -> uint16 bit mask (KEY_BITS order); the logs store buttons this way."""
    return sum(1 << KEY_BITS.index(b) for b in set(buttons))


def mask_buttons(mask):
    """uint16 bit mask -> set of button names."""
    return {name for i, name in enumerate(KEY_BITS) if int(mask) >> i & 1}


# ======================================================================================================================
# Command profiles
# ======================================================================================================================
# Segments: (duration [s], vx [m/s], vy [m/s], wz [rad/s][, dh [m]]). dh only for policies trained with --height-cmd.
# Values inside the training deadzone (|(vx, vy)| < 0.2 m/s, |wz| < 0.2 rad/s, go2_commands.py) become an exact zero,
# as in training. The profiles meant for the robot go out and come back, so a tethered test ends where it started.
SETTLE_S = 1.5  # go2_analyze.py skips this much at the start of each segment (acceleration transient)

# go2_metrics.py COMMAND_GRID, so a "grid" run gives the same rows as go2_metrics.py (copied: go2_metrics imports Genesis)
METRICS_GRID = [(0.0, 0.0, 0.0), (0.25, 0.0, 0.0), (0.5, 0.0, 0.0), (0.75, 0.0, 0.0), (1.0, 0.0, 0.0),
                (-0.5, 0.0, 0.0), (0.0, 0.3, 0.0), (0.0, -0.3, 0.0), (0.0, 0.0, 0.5), (0.0, 0.0, -0.5),
                (0.5, 0.0, 0.5)]

PROFILES = {
    # H5 stage 2: stand still on the ground for 60 s (drift, joint offsets, torques, heating)
    "stand60": [(60.0, 0.0, 0.0, 0.0)],
    # H5 stage 3, tethered: 0.25 m/s forward for 6 s (1.5 m), stop, back again
    "slow": [(2.0, 0.0, 0.0, 0.0), (6.0, 0.25, 0.0, 0.0), (2.0, 0.0, 0.0, 0.0), (6.0, -0.25, 0.0, 0.0),
             (2.0, 0.0, 0.0, 0.0)],
    # H5 stage 4: one axis at a time, out and back (about 2.5 m of floor for fwd_back)
    "fwd_back": [(2.0, 0.0, 0.0, 0.0), (5.0, 0.5, 0.0, 0.0), (2.0, 0.0, 0.0, 0.0), (5.0, -0.5, 0.0, 0.0),
                 (2.0, 0.0, 0.0, 0.0)],
    "lateral": [(2.0, 0.0, 0.0, 0.0), (5.0, 0.0, 0.3, 0.0), (2.0, 0.0, 0.0, 0.0), (5.0, 0.0, -0.3, 0.0),
                (2.0, 0.0, 0.0, 0.0)],
    "yaw": [(2.0, 0.0, 0.0, 0.0), (5.0, 0.0, 0.0, 0.6), (2.0, 0.0, 0.0, 0.0), (5.0, 0.0, 0.0, -0.6),
            (2.0, 0.0, 0.0, 0.0)],
    # every go2_metrics.py bin for 6 s, 2 s of zero command in between: about 20 m of travel, for the simulators
    "grid": [seg for vx, vy, wz in METRICS_GRID for seg in ((6.0, vx, vy, wz), (2.0, 0.0, 0.0, 0.0))],
    # policies trained with --height-cmd: crouch and rise standing still, then while walking
    "height": [(3.0, 0.0, 0.0, 0.0, 0.0), (5.0, 0.0, 0.0, 0.0, -0.06), (3.0, 0.0, 0.0, 0.0, 0.0),
               (5.0, 0.0, 0.0, 0.0, 0.03), (3.0, 0.0, 0.0, 0.0, 0.0), (5.0, 0.4, 0.0, 0.0, -0.06),
               (3.0, 0.0, 0.0, 0.0, 0.0), (5.0, -0.4, 0.0, 0.0, 0.03), (3.0, 0.0, 0.0, 0.0, 0.0)],
}


class CommandProfile:
    """A velocity-command timeline. command(t) for t in seconds since the profile started; None once it is over."""

    def __init__(self, name, num_commands=3):
        if name not in PROFILES:
            raise ValueError(f"unknown profile {name!r}; available: {', '.join(PROFILES)}")
        self.name = name
        self.num_commands = int(num_commands)
        self.segments = [tuple(float(v) for v in seg) + (0.0,) * (5 - len(seg)) for seg in PROFILES[name]]
        if self.num_commands < 4 and any(seg[4] != 0.0 for seg in self.segments):
            raise ValueError(f"profile {name!r} commands a body height: the policy was trained without --height-cmd")
        self.ends = np.cumsum([seg[0] for seg in self.segments])
        self.duration = float(self.ends[-1])

    def segment(self, t):
        """Index of the segment at time t (seconds since the start), -1 outside the profile."""
        if t < 0.0 or t >= self.duration:
            return -1
        return int(np.searchsorted(self.ends, t, side="right"))

    def command(self, t):
        k = self.segment(t)
        if k < 0:
            return None
        return np.array(self.segments[k][1:1 + self.num_commands], dtype=float)

    def describe(self):
        """For log metadata: [[start, end, vx, vy, wz, dh], ...]."""
        starts = np.concatenate(([0.0], self.ends[:-1]))
        return [[float(a), float(b), *seg[1:]] for a, b, seg in zip(starts, self.ends, self.segments)]


# ======================================================================================================================
# Joint test (actuator check, robot hung)
# ======================================================================================================================
# All 12 joints move around the stand pose with the POLICY gains, dq target 0, as when the policy runs:
#   settle, then for hips, thighs, calves in turn a step of +A, 0, -A, 0 (STEP_S each, the 4 legs together),
#   settle, a logarithmic chirp F0 -> F1 Hz on all joints (cosine taper at both ends), settle.
# The same plan runs in the three worlds, so go2_analyze.py --joint-test can overlay the responses: delay, rise time,
# overshoot, steady-state error and the effective extra delay of one world against another (cross-correlation).
JOINT_TEST = {
    "settle_s": 0.5,
    "step_s": 0.6,
    "step_amp": {"hip": 0.10, "thigh": 0.15, "calf": 0.15},  # [rad]
    "chirp_amp": {"hip": 0.04, "thigh": 0.06, "calf": 0.06},  # [rad]: 0.06 rad at 10 Hz = 3.8 rad/s peak
    "chirp_f0": 0.5,  # [Hz]
    "chirp_f1": 10.0,  # [Hz]
    "chirp_s": 12.0,
    "taper_s": 0.5,
}
KINDS = ("hip", "thigh", "calf")


def joint_kind(name):
    """'FR_thigh_joint' -> 'thigh'."""
    return name.split("_")[1]


class JointTest:
    """Target offsets [rad] (policy joint order) around the stand pose; t in seconds since the plan started."""

    def __init__(self, joint_names, cfg=None):
        self.cfg = dict(JOINT_TEST, **(cfg or {}))
        c = self.cfg
        self.kind = [joint_kind(n) for n in joint_names]
        self.step_amp = np.array([c["step_amp"][k] for k in self.kind])
        self.chirp_amp = np.array([c["chirp_amp"][k] for k in self.kind])
        t = c["settle_s"]
        self.phases = [(0.0, t, "settle")]
        self.steps = []  # (t_start, kind, amplitude) of every target step, for go2_analyze.py
        for kind in KINDS:
            t0 = t
            for amp in (1.0, 0.0, -1.0, 0.0):
                self.steps.append((t, kind, amp * c["step_amp"][kind]))
                t += c["step_s"]
            self.phases.append((t0, t, f"step_{kind}"))
        self.phases.append((t, t + c["settle_s"], "settle"))
        t += c["settle_s"]
        self.chirp_t0 = t
        self.phases.append((t, t + c["chirp_s"], "chirp"))
        t += c["chirp_s"]
        self.phases.append((t, t + c["settle_s"], "settle"))
        self.duration = t + c["settle_s"]

    def phase(self, t):
        for a, b, name in self.phases:
            if a <= t < b:
                return name
        return "done"

    def chirp_frequency(self, tc):
        """Instantaneous chirp frequency [Hz] at tc seconds into the chirp."""
        c = self.cfg
        return c["chirp_f0"] * (c["chirp_f1"] / c["chirp_f0"]) ** (tc / c["chirp_s"])

    def offset(self, t):
        c = self.cfg
        out = np.zeros(len(self.kind))
        # steps: the last step that started at or before t sets the offset of its joint kind
        for t0, kind, amp in self.steps:
            if t0 <= t < t0 + c["step_s"]:
                mask = np.array([k == kind for k in self.kind])
                out[mask] = amp
                return out
        tc = t - self.chirp_t0
        if 0.0 <= tc < c["chirp_s"]:
            T, f0, f1, tp = c["chirp_s"], c["chirp_f0"], c["chirp_f1"], c["taper_s"]
            k = math.log(f1 / f0) / T
            phase = 2.0 * math.pi * f0 * (math.exp(k * tc) - 1.0) / k
            taper = min(1.0, tc / tp, (T - tc) / tp)
            taper = 0.5 - 0.5 * math.cos(math.pi * max(taper, 0.0))
            return self.chirp_amp * taper * math.sin(phase)
        return out

    def describe(self):
        """For log metadata."""
        return {"cfg": self.cfg, "phases": [list(p) for p in self.phases], "steps": [list(s) for s in self.steps],
                "chirp_t0": self.chirp_t0, "duration": self.duration}


# ======================================================================================================================
# Scenarios (simulators only)
# ======================================================================================================================
PRESS_S = 0.1  # a scripted press holds its buttons this long
# Steps: ("wait", s) | ("press", combo) | ("wait_profile",) | ("wait_joint_test",). Combos are go2_fsm.COMBOS names.
# Stand-up from damping takes lie_s + stand_s = 2 s, and L1+A is accepted ready_hold_s = 0.5 s later (go2_fsm.py).
SCENARIOS = {
    "stand": [("wait", 1.0), ("press", "stand_up"), ("wait", 4.5), ("press", "sit_down"), ("wait", 4.0)],
    "policy": [("wait", 1.0), ("press", "stand_up"), ("wait", 3.5), ("press", "policy"), ("wait", 20.0),
               ("press", "sit_down"), ("wait", 4.0)],
    "walk": [("wait", 1.0), ("press", "stand_up"), ("wait", 3.5), ("press", "policy"), ("wait", 3.0),
             ("press", "profile"), ("wait_profile",), ("wait", 2.0), ("press", "sit_down"), ("wait", 4.0)],
    "estop": [("wait", 1.0), ("press", "stand_up"), ("wait", 3.5), ("press", "policy"), ("wait", 4.0),
              ("press", "damping"), ("wait", 4.0)],
    "estop_walk": [("wait", 1.0), ("press", "stand_up"), ("wait", 3.5), ("press", "policy"), ("wait", 2.0),
                   ("press", "profile"), ("wait", 5.0), ("press", "damping"), ("wait", 4.0)],
    "restand": [("wait", 1.0), ("press", "stand_up"), ("wait", 3.5), ("press", "policy"), ("wait", 4.0),
                ("press", "stand_up"), ("wait", 2.0), ("press", "policy"), ("wait", 4.0), ("press", "sit_down"),
                ("wait", 4.0)],
    "joint_test": [("wait", 1.0), ("press", "stand_up"), ("wait", 3.5), ("press", "joint_test"),
                   ("wait_joint_test",), ("wait", 2.0), ("press", "sit_down"), ("wait", 4.0)],
}


class Scenario:
    """A scripted button sequence; t in seconds since the first robot state."""

    def __init__(self, name, profile=None, joint_test_s=None):
        if name not in SCENARIOS:
            raise ValueError(f"unknown scenario {name!r}; available: {', '.join(SCENARIOS)}")
        self.name = name
        self.presses = []  # (t, combo name)
        t = 0.0
        for step in SCENARIOS[name]:
            if step[0] == "wait":
                t += step[1]
            elif step[0] == "press":
                self.presses.append((t, step[1]))
                t += PRESS_S
            elif step[0] == "wait_profile":
                if profile is None:
                    raise ValueError(f"scenario {name!r} starts a command profile: give --profile")
                t += profile.duration
            elif step[0] == "wait_joint_test":
                if joint_test_s is None:
                    raise ValueError(f"scenario {name!r} needs the joint test duration")
                t += joint_test_s
        self.duration = t
        self.uses_profile = any(s[0] == "wait_profile" or s == ("press", "profile") for s in SCENARIOS[name])
        self.uses_joint_test = any(s == ("press", "joint_test") for s in SCENARIOS[name])

    def combos_at(self, t):
        return [combo for t0, combo in self.presses if t0 <= t < t0 + PRESS_S]

    def buttons_at(self, t, combos):
        """Pressed buttons at time t; combos = the state machine's combo -> buttons map (go2_fsm.COMBOS)."""
        return {b for combo in self.combos_at(t) for b in combos[combo]}
