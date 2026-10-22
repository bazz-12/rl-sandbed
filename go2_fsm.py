"""
go2_fsm.py - [P1.5] Deploy-side state machine for the Go2: damping -> stand up -> policy -> sit down.

The state machine turns what the robot reports into one motor command per joint. The robot reports time, joint
positions and velocities, IMU, and gamepad buttons and sticks. Each motor command is (q, dq, kp, kd, tau), and the
motor drivers apply
    tau_motor = kp * (q - q_meas) + kd * (dq - dq_meas) + tau
Everything is in POLICY joint order (env_cfg["joint_names"]). The same class drives the real robot
(go2_deploy_sdk2.py), unitree_mujoco (go2_deploy_sdk2.py --sim) and the Genesis test bench (go2_fsm_sim.py).

Imports numpy and torch only (torch for the TorchScript policy). Copy go2_fsm.py, go2_commands.py, go2_inputs.py and the
logs/<exp>/exported/ folder (go2_export.py) to the robot computer.

Modes
    PASSIVE   damping, kp = 0, kd = 3 (unitree_rl_lab Go2 "Passive"). Start mode and software e-stop.
    STAND_UP  minimum-jerk interpolation with moderate gains (unitree_rl_lab Go2 "FixStand": kp 60/80/80,
              kd 5/4/4). From PASSIVE: current pose -> lying pose (1 s, folds the legs under the body) -> default
              pose (1 s), as unitree_rl_lab. From POLICY: current pose -> default pose (1 s). Then holds.
    POLICY    hand-over: gains ramp from the stand gains to the policy gains (kp 20, kd 0.5) over 0.5 s at the
              default pose, then the policy runs at 50 Hz. Damping if the base tilts > 1 rad (unitree_rl_lab
              bad_orientation) or a joint reads far outside its range.
    SIT_DOWN  current pose -> lying pose in 2 s with the stand gains, hold 0.5 s, then PASSIVE.
    JOINT_TEST [S2R] robot hung only (FsmConfig.allow_joint_test): from STAND_UP, gains ramp to the policy gains, the
              go2_inputs.JointTest plan runs (steps and a chirp around the default pose), gains ramp back, STAND_UP.

Gamepad (go2_deploy convention, L1 is the modifier for every mode change)
    L1+R2 stand up    L1+A policy (only once standing)    L1+R1 sit down    L1+Y damping (software e-stop)
    Policy mode: left stick = forward / sideways speed, right stick (x) = yaw rate (unitree_rl_gym deploy_real).
    [P1.6] Policies trained with the body-height command: D-pad up / down = body height +-1 cm, clamped to the
    trained range, back to 0 (nominal height) at each policy start.
    [S2R] L1+X (policy mode) starts the command profile loaded with load_profile(); it replaces the sticks until it
    ends or a stick is pushed past go2_inputs.STICK_ABORT. L1+B (standing, joint test allowed) starts the joint test.

[S2R] Safety on every command: position targets clipped to the URDF joint range; damping when a joint is asked for more
than torque_trip_ratio x its motor limit for torque_trip_s (a wrong joint order or sign, a leg fighting an obstacle, an
oscillation), or when the policy output is not finite.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from enum import Enum

import numpy as np

from go2_commands import apply_command_deadzone, sticks_to_command
from go2_inputs import STICK_ABORT, JointTest

# Lying pose per leg (hip, thigh, calf) [rad]: unitree_rl_lab Go2 FixStand qs[1] (= go2_stand_example _targetPos_1
# without the +-0.2 rad rear-hip offsets).
LIE_POSE_LEG = (0.0, 1.36, -2.65)

COMBOS = {"stand_up": ("L1", "R2"), "policy": ("L1", "A"), "sit_down": ("L1", "R1"), "damping": ("L1", "Y"),
          "profile": ("L1", "X"), "joint_test": ("L1", "B")}  # [S2R] profile, joint_test


class Mode(Enum):
    PASSIVE = "passive"
    STAND_UP = "stand_up"
    POLICY = "policy"
    SIT_DOWN = "sit_down"
    JOINT_TEST = "joint_test"  # [S2R]


MODES = list(Mode)  # [S2R] logs store the mode as its index in this list


@dataclass
class FsmConfig:
    stand_kp: tuple = (60.0, 80.0, 80.0)  # hip, thigh, calf (unitree_rl_lab Go2 FixStand)
    stand_kd: tuple = (5.0, 4.0, 4.0)
    passive_kd: float = 3.0  # unitree_rl_lab Go2 Passive
    lie_s: float = 1.0  # stand up from PASSIVE, phase 1: current pose -> lying pose
    stand_s: float = 1.0  # phase 2: lying pose -> default pose
    restand_s: float = 1.0  # stand up from POLICY or SIT_DOWN: current pose -> default pose
    ready_hold_s: float = 0.5  # L1+A is accepted after this long at the default pose
    handover_s: float = 0.5  # policy entry: gain ramp stand -> policy at the default pose (0 = switch directly)
    sit_s: float = 2.0  # current pose -> lying pose
    sit_hold_s: float = 0.5  # then hold before PASSIVE
    max_tilt_rad: float = 1.0  # POLICY -> PASSIVE beyond this base tilt
    entry_max_tilt_rad: float = 0.35  # [S2R] L1+A refused above this base tilt (20 deg) after the stand-up: a reversed
    # motor or a leg on an obstacle leaves the robot crooked. On a slope the base tilts with the ground: raise it there.
    joint_limit_margin: float = 0.2  # POLICY -> PASSIVE when a joint reads this far outside its URDF range [rad]
    height_step: float = 0.01  # [P1.6] body-height change per D-pad press [m]
    clamp_targets: bool = True  # [S2R] clip every position target to the URDF joint range
    torque_trip_ratio: float = 1.5  # [S2R] any mode but PASSIVE -> PASSIVE when |PD torque asked| > ratio x limit ...
    torque_trip_s: float = 0.1  # ... on any joint for this long (ratio 0 = off)
    allow_joint_test: bool = False  # [S2R] L1+B joint test: only with the robot hung, feet clear of the ground
    joint_test: dict = field(default_factory=dict)  # [S2R] overrides of go2_inputs.JOINT_TEST
    combos: dict = field(default_factory=lambda: dict(COMBOS))


def projected_gravity(quat_wxyz):
    """Gravity direction in the base frame, R^T (0, 0, -1), for a (w, x, y, z) base orientation.
    Same as Go2Env: transform_by_quat(global_gravity, inv_quat(base_quat)); same as unitree_rl_gym."""
    w, x, y, z = quat_wxyz
    return np.array([2.0 * (w * y - x * z), -2.0 * (y * z + w * x), -(1.0 - 2.0 * (x * x + y * y))])


class MinJerkPath:
    """Joint trajectory through waypoints, minimum-jerk on each segment (zero velocity and acceleration at the
    waypoints). sample(t) -> (q, dq), held at the last waypoint after the end."""

    def __init__(self, waypoints, durations):
        self.q = [np.asarray(w, dtype=float) for w in waypoints]
        self.T = list(durations)
        self.duration = float(sum(self.T))

    def sample(self, t):
        for qa, qb, T in zip(self.q[:-1], self.q[1:], self.T):
            if t < T:
                s = t / T
                pos = 10 * s**3 - 15 * s**4 + 6 * s**5
                vel = (30 * s**2 - 60 * s**3 + 30 * s**4) / T
                return qa + (qb - qa) * pos, (qb - qa) * vel
            t -= T
        return self.q[-1].copy(), np.zeros_like(self.q[-1])


class Policy:
    """Exported policy (go2_export.py): TorchScript network + deploy_cfg.json.

    Builds the observation exactly as Go2Env._update_observation() (go2_fsm_sim.py checks it against the env):
    base angular velocity, projected gravity, command, joint positions - default, joint velocities, last action.
    policy_dir=None -> zero actions, i.e. hold the default pose with the policy gains (bench tests only).
    [S2R] zero_action=True reads deploy_cfg.json from policy_dir but not the network (first runs on the robot).
    """

    def __init__(self, policy_dir=None, deploy_cfg=None, zero_action=False, warmup=50):
        if deploy_cfg is None:
            with open(os.path.join(policy_dir, "deploy_cfg.json")) as f:
                deploy_cfg = json.load(f)
        self.cfg = deploy_cfg
        self.net = None
        n = len(deploy_cfg["joint_names"])
        self.n_obs = 3 + 3 + int(deploy_cfg.get("num_commands", 3)) + 3 * n  # [S2R]
        self.warmup_s = 0.0
        if policy_dir is not None and not zero_action:
            import torch

            self._torch = torch
            self.net = torch.jit.load(os.path.join(policy_dir, "policy.pt"), map_location="cpu").eval()
            # [S2R] The first calls of a TorchScript module are slow (its profiling executor optimizes it during the
            # first runs: 2-6 ms each on a laptop CPU against 0.1-0.3 ms after): make them now, not on the robot.
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(warmup):
                    self.net(torch.zeros(1, self.n_obs))
            self.warmup_s = time.perf_counter() - t0
            self.net.reset()
        self.default_q = np.asarray(deploy_cfg["default_joint_angles"], dtype=float)
        self.obs_scales = deploy_cfg["obs_scales"]
        self.commands_scale = np.asarray(deploy_cfg["commands_scale"], dtype=float)
        self.action_scale = deploy_cfg["action_scale"]
        self.clip_actions = deploy_cfg["clip_actions"]
        self.last_action = np.zeros(n)
        self.last_obs = None
        self.last_infer_s = 0.0  # [S2R] duration of the last network call [s]
        self.steps = 0  # number of policy steps run so far

    def reset(self):
        self.last_action[:] = 0.0
        if self.net is not None:  # [P1.6] clear the memory of a recurrent export (empty reset() for an MLP)
            self.net.reset()

    def observation(self, gyro, quat_wxyz, q, dq, cmd):
        s = self.obs_scales
        return np.concatenate(
            (
                np.asarray(gyro) * s["ang_vel"],
                projected_gravity(quat_wxyz),
                np.asarray(cmd) * self.commands_scale,
                (np.asarray(q) - self.default_q) * s["dof_pos"],
                np.asarray(dq) * s["dof_vel"],
                self.last_action,
            )
        ).astype(np.float32)

    def step(self, gyro, quat_wxyz, q, dq, cmd):
        """One policy step: returns the joint position targets [rad]."""
        self.last_obs = self.observation(gyro, quat_wxyz, q, dq, cmd)
        t0 = time.perf_counter()
        if self.net is None:
            action = np.zeros_like(self.last_action)
        else:
            with self._torch.no_grad():
                action = self.net(self._torch.from_numpy(self.last_obs)[None])[0].numpy().astype(float)
        self.last_infer_s = time.perf_counter() - t0
        if not np.all(np.isfinite(action)):  # [S2R]
            raise FloatingPointError("policy output is not finite")
        # as Go2Env.step(): the clipped action is applied AND fed back as the last-action observation
        self.last_action = np.clip(action, -self.clip_actions, self.clip_actions)
        self.steps += 1
        return self.default_q + self.action_scale * self.last_action


class Go2StateMachine:
    """Call update() at the control rate (500 Hz on the robot, the physics rate in the bench)."""

    def __init__(self, policy: Policy, cfg: FsmConfig | None = None, verbose=True):
        self.policy = policy
        self.cfg = cfg or FsmConfig()
        self.verbose = verbose
        names = policy.cfg["joint_names"]
        self.joint_names = list(names)
        kind = [{"hip": 0, "thigh": 1, "calf": 2}[n.split("_")[1]] for n in names]
        self.stand_kp = np.array([self.cfg.stand_kp[k] for k in kind])
        self.stand_kd = np.array([self.cfg.stand_kd[k] for k in kind])
        self.policy_kp = np.full(len(names), float(policy.cfg["kp"]))
        self.policy_kd = np.full(len(names), float(policy.cfg["kd"]))
        self.stand_pose = policy.default_q.copy()
        self.lie_pose = np.array([LIE_POSE_LEG[k] for k in kind])
        lim = np.asarray(policy.cfg["dof_pos_limits"], dtype=float)  # [12, 2]
        self.q_lo, self.q_hi = lim[:, 0].copy(), lim[:, 1].copy()  # [S2R] target clipping
        self.q_min = lim[:, 0] - self.cfg.joint_limit_margin
        self.q_max = lim[:, 1] + self.cfg.joint_limit_margin
        self.torque_limits = np.asarray(policy.cfg["torque_limits"], dtype=float)  # [S2R]
        ranges = policy.cfg["command_ranges"]
        self.max_cmd = (max(map(abs, ranges["lin_vel_x"])), max(map(abs, ranges["lin_vel_y"])),
                        max(map(abs, ranges["ang_vel"])))
        self.deadzones = (policy.cfg["lin_deadzone"], policy.cfg["ang_deadzone"])
        self.policy_dt = policy.cfg["policy_dt"]
        # [P1.6] exports from before P1.6 have no "num_commands": 3-command policies
        self.num_commands = int(policy.cfg.get("num_commands", 3))
        self.height_range = ranges.get("body_height", (0.0, 0.0))
        self.height_offset = 0.0  # [m] body-height command dh, D-pad in policy mode

        self.mode = Mode.PASSIVE
        self.t_enter = 0.0
        self.path = None
        self.q_target = self.stand_pose.copy()
        self.command = np.zeros(self.num_commands)
        self.next_policy_t = 0.0
        self.policy_running = False
        self.prev_buttons = frozenset()
        self.events = []  # (t, message), for logs
        # [S2R] command profile, joint test, safety state, and what the logs read after each update()
        self.profile = None
        self.profile_t0 = None
        self.profile_seg = -1
        self.joint_test = JointTest(names, self.cfg.joint_test)
        self.policy_stepped = False  # True when the policy ran during the last update()
        self.last_clamped = False
        self.last_tau_cmd = np.zeros(len(names))
        self.clamp_count = 0
        self.torque_over_s = 0.0
        self.t_prev = None

    # --- helpers -----------------------------------------------------------------------------------------------
    def _log(self, t, msg):
        self.events.append((t, msg))
        if self.verbose:
            print(f"[fsm {t:7.3f} s] {msg}")

    def _enter(self, mode, t, q, msg):
        if mode is Mode.STAND_UP:
            if self.mode is Mode.PASSIVE:  # fold the legs under the body first (unitree_rl_lab FixStand)
                self.path = MinJerkPath([q, self.lie_pose, self.stand_pose], [self.cfg.lie_s, self.cfg.stand_s])
            else:
                self.path = MinJerkPath([q, self.stand_pose], [self.cfg.restand_s])
        elif mode is Mode.SIT_DOWN:
            self.path = MinJerkPath([q, self.lie_pose], [self.cfg.sit_s])
        elif mode is Mode.POLICY:
            self.policy_running = False
        if self.profile_t0 is not None and mode is not Mode.POLICY:  # [S2R] leaving policy mode ends the profile
            self._stop_profile(t, f"stopped by {mode.value}")
        self.mode, self.t_enter = mode, t
        self.torque_over_s = 0.0
        self._log(t, f"-> {mode.value} ({msg})")

    def _combo(self, name, buttons):
        keys = self.cfg.combos[name]
        return all(k in buttons for k in keys) and any(k not in self.prev_buttons for k in keys)

    def emergency_damping(self, t, reason):
        """Damping now, whatever the mode (stale robot state, Ctrl+C, ...)."""
        if self.mode is not Mode.PASSIVE:
            if self.profile_t0 is not None:
                self._stop_profile(t, "stopped by damping")
            self.path = None
            self.mode, self.t_enter = Mode.PASSIVE, t
            self._log(t, f"-> passive ({reason})")

    def standing_ready(self, t):
        return self.mode is Mode.STAND_UP and t - self.t_enter >= self.path.duration + self.cfg.ready_hold_s

    def load_profile(self, profile):
        """[S2R] go2_inputs.CommandProfile started by L1+X in policy mode (None: no profile)."""
        if profile is not None and profile.num_commands != self.num_commands:
            raise ValueError(f"profile built for {profile.num_commands} commands, the policy takes {self.num_commands}")
        self.profile = profile

    def _stop_profile(self, t, why):
        self._log(t, f"profile {self.profile.name} ended: {why}")
        self.profile_t0, self.profile_seg = None, -1

    def joint_test_duration(self):
        """[S2R] Gain ramp in + plan + gain ramp out [s] (go2_inputs.Scenario needs it)."""
        return 2.0 * self.cfg.handover_s + self.joint_test.duration

    def _passive(self, q):
        zeros = np.zeros_like(q)
        return q.copy(), zeros, zeros, np.full_like(q, self.cfg.passive_kd), zeros

    def _blend(self, a):
        """Gains a fraction a of the way from the stand gains to the policy gains."""
        return self.stand_kp + a * (self.policy_kp - self.stand_kp), self.stand_kd + a * (self.policy_kd - self.stand_kd)

    # --- main entry ----------------------------------------------------------------------------------------------
    def update(self, t, q, dq, gyro, quat_wxyz, buttons=(), sticks=(0.0, 0.0, 0.0, 0.0)):
        """t [s], q / dq [12] (policy order), gyro [3] base frame [rad/s], quat (w, x, y, z), buttons: names of
        the pressed buttons ("L1", "R2", "A", ...), sticks (lx, ly, rx, ry) in [-1, 1].
        Returns q_des, dq_des, kp, kd, tau (arrays of 12, policy order)."""
        q, dq = np.asarray(q, dtype=float), np.asarray(dq, dtype=float)
        buttons = frozenset(buttons)
        dt = 0.0 if self.t_prev is None else max(t - self.t_prev, 0.0)
        self.t_prev = t
        self.policy_stepped = False

        # 1) gamepad
        if self._combo("damping", buttons) and self.mode is not Mode.PASSIVE:
            self._enter(Mode.PASSIVE, t, q, "L1+Y, software e-stop")
        elif self._combo("stand_up", buttons) and self.mode is not Mode.STAND_UP:
            self._enter(Mode.STAND_UP, t, q, f"L1+R2, from {self.mode.value}")
        elif self._combo("policy", buttons) and self.mode is Mode.STAND_UP:
            tilt = float(np.arccos(np.clip(-projected_gravity(quat_wxyz)[2], -1.0, 1.0)))
            if not self.standing_ready(t):
                self._log(t, "L1+A ignored: stand-up not finished")
            elif tilt > self.cfg.entry_max_tilt_rad:  # [S2R]
                self._log(t, f"L1+A refused: the base tilts {np.degrees(tilt):.0f} deg after the stand-up (limit "
                             f"{np.degrees(self.cfg.entry_max_tilt_rad):.0f}): reversed motor? obstacle? slope? "
                             f"Sit down (L1+R1), check the joints with go2_probe.py --watch joints")
            else:
                self._enter(Mode.POLICY, t, q, "L1+A")
        elif self._combo("sit_down", buttons) and self.mode in (Mode.STAND_UP, Mode.POLICY, Mode.JOINT_TEST):
            self._enter(Mode.SIT_DOWN, t, q, "L1+R1")
        elif self._combo("profile", buttons) and self.mode is Mode.POLICY:  # [S2R]
            if self.profile is None:
                self._log(t, "L1+X ignored: no command profile loaded (--profile)")
            elif not self.policy_running:
                self._log(t, "L1+X ignored: policy hand-over not finished")
            else:
                self.profile_t0 = t
                self._log(t, f"profile {self.profile.name} started ({self.profile.duration:.0f} s)")
        elif self._combo("joint_test", buttons) and self.mode is Mode.STAND_UP:  # [S2R]
            if not self.cfg.allow_joint_test:
                self._log(t, "L1+B ignored: joint test not allowed (robot hung? --allow-joint-test)")
            elif not self.standing_ready(t):
                self._log(t, "L1+B ignored: stand-up not finished")
            else:
                self._enter(Mode.JOINT_TEST, t, q, f"L1+B, {self.joint_test_duration():.1f} s")
        if self.mode is Mode.POLICY and self.num_commands >= 4:  # [P1.6] D-pad: body height, one step per press
            step = (("up" in buttons and "up" not in self.prev_buttons)
                    - ("down" in buttons and "down" not in self.prev_buttons)) * self.cfg.height_step
            if step:
                lo, hi = self.height_range
                new = float(np.clip(self.height_offset + step, lo, hi))
                if new != self.height_offset:
                    self.height_offset = new
                    self._log(t, f"body height offset {100 * new:+.0f} cm")
        self.prev_buttons = buttons

        # 2) automatic transitions
        if self.mode is Mode.SIT_DOWN and t - self.t_enter >= self.path.duration + self.cfg.sit_hold_s:
            self._enter(Mode.PASSIVE, t, q, "sit-down done")
        if self.mode in (Mode.POLICY, Mode.JOINT_TEST):
            g = projected_gravity(quat_wxyz)
            tilt = float(np.arccos(np.clip(-g[2], -1.0, 1.0)))
            if tilt > self.cfg.max_tilt_rad:
                limit = np.degrees(self.cfg.max_tilt_rad)
                self._enter(Mode.PASSIVE, t, q, f"tilt {np.degrees(tilt):.0f} deg > {limit:.0f}")
            elif np.any(q < self.q_min) or np.any(q > self.q_max):
                j = int(np.argmax(np.maximum(self.q_min - q, q - self.q_max)))
                self._enter(Mode.PASSIVE, t, q, f"{self.joint_names[j]} at {q[j]:+.2f} rad, outside its range")

        # 3) motor command of the current mode, 4) safety checks on it
        command = self._command(t, q, dq, gyro, quat_wxyz, sticks)
        return self._safety(t, dt, q, dq, *command)

    def _command(self, t, q, dq, gyro, quat_wxyz, sticks):
        zeros = np.zeros_like(q)
        if self.mode is Mode.PASSIVE:
            return self._passive(q)
        if self.mode in (Mode.STAND_UP, Mode.SIT_DOWN):
            q_des, dq_des = self.path.sample(t - self.t_enter)
            return q_des, dq_des, self.stand_kp, self.stand_kd, zeros
        if self.mode is Mode.JOINT_TEST:  # [S2R] ramp to the policy gains, plan, ramp back, then STAND_UP
            tr, ramp, plan = t - self.t_enter, self.cfg.handover_s, self.joint_test
            if tr < ramp:
                return (self.stand_pose.copy(), zeros, *self._blend(tr / ramp), zeros)
            if tr < ramp + plan.duration:
                return self.stand_pose + plan.offset(tr - ramp), zeros, self.policy_kp, self.policy_kd, zeros
            if tr < 2.0 * ramp + plan.duration:
                return (self.stand_pose.copy(), zeros, *self._blend(1.0 - (tr - ramp - plan.duration) / ramp), zeros)
            self._enter(Mode.STAND_UP, t, q, "joint test done")
            q_des, dq_des = self.path.sample(0.0)
            return q_des, dq_des, self.stand_kp, self.stand_kd, zeros
        # POLICY: gain ramp at the default pose, then the policy at its own rate (targets held in between)
        ramp = 1.0 if self.cfg.handover_s <= 0 else min((t - self.t_enter) / self.cfg.handover_s, 1.0)
        if ramp < 1.0:
            return (self.stand_pose.copy(), zeros, *self._blend(ramp), zeros)
        if not self.policy_running:
            self.policy.reset()  # last action = 0 (and empty memory), as at the start of a training episode
            self.height_offset = 0.0  # [P1.6] nominal height at each policy start
            self.policy_running, self.next_policy_t = True, t
            self._log(t, "policy running")
        if t >= self.next_policy_t - 1e-9:
            self.command = self._policy_command(t, sticks)
            try:
                self.q_target = self.policy.step(gyro, quat_wxyz, q, dq, self.command)
            except FloatingPointError as e:  # [S2R]
                self._enter(Mode.PASSIVE, t, q, str(e))
                return self._passive(q)
            self.policy_stepped = True
            self.next_policy_t += self.policy_dt
            if self.next_policy_t < t:  # the control loop fell behind: do not try to catch up
                self.next_policy_t = t + self.policy_dt
        return self.q_target.copy(), zeros, self.policy_kp, self.policy_kd, zeros

    def _policy_command(self, t, sticks):
        """(vx, vy, wz[, dh]) for this policy step: the running profile, else the sticks."""
        lx, ly, rx, _ = sticks
        if self.profile_t0 is not None:  # [S2R]
            if max(abs(lx), abs(ly), abs(rx)) > STICK_ABORT:
                self._stop_profile(t, "stick moved, the operator takes over")
            else:
                cmd = self.profile.command(t - self.profile_t0)
                if cmd is None:
                    self._stop_profile(t, "done")
                else:
                    self.profile_seg = self.profile.segment(t - self.profile_t0)
                    velocity = apply_command_deadzone(cmd[:3], *self.deadzones)
                    return np.append(velocity, cmd[3]) if self.num_commands >= 4 else velocity
        velocity = sticks_to_command(ly, -lx, -rx, max_cmd=self.max_cmd, lin_deadzone=self.deadzones[0],
                                     ang_deadzone=self.deadzones[1])
        # [P1.6] (vx, vy, wz), plus dh for policies trained with the body-height command
        return np.append(velocity, self.height_offset) if self.num_commands >= 4 else velocity

    def _safety(self, t, dt, q, dq, q_des, dq_des, kp, kd, tau):
        """[S2R] Every command: targets clipped to the joint range; damping after a sustained torque demand beyond
        torque_trip_ratio x the motor limit (a wrong joint mapping or sign shows up this way)."""
        self.last_clamped = False
        if self.mode is not Mode.PASSIVE and self.cfg.clamp_targets:
            clipped = np.clip(q_des, self.q_lo, self.q_hi)
            if np.any(clipped != q_des):
                self.last_clamped = True
                self.clamp_count += 1
            q_des = clipped
        tau_cmd = kp * (q_des - q) + kd * (dq_des - dq) + tau
        self.last_tau_cmd = tau_cmd
        if self.mode is not Mode.PASSIVE and self.cfg.torque_trip_ratio > 0:
            ratio = np.abs(tau_cmd) / self.torque_limits
            self.torque_over_s = self.torque_over_s + dt if np.any(ratio > self.cfg.torque_trip_ratio) else 0.0
            if self.torque_over_s >= self.cfg.torque_trip_s:
                j = int(np.argmax(ratio))
                self._enter(Mode.PASSIVE, t, q, f"{self.joint_names[j]} asked {ratio[j]:.1f} x its torque limit "
                                                f"for {1000 * self.torque_over_s:.0f} ms")
                out = self._passive(q)
                self.last_tau_cmd = out[3] * (0.0 - dq)
                return out
        return q_des, dq_des, kp, kd, tau

    def config_dict(self):
        """[S2R] For log metadata."""
        return asdict(self.cfg)
