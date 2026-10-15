"""
go2_fsm.py - [P1.5] Deploy-side state machine for the Go2: damping -> stand up -> policy -> sit down.

The state machine turns what the robot reports into one motor command per joint. The robot reports time, joint
positions and velocities, IMU, and gamepad buttons and sticks. Each motor command is (q, dq, kp, kd, tau), and the
motor drivers apply
    tau_motor = kp * (q - q_meas) + kd * (dq - dq_meas) + tau
Everything is in POLICY joint order (env_cfg["joint_names"]). The same class drives the real robot
(go2_deploy_sdk2.py) and the Genesis test bench (go2_fsm_sim.py).

Imports numpy and torch only (torch for the TorchScript policy). Copy go2_fsm.py, go2_commands.py and the
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

Gamepad (go2_deploy convention, L1 is the modifier for every mode change)
    L1+R2 stand up    L1+A policy (only once standing)    L1+R1 sit down    L1+Y damping (software e-stop)
    Policy mode: left stick = forward / sideways speed, right stick (x) = yaw rate (unitree_rl_gym deploy_real).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from go2_commands import sticks_to_command

# Lying pose per leg (hip, thigh, calf) [rad]: unitree_rl_lab Go2 FixStand qs[1] (= go2_stand_example _targetPos_1
# without the +-0.2 rad rear-hip offsets).
LIE_POSE_LEG = (0.0, 1.36, -2.65)

COMBOS = {"stand_up": ("L1", "R2"), "policy": ("L1", "A"), "sit_down": ("L1", "R1"), "damping": ("L1", "Y")}


class Mode(Enum):
    PASSIVE = "passive"
    STAND_UP = "stand_up"
    POLICY = "policy"
    SIT_DOWN = "sit_down"


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
    joint_limit_margin: float = 0.2  # POLICY -> PASSIVE when a joint reads this far outside its URDF range [rad]
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
    """

    def __init__(self, policy_dir=None, deploy_cfg=None):
        if deploy_cfg is None:
            with open(os.path.join(policy_dir, "deploy_cfg.json")) as f:
                deploy_cfg = json.load(f)
        self.cfg = deploy_cfg
        self.net = None
        if policy_dir is not None:
            import torch

            self._torch = torch
            self.net = torch.jit.load(os.path.join(policy_dir, "policy.pt"), map_location="cpu").eval()
        n = len(deploy_cfg["joint_names"])
        self.default_q = np.asarray(deploy_cfg["default_joint_angles"], dtype=float)
        self.obs_scales = deploy_cfg["obs_scales"]
        self.commands_scale = np.asarray(deploy_cfg["commands_scale"], dtype=float)
        self.action_scale = deploy_cfg["action_scale"]
        self.clip_actions = deploy_cfg["clip_actions"]
        self.last_action = np.zeros(n)
        self.last_obs = None
        self.steps = 0  # number of policy steps run so far

    def reset(self):
        self.last_action[:] = 0.0

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
        if self.net is None:
            action = np.zeros_like(self.last_action)
        else:
            with self._torch.no_grad():
                action = self.net(self._torch.from_numpy(self.last_obs)[None])[0].numpy().astype(float)
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
        kind = [{"hip": 0, "thigh": 1, "calf": 2}[n.split("_")[1]] for n in names]
        self.stand_kp = np.array([self.cfg.stand_kp[k] for k in kind])
        self.stand_kd = np.array([self.cfg.stand_kd[k] for k in kind])
        self.policy_kp = np.full(len(names), float(policy.cfg["kp"]))
        self.policy_kd = np.full(len(names), float(policy.cfg["kd"]))
        self.stand_pose = policy.default_q.copy()
        self.lie_pose = np.array([LIE_POSE_LEG[k] for k in kind])
        lim = np.asarray(policy.cfg["dof_pos_limits"], dtype=float)  # [12, 2]
        self.q_min = lim[:, 0] - self.cfg.joint_limit_margin
        self.q_max = lim[:, 1] + self.cfg.joint_limit_margin
        ranges = policy.cfg["command_ranges"]
        self.max_cmd = (max(map(abs, ranges["lin_vel_x"])), max(map(abs, ranges["lin_vel_y"])),
                        max(map(abs, ranges["ang_vel"])))
        self.deadzones = (policy.cfg["lin_deadzone"], policy.cfg["ang_deadzone"])
        self.policy_dt = policy.cfg["policy_dt"]

        self.mode = Mode.PASSIVE
        self.t_enter = 0.0
        self.path = None
        self.q_target = self.stand_pose.copy()
        self.command = np.zeros(3)
        self.next_policy_t = 0.0
        self.policy_running = False
        self.prev_buttons = frozenset()
        self.events = []  # (t, message), for logs

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
        self.mode, self.t_enter = mode, t
        self._log(t, f"-> {mode.value} ({msg})")

    def _combo(self, name, buttons):
        keys = self.cfg.combos[name]
        return all(k in buttons for k in keys) and any(k not in self.prev_buttons for k in keys)

    def emergency_damping(self, t, reason):
        """Damping now, whatever the mode (stale robot state, Ctrl+C, ...)."""
        if self.mode is not Mode.PASSIVE:
            self.path = None
            self.mode, self.t_enter = Mode.PASSIVE, t
            self._log(t, f"-> passive ({reason})")

    def standing_ready(self, t):
        return self.mode is Mode.STAND_UP and t - self.t_enter >= self.path.duration + self.cfg.ready_hold_s

    # --- main entry ----------------------------------------------------------------------------------------------
    def update(self, t, q, dq, gyro, quat_wxyz, buttons=(), sticks=(0.0, 0.0, 0.0, 0.0)):
        """t [s], q / dq [12] (policy order), gyro [3] base frame [rad/s], quat (w, x, y, z), buttons: names of
        the pressed buttons ("L1", "R2", "A", ...), sticks (lx, ly, rx, ry) in [-1, 1].
        Returns q_des, dq_des, kp, kd, tau (arrays of 12, policy order)."""
        q, dq = np.asarray(q, dtype=float), np.asarray(dq, dtype=float)
        buttons = frozenset(buttons)

        # 1) gamepad
        if self._combo("damping", buttons) and self.mode is not Mode.PASSIVE:
            self._enter(Mode.PASSIVE, t, q, "L1+Y, software e-stop")
        elif self._combo("stand_up", buttons) and self.mode is not Mode.STAND_UP:
            self._enter(Mode.STAND_UP, t, q, f"L1+R2, from {self.mode.value}")
        elif self._combo("policy", buttons) and self.mode is Mode.STAND_UP:
            if self.standing_ready(t):
                self._enter(Mode.POLICY, t, q, "L1+A")
            else:
                self._log(t, "L1+A ignored: stand-up not finished")
        elif self._combo("sit_down", buttons) and self.mode in (Mode.STAND_UP, Mode.POLICY):
            self._enter(Mode.SIT_DOWN, t, q, "L1+R1")
        self.prev_buttons = buttons

        # 2) automatic transitions
        if self.mode is Mode.SIT_DOWN and t - self.t_enter >= self.path.duration + self.cfg.sit_hold_s:
            self._enter(Mode.PASSIVE, t, q, "sit-down done")
        if self.mode is Mode.POLICY:
            g = projected_gravity(quat_wxyz)
            tilt = float(np.arccos(np.clip(-g[2], -1.0, 1.0)))
            if tilt > self.cfg.max_tilt_rad:
                limit = np.degrees(self.cfg.max_tilt_rad)
                self._enter(Mode.PASSIVE, t, q, f"tilt {np.degrees(tilt):.0f} deg > {limit:.0f}")
            elif np.any(q < self.q_min) or np.any(q > self.q_max):
                self._enter(Mode.PASSIVE, t, q, "joint position outside its range")

        # 3) motor command of the current mode
        zeros = np.zeros_like(q)
        if self.mode is Mode.PASSIVE:
            return q.copy(), zeros, zeros, np.full_like(q, self.cfg.passive_kd), zeros
        if self.mode in (Mode.STAND_UP, Mode.SIT_DOWN):
            q_des, dq_des = self.path.sample(t - self.t_enter)
            return q_des, dq_des, self.stand_kp, self.stand_kd, zeros
        # POLICY: gain ramp at the default pose, then the policy at its own rate (targets held in between)
        ramp = 1.0 if self.cfg.handover_s <= 0 else min((t - self.t_enter) / self.cfg.handover_s, 1.0)
        if ramp < 1.0:
            kp = self.stand_kp + ramp * (self.policy_kp - self.stand_kp)
            kd = self.stand_kd + ramp * (self.policy_kd - self.stand_kd)
            return self.stand_pose.copy(), zeros, kp, kd, zeros
        if not self.policy_running:
            self.policy.reset()  # last action = 0, as at the start of a training episode
            self.policy_running, self.next_policy_t = True, t
            self._log(t, "policy running")
        if t >= self.next_policy_t - 1e-9:
            lx, ly, rx, _ = sticks
            self.command = sticks_to_command(ly, -lx, -rx, max_cmd=self.max_cmd, lin_deadzone=self.deadzones[0],
                                             ang_deadzone=self.deadzones[1])
            self.q_target = self.policy.step(gyro, quat_wxyz, q, dq, self.command)
            self.next_policy_t += self.policy_dt
            if self.next_policy_t < t:  # the control loop fell behind: do not try to catch up
                self.next_policy_t = t + self.policy_dt
        return self.q_target.copy(), zeros, self.policy_kp, self.policy_kd, zeros
