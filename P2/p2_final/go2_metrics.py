"""
go2_metrics.py - deterministic evaluation harness for Go2 locomotion policies.

Step 1 (this version): skeleton + velocity tracking error per command bin + falls.
Later metrics are added as new m_*() functions in the METRICS list (see bottom of the
"metrics" section), plus whatever raw signals they need in run_rollout().

Usage (one run = one policy = one table):
    python go2_metrics.py -e go2-walking --ckpt 100
    python go2_metrics.py -e slip-aware  --ckpt 1500
Put it next to go2_env.py (it imports Go2Env from there).
A run of N iterations saves model_0.pt, model_100.pt, ... and model_<N-1>.pt (rsl_rl counts from 0).
[P1.6] Runs trained with the body-height command get the HEIGHT_GRID bins and a dh column.
[P1.6] --tag dr writes metrics_ckpt<N>_seed<S>_dr.csv (and .npz, .png), so a second evaluation of the same
checkpoint (here with --keep_dr) does not overwrite the first. go2_compare.py compares the CSV files.
[P2] Gait quality: air_t and stride (P2.1) in the Gait table, a "Gait pattern & symmetry" table (diag_sync, air_cv,
air_asym, sym_err: P2.3 / P2.4), the mirror-pair check printed after the tables (P2.3), CoT from the 200 Hz motor
power (P2.5). Compare CSV files written by the same version of this script only (re-run the baseline after an
update). go2_analyze.py computes the same gait-pattern columns from deploy logs (Genesis bench, unitree_mujoco, robot).
"""

import argparse
import csv
import os
import pickle
import random

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

import genesis as gs
from go2_env import Go2Env, nominal_world, GO2_FOOT_RADIUS, TROT_PAIRS
from go2_symmetry import mirror_actions, mirror_obs  # [P2.3]


# =============================================================================
# Command grid: one row = one command bin (vx [m/s], vy [m/s], wz [rad/s]).
# (0, 0, 0) must stay in: it is the "stand still" test.
# =============================================================================
COMMAND_GRID = [
    (0.0, 0.0, 0.0),
    (0.25, 0.0, 0.0),
    (0.5, 0.0, 0.0),
    (0.75, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (-0.5, 0.0, 0.0),
    (0.0, 0.3, 0.0),
    (0.0, -0.3, 0.0),
    (0.0, 0.0, 0.5),
    (0.0, 0.0, -0.5),
    (0.5, 0.0, 0.5),
]
# [P1.6] runs trained with the body-height command (4 commands): COMMAND_GRID at dh = 0, plus these bins
# (vx, vy, wz, dh [m]). Keep them inside the run's body_height_range (the ood column flags them otherwise).
HEIGHT_GRID = [
    (0.0, 0.0, 0.0, -0.06),
    (0.0, 0.0, 0.0, 0.03),
    (0.5, 0.0, 0.0, -0.06),
    (0.5, 0.0, 0.0, 0.03),
]

FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]   # order of the foot axis in all foot signals


# =============================================================================
# Environment with pinned commands
# =============================================================================
class EvalGo2Env(Go2Env):
    """Go2Env where env i always receives fixed_commands[i] (never resampled),
    and every reset starts from the default pose + seeded uniform joint noise."""

    def __init__(self, fixed_commands, *args, init_noise=0.0, noise_seed=0, **kwargs):
        # Must exist BEFORE super().__init__(), because __init__ calls reset(),
        # which calls _resample_commands() and _reset_idx().
        self.fixed_commands = fixed_commands
        self.init_noise = init_noise                                  # [rad], 0 = off
        self.noise_gen = torch.Generator(device=gs.device).manual_seed(noise_seed)
        super().__init__(*args, **kwargs)

        # Foot links (needs links_to_keep=FEET when the URDF is loaded, see go2_env.py)
        names = [l.name for l in self.robot.links]
        missing = [f for f in FEET if f not in names]
        if missing:
            raise SystemExit(
                f"Foot links {missing} not found. Add\n"
                f'    links_to_keep=self.env_cfg.get("links_to_keep", ()),\n'
                f"to the gs.morphs.URDF(...) call of the robot in go2_env.py."
            )
        self.feet_idx = [self.robot.get_link(f).idx_local for f in FEET]

    def _resample_commands(self, envs_idx):
        self.commands.copy_(self.fixed_commands)

    def _reset_idx(self, envs_idx=None):
        super()._reset_idx(envs_idx)                  # normal reset to the default pose
        if self.init_noise <= 0:
            return
        # Always draw noise for ALL envs, then use the rows that are being reset:
        # the random stream then advances the same way whatever envs reset.
        noise = (2.0 * torch.rand((self.num_envs, self.num_actions), generator=self.noise_gen,
                                  device=gs.device, dtype=gs.tc_float) - 1.0) * self.init_noise
        idx = (torch.arange(self.num_envs, device=gs.device) if envs_idx is None
               else torch.nonzero(envs_idx).squeeze(-1))
        if idx.numel() == 0:
            return
        q = (self.default_dof_pos + noise)[idx]                        # [n_reset, 12]
        self.robot.set_dofs_position(q, self.motors_dof_idx, envs_idx=idx, zero_velocity=True)
        self.dof_pos[idx] = q                        # keep the observation buffer consistent


# =============================================================================
# Determinism
# =============================================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# Rollout: run the policy, record raw signals. Metrics are computed afterwards.
# =============================================================================
def run_rollout(env, policy, n_warmup, n_measure):
    N = env.num_envs
    dev = gs.device
    alive = torch.ones(N, dtype=torch.bool, device=dev)       # False once the env has fallen
    fall_step = torch.full((N,), -1, dtype=torch.long, device=dev)
    feet = torch.tensor(env.feet_idx, device=dev)
    not_feet = torch.tensor([i for i in range(env.robot.n_links) if i not in env.feet_idx], device=dev)

    # Raw signals, one entry per measured control step -> stacked to [T, N, ...]
    log = {
        "cmd": [],       # [T, N, 3]  commanded (vx, vy, wz)  ([T, N, 4] with dh, P1.6)
        "lin_vel": [],   # [T, N, 3]  base linear velocity, base frame
        "ang_vel": [],   # [T, N, 3]  base angular velocity, base frame
        "valid": [],     # [T, N]     True if the env has not fallen yet
        "q_target": [],  # [T, N, 12] joint position targets sent by the policy [rad]
        "dof_pos": [],   # [T, N, 12] measured joint positions [rad]
        "dof_vel": [],   # [T, N, 12] measured joint velocities [rad/s]
        "torque": [],    # [T, N, 12] torque applied by the PD controller [Nm]
        "torque_sub": [],  # [T, N, D, 12] torques of the D physics steps of each policy step [Nm]  [P1.4]
        "base_pos": [],  # [T, N, 3]  base position, world frame [m]
        "foot_pos": [],  # [T, N, 4, 3] foot sphere centre, world frame [m]
        "foot_vel": [],  # [T, N, 4, 3] foot-sphere CENTRE velocity, world frame [m/s]
        "foot_ang": [],  # [T, N, 4, 3] foot angular velocity, world frame [rad/s]
        "foot_force": [],  # [T, N, 4, 3] net contact force on each foot, world frame [N]
        "other_force": [],  # [T, N]   |contact force| on all NON-foot links summed [N]
        "power": [],     # [T, N]     motor power sum |tau * qdot|, mean over the physics steps [W]  [P2.5]
        "sym_err": [],   # [T, N]     RMS over joints of policy(mirror(obs)) - mirror(policy(obs)) [rad]  [P2.3]
    }
    # [P2.3] the mirrored query must not touch a recurrent policy's memory: MLP policies only
    check_symmetry = not getattr(policy, "is_recurrent", False)
    scale = env.env_cfg["action_scale"]

    obs = env.reset()
    with torch.no_grad():
        for t in range(n_warmup + n_measure):
            actions = policy(obs)
            if t >= n_warmup:  # [P2.3] how far the policy is from its own mirror image, in joint-target radians
                if check_symmetry:
                    mirrored = policy(TensorDict({"policy": mirror_obs(env, obs["policy"])}, batch_size=[N]))
                    err = (mirrored - mirror_actions(env, actions)).pow(2).mean(-1).sqrt() * scale
                else:
                    err = torch.full((N,), float("nan"), device=dev)
                log["sym_err"].append(err)
            obs, _, dones, extras = env.step(actions)
            policy.reset(dones)  # [P1.6] clear the memory of a reset env (recurrent policy; no-op for an MLP)

            # A reset that is not a timeout is a fall (pitch/roll limit or solver error).
            # episode_length_s is set longer than the run, so timeouts should not happen.
            fell = dones.bool() & (extras["time_outs"] == 0)
            fall_step[fell & alive] = t
            alive &= ~fell
            # Note: a fallen env is reset by Go2Env and keeps running, but its data
            # is masked out from here on via "valid".

            if t >= n_warmup:
                log["cmd"].append(env.commands.clone())
                log["lin_vel"].append(env.base_lin_vel.clone())
                log["ang_vel"].append(env.base_ang_vel.clone())
                log["valid"].append(alive.clone())
                log["q_target"].append(env.actions * env.env_cfg["action_scale"] + env.default_dof_pos)
                log["dof_pos"].append(env.dof_pos.clone())
                log["dof_vel"].append(env.dof_vel.clone())
                log["torque"].append(env.robot.get_dofs_control_force(env.motors_dof_idx).clone())
                log["torque_sub"].append(env.torques_substeps.clone())                  # [P1.4]

                # Foot signals: query all links, then keep the 4 feet (works whatever the
                # optional link-index arguments are called in your Genesis version).
                F = env.robot.get_links_net_contact_force()             # [N, n_links, 3]
                log["base_pos"].append(env.base_pos.clone())
                log["foot_pos"].append(env.robot.get_links_pos()[:, feet].clone())
                log["foot_vel"].append(env.robot.get_links_vel()[:, feet].clone())
                log["foot_ang"].append(env.robot.get_links_ang()[:, feet].clone())
                log["foot_force"].append(F[:, feet].clone())
                log["other_force"].append(torch.linalg.norm(F[:, not_feet], dim=-1).sum(-1))
                log["power"].append(env.mech_power.clone())  # [P2.5]

    rec = {k: torch.stack(v).cpu() for k, v in log.items()}
    rec["fall_step"] = fall_step.cpu()
    return rec


# =============================================================================
# Metrics. Each m_*() takes the recording and returns {name: per-env tensor [N]}.
# Per-env values are then averaged per command bin.
# =============================================================================
def masked_mean(x, mask):
    """Mean over time of x [T, N] using mask [T, N]. NaN for envs with no valid step."""
    m = mask.float()
    n = m.sum(0)
    out = (x * m).sum(0) / n.clamp(min=1)
    out[n == 0] = float("nan")
    return out


def m_tracking(rec, dt):
    v = rec["valid"]
    e_lin = torch.linalg.norm(rec["lin_vel"][..., :2] - rec["cmd"][..., :2], dim=-1)  # [T, N]
    e_yaw = (rec["ang_vel"][..., 2] - rec["cmd"][..., 2]).abs()                       # [T, N]
    return {"lin_err[m/s]": masked_mean(e_lin, v), "yaw_err[rad/s]": masked_mean(e_yaw, v)}


def m_measured(rec, dt):
    v = rec["valid"]
    return {
        "vx_meas": masked_mean(rec["lin_vel"][..., 0], v),
        "vy_meas": masked_mean(rec["lin_vel"][..., 1], v),
        "wz_meas": masked_mean(rec["ang_vel"][..., 2], v),
    }


def m_height(rec, dt):
    """[P1.6] Base height and its error against the commanded height: base_height_target + dh (dh = 0 for runs
    without the body-height command, so z_err then measures against base_height_target)."""
    v = rec["valid"]
    z = rec["base_pos"][..., 2]                                                     # [T, N]
    return {"z[m]": masked_mean(z, v), "z_err[cm]": 100.0 * masked_mean((z - rec["z_target"]).abs(), v)}


HF_CUTOFF_HZ = 10.0  # gait ~1-3 Hz + first harmonics stay below this; above = jitter


def m_action_smoothness(rec, dt):
    q = rec["q_target"]                                   # [T, N, 12]
    v = rec["valid"]                                      # [T, N]

    # 1) Action rate: how fast the targets change, RMS over joints, averaged over time.
    dq = (q[1:] - q[:-1]) / dt                            # [T-1, N, 12]  rad/s
    pair_valid = v[1:] & v[:-1]                           # both steps must be valid
    rate = dq.pow(2).mean(-1).sqrt()                      # [T-1, N]
    act_rate = masked_mean(rate, pair_valid)

    # 2) High-frequency content: share of signal power above HF_CUTOFF_HZ.
    #    FFT needs an uninterrupted signal -> only envs that never fell.
    T = q.shape[0]
    x = q - q.mean(0, keepdim=True)                       # remove constant offset
    win = torch.hann_window(T, periodic=False, dtype=q.dtype, device="cpu").view(-1, 1, 1)
    P = torch.fft.rfft(x * win, dim=0).abs().pow(2)       # [F, N, 12]  power spectrum
    f = torch.fft.rfftfreq(T, d=dt, device="cpu")                       # [F]  0 ... 25 Hz
    ratio = P[f >= HF_CUTOFF_HZ].sum(0) / P[1:].sum(0).clamp(min=1e-12)   # [N, 12]
    hf = 100.0 * ratio.mean(-1)                           # [N]  percent, mean over joints
    hf[~v.all(0)] = float("nan")

    # 3) [P1.4] Target acceleration: second difference, i.e. what the action_smoothness reward penalizes.
    d2q = (q[2:] - 2.0 * q[1:-1] + q[:-2]) / dt**2        # [T-2, N, 12]  rad/s^2
    triple_valid = v[2:] & v[1:-1] & v[:-2]
    act_acc = masked_mean(d2q.pow(2).mean(-1).sqrt(), triple_valid)

    return {"act_rate[rad/s]": act_rate, "act_acc[rad/s2]": act_acc, f"hf>{HF_CUTOFF_HZ:.0f}Hz[%]": hf}


def psd(x, fs):
    """[P1.4] One-sided power spectral density of x [T, ...] along time: Hann window, mean removed.
    Returns (f [F], psd [F, ...]) in units^2/Hz, scaled so that sum(psd) * df = mean square of x."""
    T = x.shape[0]
    x = x - x.mean(0, keepdim=True)
    w = torch.hann_window(T, periodic=False, dtype=x.dtype, device="cpu").view(-1, *([1] * (x.ndim - 1)))
    p = torch.fft.rfft(x * w, dim=0).abs().pow(2) / (fs * w.pow(2).sum())
    p[1:] *= 2.0                                          # one-sided: fold the negative frequencies
    if T % 2 == 0:
        p[-1] /= 2.0                                      # ... except the Nyquist bin
    return torch.fft.rfftfreq(T, d=1.0 / fs, device="cpu"), p


def physics_rate_torques(rec):
    """[P1.4] Torques of every physics step as one time series [T*D, N, 12], sampled at D / dt."""
    tau = rec["torque_sub"]                               # [T, N, D, 12]
    T, N, D, J = tau.shape
    return tau.permute(0, 2, 1, 3).reshape(T * D, N, J), D


def m_torque_smoothness(rec, dt):
    """[P1.4] Torques at the physics rate (200 Hz), where PD chatter is visible:
    - tau_rms: RMS torque over joints and time. Motor Joule losses go with current^2, i.e. torque^2.
    - tau>10Hz: RMS of the torque content above HF_CUTOFF_HZ (vibration and heat without useful work).
      Envs that never fell only, as the FFT needs an uninterrupted signal."""
    v = rec["valid"]                                      # [T, N]
    tau, D = physics_rate_torques(rec)                    # [T*D, N, 12]
    vs = v.repeat_interleave(D, dim=0)                    # [T*D, N]
    rms = masked_mean(tau.pow(2).mean(-1), vs).sqrt()     # [N]
    f, p = psd(tau, D / dt)                               # [F, N, 12]
    hf = (p[f >= HF_CUTOFF_HZ].sum(0) * (f[1] - f[0])).mean(-1).sqrt()   # [N]
    hf[~v.all(0)] = float("nan")
    return {"tau_rms[Nm]": rms, f"tau>{HF_CUTOFF_HZ:.0f}Hz[Nm]": hf}


TAU_SAT_FRAC = 0.95   # |tau| >= 95 % of the limit counts as "saturated"
JLIM_MARGIN = 0.02    # [rad] closer than this to a joint limit counts as a "hit"


def m_torque_limits(rec, dt):
    v = rec["valid"]                                       # [T, N]
    vj = v[..., None]                                      # [T, N, 1] broadcast over joints
    n_valid = v.sum(0)                                     # [N]

    # 1) Peak torque as % of the joint's torque limit (worst joint, worst moment).
    tau_ratio = rec["torque"].abs() / rec["tau_lim"]       # [T, N, 12]
    peak = 100.0 * tau_ratio.masked_fill(~vj, 0.0).amax(dim=(0, 2))              # [N]

    # 2) Share of time where at least one joint is saturated.
    sat = (tau_ratio >= TAU_SAT_FRAC).any(-1)              # [T, N]
    sat_time = 100.0 * masked_mean(sat.float(), v)         # [N]

    # 3) Joint-limit hits per minute: count the moments a joint ENTERS the margin zone.
    q = rec["dof_pos"]
    near = ((q <= rec["q_lo"] + JLIM_MARGIN) | (q >= rec["q_hi"] - JLIM_MARGIN)) & vj   # [T, N, 12]
    onsets = near[0].sum(-1) + (near[1:] & ~near[:-1]).sum(dim=(0, 2))                 # [N]
    hits_per_min = onsets.float() / (n_valid * dt / 60.0).clamp(min=1e-9)

    for x in (peak, hits_per_min):
        x[n_valid == 0] = float("nan")
    return {"tau_peak[%lim]": peak, "tau_sat[%time]": sat_time, "jlim_hits[/min]": hits_per_min}


G = 9.81
COT_MIN_SPEED = 0.1   # [m/s] below this commanded planar speed, CoT is not defined


def m_cot(rec, dt):
    v = rec["valid"]
    # Mechanical power: sum over joints of |tau * qdot| (negative work is NOT recovered).
    # [P2.5] averaged over the 200 Hz physics steps by Go2Env (the quantity the energy reward penalizes); before P2.5
    # it was sampled once per policy step, which misses the PD chatter between samples
    if "power" in rec:
        power = rec["power"]                                                       # [T, N]  W
    else:
        power = (rec["torque"] * rec["dof_vel"]).abs().sum(-1)                    # [T, N]  W
    p_mean = masked_mean(power, v)                                                 # [N]
    speed = masked_mean(torch.linalg.norm(rec["lin_vel"][..., :2], dim=-1), v)    # [N]  m/s
    cot = p_mean / (rec["mass"] * G * speed.clamp(min=1e-6))
    cmd_speed = torch.linalg.norm(rec["cmd"][0, :, :2], dim=-1)                    # [N]
    cot[cmd_speed < COT_MIN_SPEED] = float("nan")          # standing / turning in place
    return {"power[W]": p_mean, "speed[m/s]": speed, "CoT[-]": cot}


CONTACT_N = 3.0   # [N] |contact force| above this = in contact (stance forces are ~20-50 N)
FOOT_R = GO2_FOOT_RADIUS   # [m] foot-sphere centre height when the foot is on flat ground (check_feet.py): 0.023


def runs(mask):
    """Start (inclusive) and end (exclusive) indices of each run of True in a 1-D bool array."""
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def foot_contact(rec):
    return torch.linalg.norm(rec["foot_force"], dim=-1) > CONTACT_N       # [T, N, 4]


def swings(rec, dt):
    """[P2.1] Every COMPLETE swing (lift-off and touchdown both inside the valid window) of every env and foot:
    per env a list of (foot, duration [s], peak clearance [m], stride [m]). The stride is the horizontal distance the
    foot travels from its last contact sample to its first contact sample (= stride length of that foot)."""
    c = foot_contact(rec).numpy()                      # [T, N, 4]
    n_ok = rec["valid"].sum(0).numpy()                # valid = a prefix of the window
    pos = rec["foot_pos"].numpy()                      # [T, N, 4, 3]
    out = []
    for n in range(c.shape[1]):
        L, items = n_ok[n], []
        for f in range(4):
            starts, ends = runs(~c[:L, n, f])
            for a, b in zip(starts, ends):
                if a > 0 and b < L:
                    stride = np.linalg.norm(pos[b, n, f, :2] - pos[a - 1, n, f, :2])
                    items.append((f, (b - a) * dt, pos[a:b, n, f, 2].max() - FOOT_R, stride))
        out.append(items)
    return out


def m_gait(rec, dt):
    v = rec["valid"]                                   # [T, N]
    vj = v[..., None]                                  # [T, N, 1]
    c = foot_contact(rec)                              # [T, N, 4]
    n_valid = v.sum(0)                                 # [N]

    # Duty factor: share of time each foot is on the ground, averaged over the 4 feet.
    duty = masked_mean(c.float().mean(-1), v)

    # Step frequency: touchdowns (contact starts) per foot per second.
    # = stride frequency when walking, = "stepping rate" at zero command.
    td = (c[1:] & ~c[:-1] & vj[1:] & vj[:-1]).sum(dim=(0, 2)).float()
    step_freq = td / 4.0 / (n_valid * dt).clamp(min=1e-9)

    # Non-foot contacts: share of time any other link (knee, shin, base...) touches the ground.
    nonfoot = 100.0 * masked_mean((rec["other_force"] > CONTACT_N).float(), v)

    # Means over all complete swings of an env (all feet):
    # - air_t [P2.1]: swing duration, what the feet_air_time reward pays for (the reward detects contact at 1 N, so
    #   the swings it sees can be 1-2 policy steps, 0.02-0.04 s, shorter than this 3 N measure),
    # - stride [P2.1]: horizontal foot travel per swing,
    # - swing_h: highest foot point minus its on-ground height (clearance of the sphere bottom, target of P2.2).
    air_t, stride, swing_h = (np.full(c.shape[1], np.nan) for _ in range(3))
    for n, items in enumerate(swings(rec, dt)):
        if items:
            _, durations, heights, strides = zip(*items)
            air_t[n], swing_h[n], stride[n] = np.mean(durations), np.mean(heights), np.mean(strides)

    for x in (step_freq, duty):
        x[n_valid == 0] = float("nan")
    as_tensor = lambda x: torch.from_numpy(x).float()
    return {"duty[-]": duty, "step_freq[Hz]": step_freq, "air_t[s]": as_tensor(air_t),
            "stride[cm]": 100.0 * as_tensor(stride), "swing_h[cm]": 100.0 * as_tensor(swing_h),
            "nonfoot[%time]": nonfoot}


MOVING_CMD = 0.1   # [P2.4] |(vx, vy, wz)| below this = standing command (Go2Env.zero_cmd)


def m_gait_pattern(rec, dt):
    """[P2.4] Is the gait a regular trot?
    - diag_sync: while at least one foot is in the air, share of time the two feet of each diagonal pair (FL-RR, FR-RL)
      are in the same contact state, mean of both pairs. 100 % = trot (or pronk: check the gait plot), ~50 % = a
      4-beat walk or feet unrelated, ~0 % = pace or bound. n/a at standing commands, and when a foot is in the air
      less than 5 % of the time (no stepping to judge).
    - air_cv: cycle-to-cycle irregularity: coefficient of variation (std / mean) of each foot's swing durations, mean
      over the feet with at least 3 complete swings (front and hind feet may swing for different times in a regular
      gait, so they are not pooled). ~5-15 % for a periodic gait; > 30 % = irregular stepping."""
    v = rec["valid"]
    c = foot_contact(rec)                              # [T, N, 4]
    (a0, a1), (b0, b1) = TROT_PAIRS
    agree = 0.5 * ((c[..., a0] == c[..., a1]).float() + (c[..., b0] == c[..., b1]).float())   # [T, N]
    stepping = v & ~c.all(-1)                          # some foot in the air
    diag = 100.0 * masked_mean(agree, stepping)
    diag[stepping.sum(0) < 0.05 * v.sum(0).clamp(min=1)] = float("nan")
    diag[torch.linalg.norm(rec["cmd"][0, :, :3], dim=-1) < MOVING_CMD] = float("nan")
    cv = np.full(c.shape[1], np.nan)
    for n, items in enumerate(swings(rec, dt)):
        per_foot = [np.array([d for f, d, _, _ in items if f == k]) for k in range(4)]
        cvs = [100.0 * d.std() / d.mean() for d in per_foot if d.size >= 3]
        if cvs:
            cv[n] = np.mean(cvs)
    return {"diag_sync[%]": diag, "air_cv[%]": torch.from_numpy(cv).float()}


def m_symmetry(rec, dt):
    """[P2.3] Left-right symmetry.
    - sym_err: RMS over joints of policy(mirror(obs)) - mirror(policy(obs)) on the visited states, in joint-target
      radians (go2_symmetry.py). What the mirror loss minimizes; augmentation drives it down too. n/a for RNNs.
    - air_asym: mean over the mirror pairs (FL, FR) and (RL, RR) of |mean swing L - mean swing R| / their mean.
      A limp or a favoured side shows here. n/a when a foot made no complete swing."""
    sym = masked_mean(rec["sym_err"], rec["valid"])
    asym = np.full(rec["valid"].shape[1], np.nan)
    for n, items in enumerate(swings(rec, dt)):
        per_foot = [[d for f, d, _, _ in items if f == k] for k in range(4)]
        if all(per_foot):
            m = [np.mean(d) for d in per_foot]          # FL, FR, RL, RR (FEET order)
            front = abs(m[0] - m[1]) / (0.5 * (m[0] + m[1]))
            hind = abs(m[2] - m[3]) / (0.5 * (m[2] + m[3]))
            asym[n] = 100.0 * 0.5 * (front + hind)
    return {"sym_err[rad]": sym, "air_asym[%]": torch.from_numpy(asym).float()}


SPECTRA_BINS = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.5)]  # [P1.4] bins in the plot
SERIES_STYLE = [("#2a78d6", "-"), ("#eb6834", "--"), ("#1baf7a", "-."), ("#eda100", ":")]  # fixed order


def plot_spectra(rec, bin_id, grid, dt, path):
    """[P1.4] Power spectral density of the joint targets (policy rate) and of the torques (physics rate) for the
    SPECTRA_BINS commands, averaged over joints and over the envs of the bin that never fell.
    The gait sits left of the dashed line (HF_CUTOFF_HZ); jitter shows up right of it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    never_fell = rec["valid"].all(0)
    bin_id = bin_id.cpu()
    tau, D = physics_rate_torques(rec)
    panels = (
        ("Joint targets (policy, 50 Hz)", "rad²/Hz", psd(rec["q_target"], 1.0 / dt)),
        ("Joint torques (physics, 200 Hz)", "Nm²/Hz", psd(tau, D / dt)),
    )
    ink, muted, hairline = "#0b0b0b", "#52514e", "#e6e5e0"
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    missing = []
    for k, cmd in enumerate(SPECTRA_BINS):
        # [P1.6] match (vx, vy, wz) at dh = 0 when the grid has a 4th (body-height) column
        same = (grid[:, :3] - torch.tensor(cmd)).abs().sum(-1) < 1e-6
        if grid.shape[1] > 3:
            same &= grid[:, 3].abs() < 1e-6
        rows = torch.nonzero(same).flatten()
        if rows.numel() == 0:
            continue
        envs = (bin_id == rows[0]) & never_fell
        if not envs.any():
            missing.append(f"vx {cmd[0]:+.1f} vy {cmd[1]:+.1f} wz {cmd[2]:+.1f}")
            continue
        color, dash = SERIES_STYLE[k % len(SERIES_STYLE)]
        for ax, (_, _, (f, p)) in zip(axes, panels):
            ax.semilogy(f[1:], p[1:, envs].mean(dim=(1, 2)), color=color, linestyle=dash, linewidth=1.6,
                        label=f"vx {cmd[0]:+.1f}  vy {cmd[1]:+.1f}  wz {cmd[2]:+.1f}")
    for ax, (title, unit, (f, _)) in zip(axes, panels):
        ax.axvline(HF_CUTOFF_HZ, color=muted, linewidth=0.8, linestyle=(0, (4, 3)))
        ax.annotate(f"jitter >{HF_CUTOFF_HZ:.0f} Hz", (HF_CUTOFF_HZ, 0.0), xycoords=("data", "axes fraction"),
                    xytext=(4, 4), textcoords="offset points", fontsize=7, color=muted)
        ax.set_xlim(0, float(f[-1]))
        ax.set_title(title, fontsize=9, loc="left", color=ink)
        ax.set_xlabel("frequency [Hz]", fontsize=8, color=ink)
        ax.set_ylabel(f"PSD [{unit}]", fontsize=8, color=ink)
        ax.tick_params(labelsize=7, colors=muted)
        ax.grid(True, which="major", color=hairline, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    if len(missing) < len(SPECTRA_BINS) and axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(frameon=False, fontsize=7, labelcolor=ink, loc="upper right")
    if missing:
        fig.text(0.01, 0.01, "not shown, every env fell: " + "; ".join(missing), fontsize=7, color=muted)
    fig.tight_layout(rect=(0, 0.05 if missing else 0, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"spectra plot saved to {path}")


def plot_gait(rec, bin_id, grid, dt, path, seconds=3.0):
    """Contact sequence of one representative env per command bin (first env that never fell)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c, v = foot_contact(rec).numpy(), rec["valid"].numpy()
    bin_id = bin_id.cpu().numpy()
    Tp = min(c.shape[0], round(seconds / dt))
    fig, axes = plt.subplots(len(grid), 1, figsize=(8, 0.9 * len(grid) + 0.8), sharex=True, squeeze=False)
    for b, ax in enumerate(axes[:, 0]):
        envs = np.flatnonzero((bin_id == b) & v.all(0))
        if envs.size:
            n = envs[0]
            for f in range(4):
                s, e = runs(c[:Tp, n, f])
                ax.broken_barh([(i * dt, (j - i) * dt) for i, j in zip(s, e)], (f - 0.35, 0.7), color="#3b4a5a")
        else:
            ax.text(0.5, 0.5, "all envs fell", transform=ax.transAxes, ha="center", va="center", fontsize=8)
        ax.set_yticks(range(4), [f.split("_")[0] for f in FEET], fontsize=7)
        ax.set_ylim(3.6, -0.6)
        label = f"vx {grid[b, 0]:+.2f}\nvy {grid[b, 1]:+.2f}\nwz {grid[b, 2]:+.2f}"
        if grid.shape[1] > 3:  # [P1.6]
            label += f"\ndh {grid[b, 3]:+.2f}"
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=7)
        ax.tick_params(axis="x", labelsize=7)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="x", color="0.9", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[-1, 0].set_xlim(0, Tp * dt)
    axes[-1, 0].set_xlabel("time since start of measurement [s]", fontsize=8)
    fig.suptitle("Foot contact sequence (bar = foot on the ground)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"gait plot saved to {path}")


def m_slip(rec, dt):
    v = rec["valid"]                                   # [T, N]
    vj = v[..., None]
    c = foot_contact(rec)                              # [T, N, 4]
    n_valid = v.sum(0)

    # Velocity of the CONTACT POINT (bottom of the foot sphere), not of its centre:
    # v_cp = v_centre + w x r, with r = (0, 0, -FOOT_R). A rolling foot then gives 0.
    # Horizontal part: (vx - R*wy, vy + R*wx).
    fv, fw = rec["foot_vel"], rec["foot_ang"]
    v_cp = torch.stack((fv[..., 0] - FOOT_R * fw[..., 1], fv[..., 1] + FOOT_R * fw[..., 0]), dim=-1)

    # Tangential (horizontal) contact-point speed. On flat ground "tangential" = world xy.
    # (On a slope this must become the component in the terrain plane.)
    vt = torch.linalg.norm(v_cp, dim=-1)                            # [T, N, 4]

    # 1) Mean foot speed during stance. The first sample of each stance is dropped
    #    (contact data lag by ~1 step, so that sample can still be the landing motion).
    core = c[1:] & c[:-1] & vj[1:] & vj[:-1]           # [T-1, N, 4]
    m = core.float()
    n_st = m.sum(dim=(0, 2))
    slip_v = (vt[1:] * m).sum(dim=(0, 2)) / n_st.clamp(min=1)
    slip_v[n_st == 0] = float("nan")

    # 2) Slip distance per step: distance the foot slides during one complete stance
    #    (touchdown and lift-off both inside the valid window), averaged over all stances.
    cn, vtn, n_ok = c.numpy(), vt.numpy(), n_valid.numpy()
    slip_d = np.full(c.shape[1], np.nan)
    for n in range(c.shape[1]):
        L, dists = n_ok[n], []
        for f in range(4):
            starts, ends = runs(cn[:L, n, f])
            dists += [vtn[a + 1:b, n, f].sum() * dt for a, b in zip(starts, ends) if a > 0 and b < L]
        if dists:
            slip_d[n] = np.mean(dists)

    # 3) Slip ratio = 1 - distance gained / distance commanded, along the commanded direction.
    #    "Gained" integrates the base velocity (base frame, same frame as the command).
    cmd = rec["cmd"][..., :2]                                        # [T, N, 2]
    cmd_speed = torch.linalg.norm(cmd[0], dim=-1)                    # [N]
    direction = cmd / torch.linalg.norm(cmd, dim=-1, keepdim=True).clamp(min=1e-6)
    progress = (rec["lin_vel"][..., :2] * direction).sum(-1)         # [T, N] m/s
    gained = (progress * v).sum(0) * dt                              # [N] m
    commanded = cmd_speed * n_valid * dt                             # [N] m
    ratio = 1.0 - gained / commanded.clamp(min=1e-9)
    ratio[(cmd_speed < COT_MIN_SPEED) | (n_valid == 0)] = float("nan")

    return {"slip_v[cm/s]": 100.0 * slip_v,
            "slip_d[cm/step]": 100.0 * torch.from_numpy(slip_d).float(),
            "slip_ratio[-]": ratio}


# One printed table per entry. Add new metric functions to the right table.
TABLES = {
    "Tracking & stability": [m_tracking, m_measured, m_height],   # [P1.6] + m_height
    "Smoothness": [m_action_smoothness, m_torque_smoothness],   # [P1.4] split from "Smoothness & actuation"
    "Actuation limits": [m_torque_limits],
    "Energy": [m_cot],
    "Gait": [m_gait],                                             # [P2.1] + air_t, stride
    "Gait pattern & symmetry": [m_gait_pattern, m_symmetry],      # [P2.3] [P2.4]
    "Foot slip": [m_slip],
}

# [P2.3] Mirror-pair check: a symmetric policy performs equally on a command and on its mirror image (Mittal et al.
# 2024 report this "variation across equivalent goals"). Shown for these metrics, on every pair of grid bins that are
# mirror images (vy -> -vy, wz -> -wz).
MIRROR_PAIR_METRICS = ("lin_err[m/s]", "yaw_err[rad/s]", "air_t[s]", "stride[cm]", "power[W]")
# differences are relative to max(mean of the pair, this floor), so that two near-zero values do not show as 100 %
MIRROR_PAIR_FLOOR = {"lin_err[m/s]": 0.05, "yaw_err[rad/s]": 0.05, "air_t[s]": 0.02, "stride[cm]": 1.0, "power[W]": 1.0}


# =============================================================================
# Aggregation + printing
# =============================================================================
def compute_stats(rec, bin_id, n_bins, dt):
    """Run every metric, then mean/std over the envs of each command bin.
    Returns stats[name] = (mean[n_bins], std[n_bins]) and table_cols[title] = [names]."""
    bin_id = bin_id.cpu().numpy()
    stats, table_cols = {}, {}
    for title, fns in TABLES.items():
        table_cols[title] = []
        for fn in fns:
            for name, vals in fn(rec, dt).items():
                x = vals.numpy()
                mean, std = np.full(n_bins, np.nan), np.full(n_bins, np.nan)
                for b in range(n_bins):
                    xb = x[bin_id == b]
                    xb = xb[np.isfinite(xb)]
                    if xb.size:
                        mean[b], std[b] = xb.mean(), xb.std()
                stats[name] = (mean, std)
                table_cols[title].append(name)
    return stats, table_cols


def bin_header(rec, bin_id, grid, ood):
    """Columns shared by every table: the command, the OOD flag, the falls."""
    fell = rec["fall_step"] >= 0
    bin_id = bin_id.cpu()
    header = []
    for b in range(len(grid)):
        h = {"vx": grid[b, 0].item(), "vy": grid[b, 1].item(), "wz": grid[b, 2].item()}
        if grid.shape[1] > 3:  # [P1.6] body-height offset
            h["dh"] = grid[b, 3].item()
        h["ood"] = "*" if ood[b] else ""
        h["falls"] = f"{int(fell[bin_id == b].sum())}/{int((bin_id == b).sum())}"
        header.append(h)
    return header


def print_table(rows, title):
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c])) for r in rows)) for c in cols}
    fmt = lambda c, v: (f"{v:.2f}" if isinstance(v, float) else str(v)).rjust(widths[c])
    print(f"\n--- {title} ---")
    print("  ".join(c.rjust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(fmt(c, r[c]) for c in cols))


def print_mirror_pairs(stats, grid, cmd_cols):
    """[P2.3] For each pair of bins that are mirror images, the metrics of MIRROR_PAIR_METRICS side by side and their
    relative difference |A - B| / max(mean(A, B), floor). Large differences = left and right are treated differently."""
    flip = torch.ones(grid.shape[1])
    flip[1:3] = -1.0                                   # (vx, vy, wz[, dh]) -> (vx, -vy, -wz[, dh])
    rows = []
    for a in range(len(grid)):
        for b in range(a + 1, len(grid)):
            if (grid[b] - grid[a] * flip).abs().max() < 1e-6 and (grid[a] - grid[b]).abs().max() > 1e-6:
                row = {"A": " ".join(f"{x:+.2f}" for x in grid[a].tolist()),
                       "B": " ".join(f"{x:+.2f}" for x in grid[b].tolist())}
                for name in MIRROR_PAIR_METRICS:
                    va, vb = stats[name][0][a], stats[name][0][b]
                    rel = abs(va - vb) / max(0.5 * abs(va + vb), MIRROR_PAIR_FLOOR.get(name, 1e-9))
                    row[name] = f"{va:.3g} / {vb:.3g} ({100 * rel:.0f}%)" if np.isfinite(rel) else "n/a"
                rows.append(row)
    if rows:
        title = f"Mirror-pair check: A / B (|A-B| / max(mean, floor)), commands as {' '.join(cmd_cols)}"
        print_table(rows, title)


def report(rec, bin_id, grid, ood, dt, title, csv_path):
    stats, table_cols = compute_stats(rec, bin_id, len(grid), dt)
    header = bin_header(rec, bin_id, grid, ood)

    cmd_cols = [k for k in ("vx", "vy", "wz", "dh") if k in header[0]]  # [P1.6] dh for 4-command runs
    print(f"\n=== {title} ===")
    for i, (tname, names) in enumerate(table_cols.items()):
        rows = []
        for b, h in enumerate(header):
            row = dict(h) if i == 0 else {k: h[k] for k in (*cmd_cols, "ood")}
            for n in names:
                m, s = stats[n][0][b], stats[n][1][b]
                row[n] = f"{m:.3f} ± {s:.3f}" if np.isfinite(m) else "n/a"
            rows.append(row)
        print_table(rows, tname)
    print("(* = command outside the training command range; n/a = not defined or no valid data)")
    print_mirror_pairs(stats, grid, cmd_cols)  # [P2.3]

    # Same numbers as a CSV (mean and std in separate numeric columns) for plots / comparisons.
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        names = list(stats.keys())
        w.writerow(cmd_cols + ["ood", "falls"] + [f"{n}_{s}" for n in names for s in ("mean", "std")])
        for b, h in enumerate(header):
            w.writerow([h[c] for c in cmd_cols] + [h["ood"] == "*", h["falls"]]
                       + [f"{stats[n][k][b]:.6g}" for n in names for k in (0, 1)])
    print(f"table saved to {csv_path}")


# =============================================================================
# Joint limits (read once from the robot model, i.e. from the URDF)
# =============================================================================
def get_joint_limits(env):
    """Position limits [rad] and torque limits [Nm] of the 12 motors, in joint_names order."""
    first = lambda x: (x[0] if x.ndim == 2 else x).cpu()   # some versions return one row per env
    q_lo, q_hi = env.robot.get_dofs_limit(env.motors_dof_idx)
    f_lo, f_hi = env.robot.get_dofs_force_range(env.motors_dof_idx)
    lim = {
        "q_lo": first(q_lo),
        "q_hi": first(q_hi),
        "tau_lim": torch.minimum(first(f_lo).abs(), first(f_hi).abs()),
    }
    print("\nJoint limits used (check against the Go2 datasheet):")
    for i, name in enumerate(env.env_cfg["joint_names"]):
        print(f"  {name:18s} q [{lim['q_lo'][i]:+.2f}, {lim['q_hi'][i]:+.2f}] rad   "
              f"tau_max {lim['tau_lim'][i]:.1f} Nm")
    if not torch.isfinite(lim["tau_lim"]).all() or (lim["tau_lim"] > 1e3).any():
        print("  WARNING: some torque limits are infinite/huge -> the URDF effort limits were not loaded.")

    # [P1.2] per-env mass when the base mass is randomized (batched link info): a scalar or [N], CoT broadcasts it
    lim["mass"] = torch.as_tensor(env.robot.get_mass(), dtype=torch.float32).cpu()
    spread = f" (per env {lim['mass'].min():.2f} .. {lim['mass'].max():.2f})" if lim["mass"].numel() > 1 else ""
    print(f"  total robot mass: {lim['mass'].mean():.2f} kg{spread}")
    return lim


# =============================================================================
# Main
# =============================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    p.add_argument("--ckpt", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reps", type=int, default=20, help="envs per command bin")
    p.add_argument("--warmup_s", type=float, default=2.0, help="discarded transient")
    p.add_argument("--measure_s", type=float, default=10.0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--init_noise", type=float, default=0.1,
                   help="uniform noise on initial joint positions [rad] (0 = identical starts)")
    p.add_argument("--log_dir", type=str, default=None,
                   help="folder containing cfgs.pkl and model_*.pt (default: logs/<exp_name>)")
    # [P1.2] every policy is measured in the same nominal world (no randomization, no noise, no pushes, fixed
    # latency) unless --keep_dr; a fall is a tilt beyond --fall_deg whatever the run's training termination
    p.add_argument("--keep_dr", action="store_true",
                   help="keep the run's domain randomization, noise and pushes (robustness test)")
    p.add_argument("--latency_ms", type=float, default=20.0, help="action latency in the nominal world")
    p.add_argument("--fall_deg", type=float, default=10.0, help="roll/pitch beyond this counts as a fall")
    # [P1.6] e.g. --tag dr for a --keep_dr evaluation: metrics_ckpt999_seed0_dr.csv, so it does not overwrite the
    # nominal one of the same checkpoint and seed
    p.add_argument("--tag", type=str, default="", help="suffix of the output file names")
    # [S2R] joint losses of the evaluation world; default: the run's own (0 for runs trained before [S2R]).
    # go2.xml: --joint_friction 0.2 --joint_damping 0.1
    p.add_argument("--joint_friction", type=float, default=None, help="joint Coulomb friction [Nm]")
    p.add_argument("--joint_damping", type=float, default=None, help="joint viscous damping [Nm s/rad]")
    args = p.parse_args()

    log_dir = args.log_dir or os.path.join("logs", args.exp_name)
    for needed in ("cfgs.pkl", f"model_{args.ckpt}.pt"):
        if not os.path.isfile(os.path.join(log_dir, needed)):
            raise SystemExit(
                f"Missing {os.path.abspath(os.path.join(log_dir, needed))}\n"
                f"Point --log_dir to the folder go2_train.py created, or run from the folder that contains logs/."
            )

    seed_everything(args.seed)
    gs.init(seed=args.seed, backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")

    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}                       # no reward computation needed
    if not args.keep_dr:                                   # [P1.2]
        nominal_world(env_cfg, obs_cfg, latency_s=args.latency_ms / 1000.0)
    env_cfg["recovery"] = {}  # [P1.6] fix: no fallen starts, also with --keep_dr (P1.5 --recovery runs had 15 %)
    env_cfg["termination_if_roll_greater_than"] = args.fall_deg    # [P1.2] same fall criterion for every run
    env_cfg["termination_if_pitch_greater_than"] = args.fall_deg
    for key in ("joint_friction", "joint_damping"):  # [S2R]
        if getattr(args, key) is not None:
            env_cfg[key] = getattr(args, key)
            env_cfg.get("domain_rand", {}).pop(f"{key}_range", None)  # a fixed value, also with --keep_dr

    total_s = args.warmup_s + args.measure_s
    env_cfg["episode_length_s"] = total_s + 10.0           # no timeouts during the run
    env_cfg["resampling_time_s"] = total_s + 10.0          # irrelevant (pinned), but harmless
    env_cfg["links_to_keep"] = FEET                        # keep foot links (needs go2_env.py edit)

    # Env layout: bins repeated `reps` times -> env i belongs to bin bin_id[i]
    # [P1.6] runs trained with the body-height command: COMMAND_GRID at dh = 0, then HEIGHT_GRID
    if command_cfg["num_commands"] >= 4:
        grid_rows = [c + (0.0,) for c in COMMAND_GRID] + list(HEIGHT_GRID)
    else:
        grid_rows = COMMAND_GRID
    grid = torch.tensor(grid_rows, dtype=gs.tc_float, device=gs.device)
    n_bins = grid.shape[0]
    bin_id = torch.arange(n_bins, device=gs.device).repeat_interleave(args.reps)
    num_envs = n_bins * args.reps

    env = EvalGo2Env(
        grid[bin_id],
        num_envs=num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg,
        reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=False,
        init_noise=args.init_noise, noise_seed=args.seed,
    )

    lo, hi = env.commands_limits
    ood = ((grid < lo - 1e-6) | (grid > hi + 1e-6)).any(dim=1).cpu()

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    n_warmup = round(args.warmup_s / env.dt)
    n_measure = round(args.measure_s / env.dt)
    rec = run_rollout(env, policy, n_warmup, n_measure)
    rec.update(get_joint_limits(env))
    # [P1.6] commanded base height [T, N]: base_height_target + dh (dh = 0 without the body-height command)
    z_nominal = float(reward_cfg["base_height_target"])
    rec["z_target"] = z_nominal + (rec["cmd"][..., 3] if rec["cmd"].shape[-1] > 3 else 0.0 * rec["cmd"][..., 0])

    world = "training DR" if args.keep_dr else f"nominal world, latency {args.latency_ms:.0f} ms"  # [P1.2]
    title = (f"{args.exp_name} | ckpt {args.ckpt} | seed {args.seed} | init_noise {args.init_noise} rad | "
             f"{num_envs} envs | {args.measure_s:.0f} s | {world} | fall > {args.fall_deg:.0f} deg | "
             f"joints {env_cfg.get('joint_friction', 0.0):g} Nm / {env_cfg.get('joint_damping', 0.0):g} Nm s/rad")  # [S2R]
    name = f"ckpt{args.ckpt}_seed{args.seed}" + (f"_{args.tag}" if args.tag else "")  # [P1.6] --tag
    csv_path = os.path.join(log_dir, f"metrics_{name}.csv")
    # Genesis makes CUDA the default device for new tensors; all metrics work on the
    # CPU copies of the recordings, so new tensors must be created on the CPU too.
    with torch.device("cpu"):
        report(rec, bin_id, grid.cpu(), ood, env.dt, title, csv_path)
        plot_gait(rec, bin_id, grid.cpu(), env.dt, os.path.join(log_dir, f"gait_{name}.png"))
        plot_spectra(rec, bin_id, grid.cpu(), env.dt, os.path.join(log_dir, f"spectra_{name}.png"))  # [P1.4]

    # Determinism check: two runs with the same seed must print the same number.
    print(f"checksum: {rec['lin_vel'].double().sum().item():.10e}")

    out = os.path.join(log_dir, f"metrics_{name}.npz")
    np.savez(out, bin_id=bin_id.cpu().numpy(), grid=grid.cpu().numpy(),
             **{k: v.numpy() for k, v in rec.items()})
    print(f"raw data saved to {out}")


if __name__ == "__main__":
    main()