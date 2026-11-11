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
[P3.1] Terrain (TEST_AND_TINKER_P3.md, section 4.4). Default: flat ground, as before, also for runs trained on terrain.
    --terrain grid      one bin per (terrain type, level) of a terrain grid: the run's own (logs/<exp>/terrain.npz),
                        else go2_terrain.TERRAIN_DEFAULTS (so a flat-trained run is measured on the same tiles), changed
                        with --terrain_set (e.g. seed=1: unseen tiles; max_slope_deg=30: beyond training). Every robot
                        starts at its tile centre, facing +x, with --grid_cmd, for the time a perfect tracker needs to
                        walk half a tile (out of its tile). Output tagged _grid, plus terrain_*.png (success heat map).
    --slope DEG         the usual command grid on a plane tilted by DEG (rising along +x: forward = uphill; negative:
                        forward = downhill; vy = across the slope). Output tagged _slope+DEG.
    Both add the tilt the ground itself causes to --fall_deg, measure heights above the ground (z, z_err, swing_h), and
    add a "Progress" table: dist[m] (distance from the start point) and through[%] (share of robots that covered at
    least --through of the commanded distance without falling).
The curriculum state saved in the checkpoint (P3.1 runs) is printed, and its command ranges set the ood column.
[P3.2] Foot slip from the simulator's ground truth (TEST_AND_TINKER_P3.md, section 5): slip is the speed of each foot
sphere's contact point along the ground's tangent plane (go2_terrain.contact_slip_speed, normal of the face under the
foot), in stance (|F| > 3 N, first sample of each stance dropped). On flat ground that is the P1 horizontal speed, and
the P1 columns keep their values. New: slip_share[%] (share of stance samples sliding faster than 2 cm/s), pen_xy and
pen_t [cm/s] (what go2_train.py --foot-slip sums, modes xy and tangent), and the "Foot slip per foot" table (FL, FR,
RL, RR: slip_v / slip_d / slip_share of each foot; separate columns in the CSV).
[P3.3] Runs with a privileged group: sym_err mirrors every group the actor reads (go2_symmetry.mirror_observations).
"""

import argparse
import csv
import os
import pickle
import random

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

import genesis as gs
from go2_env import Go2Env, nominal_world, GO2_FOOT_RADIUS, TROT_PAIRS
from go2_symmetry import mirror_actions, mirror_observations  # [P2.3] [P3.3] every observation group
from go2_terrain import GENERATION_KEYS, TerrainGrid, make_ground, terrain_config  # [P3.1]
from go2_terrain import contact_slip_speed  # [P3.2]


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
    and every reset starts from the default pose + seeded uniform joint noise.
    [P3.1] On a terrain grid, env i starts on the tile fixed_tiles[i] = (column, level); tilt_limit [N] (deg) replaces
    the tilt termination of each env (fall threshold + the tilt its ground causes)."""

    def __init__(self, fixed_commands, *args, init_noise=0.0, noise_seed=0, fixed_tiles=None, tilt_limit=None,
                 **kwargs):
        # Must exist BEFORE super().__init__(), because __init__ calls reset(),
        # which calls _resample_commands() and _reset_idx().
        self.fixed_commands = fixed_commands
        self.fixed_tiles = fixed_tiles                                # [P3.1]
        self.init_noise = init_noise                                  # [rad], 0 = off
        self.noise_gen = torch.Generator(device=gs.device).manual_seed(noise_seed)
        super().__init__(*args, **kwargs)
        if tilt_limit is not None:                                    # [P3.1]
            self.max_pitch_deg.copy_(tilt_limit)
            self.max_roll_deg.copy_(tilt_limit)

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

    def _terrain_assignment(self):
        """[P3.1] The tile of each env's bin (Go2Env: by type shares and random levels)."""
        if self.fixed_tiles is None:
            return super()._terrain_assignment()
        cols, levels = self.fixed_tiles
        return cols.to(gs.device), levels.to(gs.device)

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
        if self.on_terrain:
            # [P3.1] Genesis 1.4.2: set_dofs_position() after the reset's set_qpos(skip_forward=True) rebuilds the free
            # base from the stale state, i.e. puts the base back where it was before the reset (harmless on flat ground,
            # where every reset starts at the origin; on terrain it undid the spawn on the tile). Joints and spawn
            # pose are written together instead, WITH forward kinematics: like set_dofs_position() on flat ground, so
            # the PD of the first physics substep reads the new joint angles (skip_forward=True leaves the old ones
            # there: a kick of up to the torque limit that made robots fall 4x more often at their first landing on
            # flat tiles than on the plane, from identical states).
            qpos = self.init_qpos.repeat(self.num_envs, 1)
            qpos[:, :3] = self.spawn_pos
            qpos[:, 3:7] = self.spawn_quat
            qpos[idx[:, None], self.motors_dof_idx + 1] = q          # qpos = [pos 3, quat 4, joints]: dof d -> d + 1
            mask = torch.zeros((self.num_envs,), dtype=gs.tc_bool, device=gs.device)
            mask[idx] = True
            self.robot.set_qpos(qpos, envs_idx=mask, zero_velocity=True, skip_forward=False)
        else:
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
    if env.on_terrain:  # [P3.1] (not recorded on flat ground: the flat recordings stay as they were)
        log["foot_ground"] = []  # [T, N, 4] ground height under each foot [m]
        log["base_height"] = []  # [T, N]    base height above the mean ground under the feet [m]
        log["foot_normal"] = []  # [T, N, 4, 3] [P3.2] unit normal of the ground face under each foot
    # [P2.3] the mirrored query must not touch a recurrent policy's memory: MLP policies only
    check_symmetry = not getattr(policy, "is_recurrent", False)
    scale = env.env_cfg["action_scale"]

    obs = env.reset()
    spawn_pos = env.spawn_pos.clone()  # [P3.1] start point of every env (dist[m], through[%])
    with torch.no_grad():
        for t in range(n_warmup + n_measure):
            actions = policy(obs)
            if t >= n_warmup:  # [P2.3] how far the policy is from its own mirror image, in joint-target radians
                if check_symmetry:
                    mirrored = policy(mirror_observations(env, obs))  # [P3.3] every group (a teacher reads two)
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
                foot_pos = env.robot.get_links_pos()[:, feet].clone()
                log["foot_pos"].append(foot_pos)
                log["foot_vel"].append(env.robot.get_links_vel()[:, feet].clone())
                log["foot_ang"].append(env.robot.get_links_ang()[:, feet].clone())
                log["foot_force"].append(F[:, feet].clone())
                log["other_force"].append(torch.linalg.norm(F[:, not_feet], dim=-1).sum(-1))
                log["power"].append(env.mech_power.clone())  # [P2.5]
                if env.on_terrain:  # [P3.1]
                    log["foot_ground"].append(env.feet_ground.clone())
                    log["base_height"].append(env.base_height.clone())
                    log["foot_normal"].append(env.ground.normal_at(foot_pos[..., :2]))  # [P3.2] at the logged feet

    rec = {k: torch.stack(v).cpu() for k, v in log.items()}
    rec["fall_step"] = fall_step.cpu()
    if env.on_terrain:  # [P3.1]
        rec["spawn_pos"] = spawn_pos.cpu()
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
    # [P3.1] on terrain, the base height above the mean ground under the feet (what the base_height reward targets)
    z = rec["base_height"] if "base_height" in rec else rec["base_pos"][..., 2]    # [T, N]
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
    # [P3.1] foot height above the ground under it (flat ground: world z)
    z = pos[..., 2] - rec["foot_ground"].numpy() if "foot_ground" in rec else pos[..., 2]   # [T, N, 4]
    out = []
    for n in range(c.shape[1]):
        L, items = n_ok[n], []
        for f in range(4):
            starts, ends = runs(~c[:L, n, f])
            for a, b in zip(starts, ends):
                if a > 0 and b < L:
                    stride = np.linalg.norm(pos[b, n, f, :2] - pos[a - 1, n, f, :2])
                    items.append((f, (b - a) * dt, z[a:b, n, f].max() - FOOT_R, stride))
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


def plot_gait(rec, bin_id, grid, dt, path, seconds=3.0, labels=None, bins=None):
    """Contact sequence of one representative env per command bin (first env that never fell).
    [P3.1] labels: per-bin names (--terrain grid); bins: the bins to draw (default all)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c, v = foot_contact(rec).numpy(), rec["valid"].numpy()
    bin_id = bin_id.cpu().numpy()
    Tp = min(c.shape[0], round(seconds / dt))
    bins = list(range(len(grid))) if bins is None else list(bins)
    fig, axes = plt.subplots(len(bins), 1, figsize=(8, 0.9 * len(bins) + 0.8), sharex=True, squeeze=False)
    for b, ax in zip(bins, axes[:, 0]):
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
        if labels:  # [P3.1]
            label = f"{labels[b]['terrain']}\nlevel {labels[b]['level']}"
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


def plot_terrain(stats, labels, path, title):
    """[P3.1] --terrain grid: success heat map, one cell per (type, level): through[%] as the colour (blue ramp: light
    = few robots got through), the value written in each cell, and the falls under it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    types = list(dict.fromkeys(lab["terrain"] for lab in labels))
    levels = sorted({lab["level"] for lab in labels})
    through = np.full((len(types), len(levels)), np.nan)
    falls = np.full_like(through, np.nan)
    for b, lab in enumerate(labels):
        i, j = types.index(lab["terrain"]), levels.index(lab["level"])
        through[i, j] = stats["through[%]"][0][b]
        falls[i, j] = lab["fall_share"]
    ink, muted = "#0b0b0b", "#52514e"
    ramp = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    fig, ax = plt.subplots(figsize=(1.0 + 0.75 * len(levels), 1.2 + 0.55 * len(types)))
    im = ax.imshow(through, cmap=ramp, vmin=0, vmax=100, aspect="auto")
    for i in range(len(types)):
        for j in range(len(levels)):
            if np.isfinite(through[i, j]):
                dark = through[i, j] > 55
                ax.text(j, i - 0.12, f"{through[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if dark else ink)
                ax.text(j, i + 0.22, f"falls {100 * falls[i, j]:.0f}%", ha="center", va="center", fontsize=5.5,
                        color="#ffffff" if dark else muted)
    ax.set_xticks(range(len(levels)), [str(k) for k in levels])
    ax.set_yticks(range(len(types)), types)
    ax.set_xlabel("terrain level", fontsize=8, color=ink)
    ax.tick_params(labelsize=7, colors=muted, length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("through [%]", fontsize=8, color=ink)
    cb.ax.tick_params(labelsize=7, colors=muted)
    fig.suptitle(title, fontsize=7, color=ink, x=0.01, ha="left", wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"terrain plot saved to {path}")


SLIP_SHARE_V = 0.02  # [m/s] [P3.2] slip_share: a stance sample counts as slipping above this contact-point speed


def contact_point_speed(rec):
    """[P3.2] Speed of each foot's contact point along the ground [T, N, 4] [m/s]: go2_terrain.contact_slip_speed on
    the face under the foot (recorded foot_normal, terrain modes). Flat recordings (no foot_normal): the P1 formula,
    character for character, so the flat tables keep their exact values (both are the same quantity on flat ground)."""
    fv, fw = rec["foot_vel"], rec["foot_ang"]
    if "foot_normal" in rec:
        return contact_slip_speed(fv, fw, rec["foot_normal"], FOOT_R)
    # Velocity of the CONTACT POINT (bottom of the foot sphere), not of its centre:
    # v_cp = v_centre + w x r, with r = (0, 0, -FOOT_R). A rolling foot then gives 0.
    # Horizontal part: (vx - R*wy, vy + R*wx).
    v_cp = torch.stack((fv[..., 0] - FOOT_R * fw[..., 1], fv[..., 1] + FOOT_R * fw[..., 0]), dim=-1)
    return torch.linalg.norm(v_cp, dim=-1)                          # [T, N, 4]


def stance_distances(c, vt, n_valid, dt):
    """[P3.2] Per env and foot, the list of slip distances [m] of the complete stances (touchdown and lift-off inside
    the valid window), first sample of each stance dropped. c, vt: [T, N, 4] numpy."""
    out = []
    for n in range(c.shape[1]):
        L = n_valid[n]
        out.append([[vt[a + 1:b, n, f].sum() * dt for a, b in zip(*runs(c[:L, n, f])) if a > 0 and b < L]
                    for f in range(c.shape[2])])
    return out


def m_slip(rec, dt):
    """Foot slip, all feet together. [P3.2] slip_v / slip_d are measured along the ground (the same as before on flat
    ground). slip_share: share of stance samples sliding faster than SLIP_SHARE_V. pen_xy / pen_t: the per-step sum
    over feet that go2_train.py --foot-slip penalizes (contact at the reward's threshold, 1 N), mean over the window:
    mode xy (Isaac Lab Spot: planar speed of the foot-sphere CENTRE) and mode tangent (contact-point slip). The penalty
    costs weight x pen per second of reward; pen_xy - pen_t is mostly the foot's rolling (R x omega of the calf)."""
    v = rec["valid"]                                   # [T, N]
    vj = v[..., None]
    c = foot_contact(rec)                              # [T, N, 4]
    n_valid = v.sum(0)

    # Tangential contact-point speed (on flat ground: horizontal, the P1 formula)
    vt = contact_point_speed(rec)                                   # [T, N, 4]

    # 1) Mean foot speed during stance. The first sample of each stance is dropped
    #    (contact data lag by ~1 step, so that sample can still be the landing motion).
    core = c[1:] & c[:-1] & vj[1:] & vj[:-1]           # [T-1, N, 4]
    m = core.float()
    n_st = m.sum(dim=(0, 2))
    slip_v = (vt[1:] * m).sum(dim=(0, 2)) / n_st.clamp(min=1)
    slip_v[n_st == 0] = float("nan")
    share = 100.0 * ((vt[1:] > SLIP_SHARE_V).float() * m).sum(dim=(0, 2)) / n_st.clamp(min=1)   # [P3.2]
    share[n_st == 0] = float("nan")

    # 2) Slip distance per step: distance the foot slides during one complete stance
    #    (touchdown and lift-off both inside the valid window), averaged over all stances.
    slip_d = np.full(c.shape[1], np.nan)
    for n, per_foot in enumerate(stance_distances(c.numpy(), vt.numpy(), n_valid.numpy(), dt)):
        dists = [d for foot in per_foot for d in foot]
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

    # 4) [P3.2] the foot_slip penalty's own sums (go2_env._reward_foot_slip), per policy step, mean over valid steps
    c_pen = torch.linalg.norm(rec["foot_force"], dim=-1) > rec["contact_threshold"]               # [T, N, 4]
    vf = v.float()
    pen_xy = ((torch.linalg.norm(rec["foot_vel"][..., :2], dim=-1) * c_pen).sum(-1) * vf).sum(0)
    pen_t = ((vt * c_pen).sum(-1) * vf).sum(0)
    pen_xy, pen_t = (x / n_valid.clamp(min=1) for x in (pen_xy, pen_t))
    for x in (pen_xy, pen_t):
        x[n_valid == 0] = float("nan")

    return {"slip_v[cm/s]": 100.0 * slip_v,
            "slip_d[cm/step]": 100.0 * torch.from_numpy(slip_d).float(),
            "slip_ratio[-]": ratio,
            "slip_share[%]": share,
            "pen_xy[cm/s]": 100.0 * pen_xy,
            "pen_t[cm/s]": 100.0 * pen_t}


def m_slip_feet(rec, dt):
    """[P3.2] Foot slip per foot (FL, FR, RL, RR), with the definitions of m_slip: slip_v (mean contact-point speed in
    stance), slip_d (slip distance per complete stance), slip_share (stance share above SLIP_SHARE_V). Which leg slips,
    and whether left and right differ (P2.3)."""
    v = rec["valid"]
    vj = v[..., None]
    c = foot_contact(rec)
    vt = contact_point_speed(rec)
    core = (c[1:] & c[:-1] & vj[1:] & vj[:-1]).float()             # [T-1, N, 4]
    n_st = core.sum(0)                                              # [N, 4]
    speed = (vt[1:] * core).sum(0) / n_st.clamp(min=1)
    share = 100.0 * ((vt[1:] > SLIP_SHARE_V).float() * core).sum(0) / n_st.clamp(min=1)
    speed[n_st == 0] = float("nan")
    share[n_st == 0] = float("nan")
    dist = np.full(tuple(n_st.shape), np.nan)
    for n, per_foot in enumerate(stance_distances(c.numpy(), vt.numpy(), v.sum(0).numpy(), dt)):
        for f, dists in enumerate(per_foot):
            if dists:
                dist[n, f] = np.mean(dists)
    dist = torch.from_numpy(dist).float()
    out = {}
    for quantity, values, unit in (("slip_v", 100.0 * speed, "cm/s"), ("slip_d", 100.0 * dist, "cm/step"),
                                   ("slip_share", share, "%")):
        for f, foot in enumerate(FEET):
            out[f"{quantity}_{foot.split('_')[0]}[{unit}]"] = values[:, f]
    return out


THROUGH_FRAC = 0.75  # [P3.1] through[%]: covered at least this share of the commanded distance, without falling


def m_progress(rec, dt):
    """[P3.1] Did the robot get through its ground? (terrain modes only, "Progress" table)
    - dist[m]: straight-line distance from the start point (spawn) at the end of the valid window, i.e. at the fall
      for a robot that fell. On the grid this is the terrain curriculum's own measure (half a tile = next level).
    - through[%]: 100 for a robot that never fell and whose dist reached THROUGH_FRAC (--through) of the distance its
      command asked for over warm-up + measurement, else 0; the bin mean is the success rate. n/a below 0.1 m/s, and
      for turning commands (|wz| > 0.05 rad/s: the path curves, so even a perfect tracker ends closer than planned)."""
    v = rec["valid"]
    n_valid = v.sum(0)                                                   # [N]
    last = (n_valid - 1).clamp(min=0)
    end = rec["base_pos"][last, torch.arange(v.shape[1]), :2]            # [N, 2] last valid position
    dist = torch.linalg.norm(end - rec["spawn_pos"][:, :2], dim=-1)
    cmd_speed = torch.linalg.norm(rec["cmd"][0, :, :2], dim=-1)
    planned = cmd_speed * rec["horizon_s"]
    through = 100.0 * ((rec["fall_step"] < 0) & (dist >= rec["through_frac"] * planned)).float()
    through[(cmd_speed < COT_MIN_SPEED) | (rec["cmd"][0, :, 2].abs() > 0.05)] = float("nan")
    dist[n_valid == 0] = float("nan")
    return {"dist[m]": dist, "through[%]": through}


# One printed table per entry. Add new metric functions to the right table.
TABLES = {
    "Tracking & stability": [m_tracking, m_measured, m_height],   # [P1.6] + m_height
    "Smoothness": [m_action_smoothness, m_torque_smoothness],   # [P1.4] split from "Smoothness & actuation"
    "Actuation limits": [m_torque_limits],
    "Energy": [m_cot],
    "Gait": [m_gait],                                             # [P2.1] + air_t, stride
    "Gait pattern & symmetry": [m_gait_pattern, m_symmetry],      # [P2.3] [P2.4]
    "Foot slip": [m_slip],
    "Foot slip per foot": [m_slip_feet],                          # [P3.2]
}
# [P3.2] tables printed one column per foot, "slip_v / slip_d / slip_share" (means; mean and std of each are in the CSV)
FOOT_TABLES = {"Foot slip per foot": ("slip_v", "slip_d", "slip_share")}
TERRAIN_TABLES = {"Progress": [m_progress]}  # [P3.1] appended in the terrain modes (--terrain grid, --slope)

# [P2.3] Mirror-pair check: a symmetric policy performs equally on a command and on its mirror image (Mittal et al.
# 2024 report this "variation across equivalent goals"). Shown for these metrics, on every pair of grid bins that are
# mirror images (vy -> -vy, wz -> -wz).
MIRROR_PAIR_METRICS = ("lin_err[m/s]", "yaw_err[rad/s]", "air_t[s]", "stride[cm]", "power[W]")
# differences are relative to max(mean of the pair, this floor), so that two near-zero values do not show as 100 %
MIRROR_PAIR_FLOOR = {"lin_err[m/s]": 0.05, "yaw_err[rad/s]": 0.05, "air_t[s]": 0.02, "stride[cm]": 1.0, "power[W]": 1.0}


# =============================================================================
# Aggregation + printing
# =============================================================================
def compute_stats(rec, bin_id, n_bins, dt, tables=None):
    """Run every metric, then mean/std over the envs of each command bin.
    Returns stats[name] = (mean[n_bins], std[n_bins]) and table_cols[title] = [names].
    [P3.1] tables: TABLES by default; the terrain modes add TERRAIN_TABLES."""
    bin_id = bin_id.cpu().numpy()
    stats, table_cols = {}, {}
    for title, fns in (tables or TABLES).items():
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


def bin_header(rec, bin_id, grid, ood, labels=None):
    """Columns shared by every table: the command, the OOD flag, the falls.
    [P3.1] labels: per bin, extra key columns put first (--terrain grid: terrain type and level)."""
    fell = rec["fall_step"] >= 0
    bin_id = bin_id.cpu()
    header = []
    for b in range(len(grid)):
        h = dict(labels[b]) if labels else {}
        h.update({"vx": grid[b, 0].item(), "vy": grid[b, 1].item(), "wz": grid[b, 2].item()})
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


def report(rec, bin_id, grid, ood, dt, title, csv_path, labels=None, tables=None):
    stats, table_cols = compute_stats(rec, bin_id, len(grid), dt, tables)
    header = bin_header(rec, bin_id, grid, ood, labels)

    # [P1.6] dh for 4-command runs; [P3.1] terrain and level first in --terrain grid
    cmd_cols = [k for k in ("terrain", "level", "vx", "vy", "wz", "dh") if k in header[0]]
    print(f"\n=== {title} ===")
    for i, (tname, names) in enumerate(table_cols.items()):
        rows = []
        for b, h in enumerate(header):
            row = dict(h) if i == 0 else {k: h[k] for k in (*cmd_cols, "ood")}
            if tname in FOOT_TABLES:  # [P3.2] one column per foot: the bin means of the quantities, "v / d / share"
                quantities = FOOT_TABLES[tname]
                for foot in (f.split("_")[0] for f in FEET):
                    cells = []
                    for q in quantities:
                        n = next(x for x in names if x.startswith(f"{q}_{foot}["))
                        m = stats[n][0][b]
                        cells.append(f"{m:.2f}" if np.isfinite(m) else "n/a")
                    row[foot] = " / ".join(cells)
            else:
                for n in names:
                    m, s = stats[n][0][b], stats[n][1][b]
                    row[n] = f"{m:.3f} ± {s:.3f}" if np.isfinite(m) else "n/a"
            rows.append(row)
        if tname in FOOT_TABLES:
            units = {"slip_v": "cm/s", "slip_d": "cm/step", "slip_share": "%"}
            tname = f"{tname}: " + " / ".join(f"{q} [{units.get(q, '')}]" for q in FOOT_TABLES[tname]) + \
                " (bin means; std in the CSV)"
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
def terrain_overrides(items):
    """[P3.1] --terrain_set KEY=VALUE ... -> dict (values as Python literals, else text)."""
    import ast
    out = {}
    for item in items:
        key, sep, text = item.partition("=")
        if not sep:
            raise SystemExit(f"--terrain_set {item}: expected KEY=VALUE")
        try:
            out[key] = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            out[key] = text
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    p.add_argument("--ckpt", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reps", type=int, default=20, help="envs per command bin")
    # [P3.1] defaults None: 2 s and 10 s on flat ground and slopes; 1 s and "half a tile" on the terrain grid
    p.add_argument("--warmup_s", type=float, default=None, help="discarded transient (default 2, grid: 1)")
    p.add_argument("--measure_s", type=float, default=None,
                   help="measured window (default 10; grid: the time --grid_cmd needs to walk half a tile, minus the "
                        "warm-up)")
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
    p.add_argument("--fall_deg", type=float, default=10.0,
                   help="roll/pitch beyond this counts as a fall ([P3.1] on terrain: beyond this + the tilt the "
                        "ground causes)")
    # [P1.6] e.g. --tag dr for a --keep_dr evaluation: metrics_ckpt999_seed0_dr.csv, so it does not overwrite the
    # nominal one of the same checkpoint and seed
    p.add_argument("--tag", type=str, default="", help="suffix of the output file names")
    # [S2R] joint losses of the evaluation world; default: the run's own (0 for runs trained before [S2R]).
    # go2.xml: --joint_friction 0.2 --joint_damping 0.1
    p.add_argument("--joint_friction", type=float, default=None, help="joint Coulomb friction [Nm]")
    p.add_argument("--joint_damping", type=float, default=None, help="joint viscous damping [Nm s/rad]")
    # [P3.1] terrain modes (module docstring)
    p.add_argument("--terrain", choices=("flat", "grid"), default="flat",
                   help="grid: one bin per (terrain type, level) of a terrain grid")
    p.add_argument("--slope", type=float, default=None, metavar="DEG",
                   help="the command grid on a plane tilted by DEG (forward = uphill; negative: downhill)")
    p.add_argument("--grid_cmd", type=float, nargs=3, default=(0.5, 0.0, 0.0), metavar=("VX", "VY", "WZ"),
                   help="command of every bin with --terrain grid")
    p.add_argument("--types", nargs="+", default=None, help="--terrain grid: these types only")
    p.add_argument("--levels", type=int, nargs="+", default=None, help="--terrain grid: these levels only")
    p.add_argument("--terrain_set", nargs="+", default=[], metavar="KEY=VALUE",
                   help="--terrain grid: change go2_terrain keys, e.g. seed=1 (unseen tiles) max_slope_deg=30")
    p.add_argument("--through", type=float, default=THROUGH_FRAC,
                   help="through[%%]: share of the commanded distance to cover without falling")
    args = p.parse_args()
    if args.slope is not None and args.terrain == "grid":
        p.error("--slope and --terrain grid exclude each other")
    grid_mode = args.terrain == "grid"
    on_terrain = grid_mode or args.slope is not None

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
    run_terrain = env_cfg.get("terrain")                   # [P3.1] what the run trained on (None: flat ground)
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

    # [P3.1] ground of the evaluation: flat (also with --keep_dr: randomization is not terrain), a tilted plane, or a
    # terrain grid (the run's exact heights when it has terrain.npz and no generation key is changed)
    env_cfg["terrain"] = None
    ground, labels, tiles = None, None, None
    if args.slope is not None:
        env_cfg["terrain"] = {"mode": "slope", "slope_deg": args.slope, "random_yaw": False, "spawn_xy_noise": 0.0}
        env_cfg["termination_if_roll_greater_than"] = args.fall_deg + abs(args.slope)
        env_cfg["termination_if_pitch_greater_than"] = args.fall_deg + abs(args.slope)
    elif grid_mode:
        overrides = terrain_overrides(args.terrain_set)
        base = dict(run_terrain) if run_terrain and run_terrain.get("mode", "grid") == "grid" else {"mode": "grid"}
        base.pop("file", None)
        base.update(overrides)
        base.update(mode="grid", curriculum=False, spawn_xy_noise=0.0, random_yaw=False, spawn_tile=None)
        saved = os.path.join(log_dir, "terrain.npz")
        if run_terrain and os.path.isfile(saved) and not set(overrides) & set(GENERATION_KEYS):
            base["file"] = saved
        env_cfg["terrain"] = base
        ground = make_ground(base)
        types = args.types or ground.types
        unknown = set(types) - set(ground.types)
        if unknown:
            raise SystemExit(f"--types {sorted(unknown)}: not in this grid ({ground.types})")
        levels = args.levels if args.levels is not None else list(range(ground.num_levels))
        if min(levels) < 0 or max(levels) >= ground.num_levels:
            raise SystemExit(f"--levels: this grid has levels 0 .. {ground.num_levels - 1}")
        labels = [{"terrain": t, "level": int(lvl)} for t in types for lvl in levels]
        tiles = [(ground.column_of_type(t), int(lvl)) for t in types for lvl in levels]
        speed = float(np.hypot(args.grid_cmd[0], args.grid_cmd[1]))
        if speed < COT_MIN_SPEED:
            raise SystemExit("--grid_cmd: give a planar speed of at least 0.1 m/s (the robots must cross their tile)")
    warmup_s = args.warmup_s if args.warmup_s is not None else (1.0 if grid_mode else 2.0)
    if args.measure_s is not None:
        measure_s = args.measure_s
    elif grid_mode:  # the time a perfect tracker needs to walk out of its tile (the curriculum's "next level")
        measure_s = max(1.0, 0.5 * ground.size / speed - warmup_s)
    else:
        measure_s = 10.0

    total_s = warmup_s + measure_s
    env_cfg["episode_length_s"] = total_s + 10.0           # no timeouts during the run
    env_cfg["resampling_time_s"] = total_s + 10.0          # irrelevant (pinned), but harmless
    env_cfg["links_to_keep"] = FEET                        # keep foot links (needs go2_env.py edit)

    # Env layout: bins repeated `reps` times -> env i belongs to bin bin_id[i]
    # [P1.6] runs trained with the body-height command: COMMAND_GRID at dh = 0, then HEIGHT_GRID
    # [P3.1] --terrain grid: one bin per tile, all with --grid_cmd
    if grid_mode:
        cmd = tuple(args.grid_cmd) + ((0.0,) if command_cfg["num_commands"] >= 4 else ())
        grid_rows = [cmd] * len(labels)
    elif command_cfg["num_commands"] >= 4:
        grid_rows = [c + (0.0,) for c in COMMAND_GRID] + list(HEIGHT_GRID)
    else:
        grid_rows = COMMAND_GRID
    grid = torch.tensor(grid_rows, dtype=gs.tc_float, device=gs.device)
    n_bins = grid.shape[0]
    bin_id = torch.arange(n_bins, device=gs.device).repeat_interleave(args.reps)
    num_envs = n_bins * args.reps
    fixed_tiles, tilt_limit = None, None
    if grid_mode:  # [P3.1] tile of each env; fall threshold + the tilt its tile causes
        tile_t = torch.tensor(tiles, dtype=torch.long, device=gs.device)[bin_id]
        fixed_tiles = (tile_t[:, 0], tile_t[:, 1])
        tilt = torch.tensor([ground.tile_tilt[lvl, col] for col, lvl in tiles], dtype=gs.tc_float, device=gs.device)
        tilt_limit = args.fall_deg + tilt[bin_id]

    env = EvalGo2Env(
        grid[bin_id],
        num_envs=num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg,
        reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=False,
        init_noise=args.init_noise, noise_seed=args.seed, fixed_tiles=fixed_tiles, tilt_limit=tilt_limit,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    infos = runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    # [P3.1] the command ranges this checkpoint trained on (its command curriculum may not have reached the configured
    # ones): the ood column flags commands outside them
    lo, hi = (t.clone() for t in env.commands_limits)
    state = (infos or {}).get("curriculum") if isinstance(infos, dict) else None
    if state:
        for k, key in enumerate(("lin_vel_x", "lin_vel_y", "ang_vel")):
            lo[k], hi[k] = state["command_ranges"][key]
        print(f"curriculum state of this checkpoint: command ranges {state['command_ranges']}"
              + (" (command curriculum)" if state.get("command_curriculum") else ""))
        if "terrain" in state:
            t = state["terrain"]
            levels_txt = ", ".join(
                f"{name} {np.dot(np.arange(t['num_levels']), c) / max(sum(c), 1):.1f}"
                + f" ({100 * c[-1] / max(sum(c), 1):.0f}% top)" for name, c in t["level_counts"].items())
            print(f"  terrain levels reached (mean of the envs, 0 .. {t['num_levels'] - 1}): {levels_txt}")
    ood = ((grid < lo - 1e-6) | (grid > hi + 1e-6)).any(dim=1).cpu()

    n_warmup = round(warmup_s / env.dt)
    n_measure = round(measure_s / env.dt)
    rec = run_rollout(env, policy, n_warmup, n_measure)
    rec.update(get_joint_limits(env))
    # [P1.6] commanded base height [T, N]: base_height_target + dh (dh = 0 without the body-height command)
    z_nominal = float(reward_cfg["base_height_target"])
    rec["z_target"] = z_nominal + (rec["cmd"][..., 3] if rec["cmd"].shape[-1] > 3 else 0.0 * rec["cmd"][..., 0])
    rec["horizon_s"] = torch.tensor(total_s)  # [P3.1] m_progress: warm-up + measurement
    rec["through_frac"] = torch.tensor(args.through)
    rec["contact_threshold"] = torch.tensor(float(reward_cfg.get("contact_force_threshold", 1.0)))  # [P3.2] pen_*

    world = "training DR" if args.keep_dr else f"nominal world, latency {args.latency_ms:.0f} ms"  # [P1.2]
    if on_terrain:  # [P3.1]
        world += f" | {env.ground.describe()}"
        world += f" | cmd {' '.join(f'{x:g}' for x in args.grid_cmd)}" if grid_mode else ""
    fall = f"fall > {args.fall_deg:.0f} deg" + (" + ground tilt" if on_terrain else "")
    title = (f"{args.exp_name} | ckpt {args.ckpt} | seed {args.seed} | init_noise {args.init_noise} rad | "
             f"{num_envs} envs | {measure_s:.0f} s | {world} | {fall} | "
             f"joints {env_cfg.get('joint_friction', 0.0):g} Nm / {env_cfg.get('joint_damping', 0.0):g} Nm s/rad")  # [S2R]
    mode_tag = "grid" if grid_mode else (f"slope{args.slope:+g}" if args.slope is not None else "")  # [P3.1]
    tag = "_".join(t for t in (mode_tag, args.tag) if t)
    name = f"ckpt{args.ckpt}_seed{args.seed}" + (f"_{tag}" if tag else "")  # [P1.6] --tag
    csv_path = os.path.join(log_dir, f"metrics_{name}.csv")
    tables = {**TABLES, **TERRAIN_TABLES} if on_terrain else TABLES
    # Genesis makes CUDA the default device for new tensors; all metrics work on the
    # CPU copies of the recordings, so new tensors must be created on the CPU too.
    with torch.device("cpu"):
        report(rec, bin_id, grid.cpu(), ood, env.dt, title, csv_path, labels=labels, tables=tables)
        if grid_mode:  # [P3.1] first and last evaluated level of each type; spectra are per command, not per tile
            picks = [b for b, lab in enumerate(labels) if lab["level"] in (min(levels), max(levels))]
            plot_gait(rec, bin_id, grid.cpu(), env.dt, os.path.join(log_dir, f"gait_{name}.png"), labels=labels,
                      bins=picks)
            fell = (rec["fall_step"] >= 0).numpy()
            ids = bin_id.cpu().numpy()
            for b, lab in enumerate(labels):
                lab["fall_share"] = float(fell[ids == b].mean())
            stats, _ = compute_stats(rec, bin_id, n_bins, env.dt, TERRAIN_TABLES)
            plot_terrain(stats, labels, os.path.join(log_dir, f"terrain_{name}.png"), title)
        else:
            plot_gait(rec, bin_id, grid.cpu(), env.dt, os.path.join(log_dir, f"gait_{name}.png"))
            plot_spectra(rec, bin_id, grid.cpu(), env.dt, os.path.join(log_dir, f"spectra_{name}.png"))  # [P1.4]

    # Determinism check: two runs with the same seed must print the same number.
    print(f"checksum: {rec['lin_vel'].double().sum().item():.10e}")

    out = os.path.join(log_dir, f"metrics_{name}.npz")
    extra = {}
    if labels:  # [P3.1] tile of each bin (go2_feet_plot.py --tile)
        extra = {"bin_terrain": np.array([lab["terrain"] for lab in labels]),
                 "bin_level": np.array([lab["level"] for lab in labels])}
    np.savez(out, bin_id=bin_id.cpu().numpy(), grid=grid.cpu().numpy(),
             **{k: v.numpy() for k, v in rec.items()}, **extra)
    print(f"raw data saved to {out}")


if __name__ == "__main__":
    main()