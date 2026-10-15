"""
go2_metrics.py - deterministic evaluation harness for Go2 locomotion policies.

Step 1 (this version): skeleton + velocity tracking error per command bin + falls.
Later metrics are added as new m_*() functions in the METRICS list (see bottom of the
"metrics" section), plus whatever raw signals they need in run_rollout().

Usage (one run = one policy = one table):
    python go2_metrics.py -e go2-walking --ckpt 100
    python go2_metrics.py -e slip-aware  --ckpt 1500
Put it next to go2_env.py (it imports Go2Env from there).
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
from go2_env import Go2Env, nominal_world


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
        "cmd": [],       # [T, N, 3]  commanded (vx, vy, wz)
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
    }

    obs = env.reset()
    with torch.no_grad():
        for t in range(n_warmup + n_measure):
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)

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
    power = (rec["torque"] * rec["dof_vel"]).abs().sum(-1)                        # [T, N]  W
    p_mean = masked_mean(power, v)                                                 # [N]
    speed = masked_mean(torch.linalg.norm(rec["lin_vel"][..., :2], dim=-1), v)    # [N]  m/s
    cot = p_mean / (rec["mass"] * G * speed.clamp(min=1e-6))
    cmd_speed = torch.linalg.norm(rec["cmd"][0, :, :2], dim=-1)                    # [N]
    cot[cmd_speed < COT_MIN_SPEED] = float("nan")          # standing / turning in place
    return {"power[W]": p_mean, "speed[m/s]": speed, "CoT[-]": cot}


CONTACT_N = 3.0   # [N] |contact force| above this = in contact (stance forces are ~20-50 N)
FOOT_R = 0.023    # [m] foot-sphere centre height when the foot is on flat ground (check_feet.py)


def runs(mask):
    """Start (inclusive) and end (exclusive) indices of each run of True in a 1-D bool array."""
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def foot_contact(rec):
    return torch.linalg.norm(rec["foot_force"], dim=-1) > CONTACT_N       # [T, N, 4]


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

    # Swing height: for each COMPLETE swing (lift-off and touchdown both inside the valid
    # window), highest foot point minus its on-ground height; averaged over all swings.
    z = rec["foot_pos"][..., 2].numpy()
    cn, n_ok = c.numpy(), n_valid.numpy()             # valid = a prefix of the window
    swing_h = np.full(c.shape[1], np.nan)
    for n in range(c.shape[1]):
        L, heights = n_ok[n], []
        for f in range(4):
            starts, ends = runs(~cn[:L, n, f])
            heights += [z[a:b, n, f].max() - FOOT_R for a, b in zip(starts, ends) if a > 0 and b < L]
        if heights:
            swing_h[n] = np.mean(heights)

    for x in (step_freq, duty):
        x[n_valid == 0] = float("nan")
    return {"duty[-]": duty, "step_freq[Hz]": step_freq,
            "swing_h[cm]": 100.0 * torch.from_numpy(swing_h).float(), "nonfoot[%time]": nonfoot}


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
        rows = torch.nonzero((grid - torch.tensor(cmd)).abs().sum(-1) < 1e-6).flatten()
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
        ax.set_ylabel(f"vx {grid[b, 0]:+.2f}\nvy {grid[b, 1]:+.2f}\nwz {grid[b, 2]:+.2f}",
                      rotation=0, ha="right", va="center", fontsize=7)
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
    "Tracking & stability": [m_tracking, m_measured],
    "Smoothness": [m_action_smoothness, m_torque_smoothness],   # [P1.4] split from "Smoothness & actuation"
    "Actuation limits": [m_torque_limits],
    "Energy": [m_cot],
    "Gait": [m_gait],
    "Foot slip": [m_slip],
}


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
    return [
        {"vx": grid[b, 0].item(), "vy": grid[b, 1].item(), "wz": grid[b, 2].item(),
         "ood": "*" if ood[b] else "",
         "falls": f"{int(fell[bin_id == b].sum())}/{int((bin_id == b).sum())}"}
        for b in range(len(grid))
    ]


def print_table(rows, title):
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c])) for r in rows)) for c in cols}
    fmt = lambda c, v: (f"{v:.2f}" if isinstance(v, float) else str(v)).rjust(widths[c])
    print(f"\n--- {title} ---")
    print("  ".join(c.rjust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(fmt(c, r[c]) for c in cols))


def report(rec, bin_id, grid, ood, dt, title, csv_path):
    stats, table_cols = compute_stats(rec, bin_id, len(grid), dt)
    header = bin_header(rec, bin_id, grid, ood)

    print(f"\n=== {title} ===")
    for i, (tname, names) in enumerate(table_cols.items()):
        rows = []
        for b, h in enumerate(header):
            row = dict(h) if i == 0 else {k: h[k] for k in ("vx", "vy", "wz", "ood")}
            for n in names:
                m, s = stats[n][0][b], stats[n][1][b]
                row[n] = f"{m:.3f} ± {s:.3f}" if np.isfinite(m) else "n/a"
            rows.append(row)
        print_table(rows, tname)
    print("(* = command outside the training command range; n/a = not defined or no valid data)")

    # Same numbers as a CSV (mean and std in separate numeric columns) for plots / comparisons.
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        names = list(stats.keys())
        w.writerow(["vx", "vy", "wz", "ood", "falls"] + [f"{n}_{s}" for n in names for s in ("mean", "std")])
        for b, h in enumerate(header):
            w.writerow([h["vx"], h["vy"], h["wz"], h["ood"] == "*", h["falls"]]
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
    env_cfg["termination_if_roll_greater_than"] = args.fall_deg    # [P1.2] same fall criterion for every run
    env_cfg["termination_if_pitch_greater_than"] = args.fall_deg

    total_s = args.warmup_s + args.measure_s
    env_cfg["episode_length_s"] = total_s + 10.0           # no timeouts during the run
    env_cfg["resampling_time_s"] = total_s + 10.0          # irrelevant (pinned), but harmless
    env_cfg["links_to_keep"] = FEET                        # keep foot links (needs go2_env.py edit)

    # Env layout: bins repeated `reps` times -> env i belongs to bin bin_id[i]
    grid = torch.tensor(COMMAND_GRID, dtype=gs.tc_float, device=gs.device)
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

    world = "training DR" if args.keep_dr else f"nominal world, latency {args.latency_ms:.0f} ms"  # [P1.2]
    title = (f"{args.exp_name} | ckpt {args.ckpt} | seed {args.seed} | init_noise {args.init_noise} rad | "
             f"{num_envs} envs | {args.measure_s:.0f} s | {world} | fall > {args.fall_deg:.0f} deg")
    csv_path = os.path.join(log_dir, f"metrics_ckpt{args.ckpt}_seed{args.seed}.csv")
    # Genesis makes CUDA the default device for new tensors; all metrics work on the
    # CPU copies of the recordings, so new tensors must be created on the CPU too.
    with torch.device("cpu"):
        report(rec, bin_id, grid.cpu(), ood, env.dt, title, csv_path)
        plot_gait(rec, bin_id, grid.cpu(), env.dt,
                  os.path.join(log_dir, f"gait_ckpt{args.ckpt}_seed{args.seed}.png"))
        plot_spectra(rec, bin_id, grid.cpu(), env.dt,                                     # [P1.4]
                     os.path.join(log_dir, f"spectra_ckpt{args.ckpt}_seed{args.seed}.png"))

    # Determinism check: two runs with the same seed must print the same number.
    print(f"checksum: {rec['lin_vel'].double().sum().item():.10e}")

    out = os.path.join(log_dir, f"metrics_ckpt{args.ckpt}_seed{args.seed}.npz")
    np.savez(out, bin_id=bin_id.cpu().numpy(), grid=grid.cpu().numpy(),
             **{k: v.numpy() for k, v in rec.items()})
    print(f"raw data saved to {out}")


if __name__ == "__main__":
    main()