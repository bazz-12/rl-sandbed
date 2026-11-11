"""
go2_feet_plot.py - [P2] Foot trajectories of one env from a go2_metrics.py recording (metrics_*.npz). No Genesis needed.

    python go2_feet_plot.py logs/go2-p22/metrics_ckpt1999_seed0.npz --cmd 0.5 0 0 --target 0.075
    python go2_feet_plot.py logs/go2-p22/metrics_ckpt1999_seed0.npz --cmd 0 0.3 0 --env 3 --seconds 3

One row per foot (FL, FR, RL, RR), shared time axis:
    clearance [cm]   height of the sphere bottom above the ground (what swing_h measures at its peak), with the
                     --target clearance as a dashed line (P2.2), stance shaded
    speed [m/s]      horizontal speed of the contact point (bottom of the sphere: centre velocity + omega x r, as
                     slip_v in go2_metrics.py): the swing motion, and in stance the slip. A dragged toe shows either
                     as low AND fast outside the shaded stance (light contact, < 3 N), or as speed inside it (slip)
Read the swing shape (P2.2: flat top near the target, no scuffing before touchdown), the timing (P2.1: swing length,
P2.4: diagonal pairs FL-RR and FR-RL lifting together) and left / right differences (P2.3: FL vs FR, RL vs RR).
"""

import argparse
import os

import numpy as np

FEET = ["FL", "FR", "RL", "RR"]
CONTACT_N = 3.0  # [N] as go2_metrics.py
FOOT_R = 0.023  # [m] as go2_metrics.py (go2_env.GO2_FOOT_RADIUS)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz", help="metrics_ckpt<N>_seed<S>[_tag].npz written by go2_metrics.py")
    p.add_argument("--cmd", type=float, nargs="+", default=[0.5, 0.0, 0.0], metavar="V", help="bin: VX VY WZ [DH]")
    p.add_argument("--env", type=int, default=0, help="which env of the bin (0 = first)")
    p.add_argument("--seconds", type=float, default=2.0, help="length of the plotted window [s]")
    p.add_argument("--target", type=float, default=None, help="clearance target [m] to draw (e.g. 0.075)")
    p.add_argument("--dt", type=float, default=0.02, help="policy period [s]")
    p.add_argument("--out", type=str, default=None, help="default: feet_<cmd>_env<i>.png next to the npz")
    args = p.parse_args()

    d = np.load(args.npz)
    grid, bin_id = d["grid"], d["bin_id"]
    cmd = np.array(args.cmd + [0.0] * (grid.shape[1] - len(args.cmd)))[: grid.shape[1]]
    rows = np.flatnonzero(np.abs(grid - cmd).sum(1) < 1e-6)
    if rows.size == 0:
        raise SystemExit(f"no bin {cmd} in {args.npz}; bins: {[tuple(np.round(g, 3)) for g in grid]}")
    envs = np.flatnonzero(bin_id == rows[0])
    n = envs[min(args.env, envs.size - 1)]
    valid = d["valid"][:, n]
    T = min(int(valid.sum()), round(args.seconds / args.dt))
    if T < 2:
        raise SystemExit(f"env {n} fell at the start of the measurement: pick another --env")
    t = np.arange(T) * args.dt
    z = d["foot_pos"][:T, n, :, 2] - FOOT_R  # [T, 4] clearance
    v, w = d["foot_vel"][:T, n], d["foot_ang"][:T, n]  # [T, 4, 3] centre velocity, angular velocity
    speed = np.hypot(v[..., 0] - FOOT_R * w[..., 1], v[..., 1] + FOOT_R * w[..., 0])  # contact point, as m_slip
    contact = np.linalg.norm(d["foot_force"][:T, n], axis=-1) > CONTACT_N  # [T, 4]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, hairline, band = "#0b0b0b", "#52514e", "#e6e5e0", "#ecebe6"
    c_z, c_v = "#2a78d6", "#eb6834"
    fig, axes = plt.subplots(4, 1, figsize=(9, 7), sharex=True)
    twins = []
    for f, ax in enumerate(axes):
        for a, b in zip(*np.flatnonzero(np.diff(np.concatenate(([0], contact[:, f], [0])))).reshape(-1, 2).T):
            ax.axvspan(t[a], t[min(b, T - 1)], color=band, linewidth=0)
        ax.plot(t, 100 * z[:, f], color=c_z, linewidth=1.6)
        if args.target is not None:
            ax.axhline(100 * args.target, color=c_z, linewidth=0.8, linestyle=(0, (4, 3)))
        ax.set_ylabel(f"{FEET[f]}\ncm", fontsize=8, color=ink, rotation=0, ha="right", va="center")
        ax.set_ylim(bottom=-0.5)
        ax2 = ax.twinx()
        twins.append(ax2)
        ax2.plot(t, speed[:, f], color=c_v, linewidth=1.0, linestyle="--")
        ax2.set_ylim(bottom=0.0)
        ax2.tick_params(labelsize=7, colors=muted)
        ax.tick_params(labelsize=7, colors=muted)
        ax.grid(True, axis="y", color=hairline, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)
    axes[0].legend([axes[0].lines[0], twins[0].lines[0]],
                   ["clearance [cm] (left axis)", "contact-point speed [m/s] (right axis, dashed)"],
                   frameon=False, fontsize=7, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
    axes[-1].set_xlabel("time since start of measurement [s]  (shaded = stance, |F| > 3 N)", fontsize=8, color=ink)
    title = f"{os.path.basename(args.npz)} | cmd {' '.join(f'{x:+.2f}' for x in cmd)} | env {n}"
    if args.target is not None:
        title += f" | dashed: target {100 * args.target:.1f} cm"
    fig.suptitle(title, fontsize=9, color=ink, x=0.01, ha="left")
    fig.tight_layout()
    out = args.out or os.path.join(os.path.dirname(args.npz) or ".",
                                   f"feet_{'_'.join(f'{x:g}' for x in cmd)}_env{n}.png")
    fig.savefig(out, dpi=150)
    print(f"foot plot saved to {out}")


if __name__ == "__main__":
    main()
