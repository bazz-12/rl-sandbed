"""
go2_fsm_sim.py - [P1.5] Genesis test bench for the deploy-side state machine (go2_fsm.py).

The robot starts lying in damping mode, then a scripted gamepad sequence runs:
    full  (default)  L1+R2 stand up -> L1+A policy -> left stick forward -> D-pad down x3 (body height -3 cm, for
                     policies trained with the body-height command; others ignore it) -> L1+R1 sit down
    estop            L1+R2 stand up -> L1+A policy -> L1+Y damping (software e-stop) while standing
The bench applies the state machine's motor commands at the physics rate, as the motor drivers would:
    tau = kp (q_des - q) + kd (dq_des - dq) + tau_ff, clipped at the Go2 torque limits.
It also checks that go2_fsm.Policy builds the same observation as Go2Env. It prints one line per phase and
saves a plot.

    python go2_fsm_sim.py                       # zero-action policy: tests stand-up, hand-over and sit-down
    python go2_fsm_sim.py -e go2-p15            # exported policy (python go2_export.py -e go2-p15 --ckpt N)
    python go2_fsm_sim.py --scenario estop
    python go2_fsm_sim.py -n 16 --dr            # 16 robots with the training randomization (all are summarized)
    python go2_fsm_sim.py --handover-s 0        # switch to the policy gains at once instead of ramping them
    python go2_fsm_sim.py --viewer
"""

import argparse
import os
import pickle

import numpy as np
import torch

import genesis as gs
from genesis.utils.geom import inv_quat, transform_by_quat

from go2_env import Go2Env, nominal_world
from go2_export import make_deploy_cfg
from go2_fsm import LIE_POSE_LEG, FsmConfig, Go2StateMachine, Mode, Policy

# Gamepad script: (time [s], buttons, held for PRESS_S) and (start, end, sticks (lx, ly, rx, ry))
SCENARIOS = {
    "full": {"buttons": [(1.0, ("L1", "R2")), (4.0, ("L1", "A")),
                         (8.6, ("down",)), (8.75, ("down",)), (8.9, ("down",)),  # [P1.6] body height -3 cm
                         (10.0, ("L1", "R1"))],
             "sticks": [(6.0, 8.5, (0.0, 0.6, 0.0, 0.0))], "end": 13.5},
    "estop": {"buttons": [(1.0, ("L1", "R2")), (4.0, ("L1", "A")), (6.5, ("L1", "Y"))],
              "sticks": [], "end": 9.0},
}
PRESS_S = 0.1
LIE_BASE_HEIGHT = 0.15  # [m] initial base height; the robot then settles in damping mode
MODES = list(Mode)


def buttons_at(scn, t):
    return {b for t0, combo in scn["buttons"] if t0 <= t < t0 + PRESS_S for b in combo}


def sticks_at(scn, t):
    return next((s for t0, t1, s in scn["sticks"] if t0 <= t < t1), (0.0, 0.0, 0.0, 0.0))


def env_observation(env, gyro, quat, q, dq, cmd, last_action):
    """Observation of env 0 computed by Go2Env itself, from the same robot state (reference for the check)."""
    dev, f = gs.device, gs.tc_float
    quat = torch.as_tensor(quat, dtype=f, device=dev)
    env.base_ang_vel[0] = torch.as_tensor(gyro, dtype=f, device=dev)
    env.projected_gravity[0] = transform_by_quat(env.global_gravity, inv_quat(quat))
    env.commands[0] = torch.as_tensor(cmd, dtype=f, device=dev)
    env.dof_pos[0] = torch.as_tensor(q, dtype=f, device=dev)
    env.dof_vel[0] = torch.as_tensor(dq, dtype=f, device=dev)
    env.actions[0] = torch.as_tensor(last_action, dtype=f, device=dev)
    env._update_observation()
    return env.obs_buf[0].cpu().numpy()


def phase_segments(t, mode0):
    """Contiguous [start, end) index ranges of env 0's mode."""
    cuts = [0] + [k for k in range(1, len(mode0)) if mode0[k] != mode0[k - 1]] + [len(mode0)]
    return [(cuts[i], cuts[i + 1], MODES[mode0[cuts[i]]]) for i in range(len(cuts) - 1)]


def summarize(log, dt, scn_name, fsm0):
    t, z, tilt, tau, err, vz, mode = (log[k] for k in ("t", "z", "tilt", "tau", "err", "vz", "mode"))
    print(f"\n=== phases (env 0 timeline; values over all {z.shape[1]} envs) ===")
    print(f"{'mode':9s} {'t [s]':>13s} {'z start':>8s} {'z end min/max':>14s} {'tilt max':>9s} "
          f"{'tau peak':>9s} {'|q*-q| max':>11s}")
    for a, b, m in phase_segments(t, mode[:, 0]):
        print(f"{m.value:9s} {t[a]:6.2f}-{t[b - 1] + dt:5.2f} {z[a].mean():8.3f} "
              f"{z[b - 1].min():6.3f}/{z[b - 1].max():.3f} {tilt[a:b].max():7.1f} d {100 * tau[a:b].max():7.0f} % "
              f"{err[a:b].max():9.2f} r")
    seg = {m: (a, b) for a, b, m in phase_segments(t, mode[:, 0])}  # last segment of each mode
    print("\n=== checks ===")
    if Mode.STAND_UP in seg:
        a, b = seg[Mode.STAND_UP]
        ok = (z[b - 1] > 0.25) & (tilt[b - 1] < 10.0)
        print(f"stand-up: {int(ok.sum())}/{len(ok)} robots standing (z > 0.25 m, tilt < 10 deg) at the end; "
              f"z = {z[b - 1].min():.3f}..{z[b - 1].max():.3f} m")
    if Mode.POLICY in seg:
        a, b = seg[Mode.POLICY]
        w = slice(a, min(b, a + round(1.5 / dt)))
        drop = z[a] - z[w].min(0)
        print(f"hand-over (first 1.5 s of policy mode): base drop {100 * drop.max():.1f} cm max, "
              f"tilt {tilt[w].max():.1f} deg max, tau peak {100 * tau[w].max():.0f} % of limit")
    if scn_name == "full" and Mode.SIT_DOWN in seg:
        a, b = seg[Mode.SIT_DOWN]
        print(f"sit-down: base sink speed {100 * (-vz[a:b]).max():.0f} cm/s max, z at the end "
              f"{z[b - 1].min():.3f}..{z[b - 1].max():.3f} m, tilt {tilt[a:b].max():.1f} deg max")
    if scn_name == "estop":
        t_stop = next(tt for tt, msg in fsm0.events if "e-stop" in msg)
        k = round(t_stop / dt)
        print(f"e-stop at {t_stop:.2f} s: base sink speed {100 * (-vz[k:]).max():.0f} cm/s max, "
              f"z after 1 s {z[min(k + round(1.0 / dt), len(t) - 1)].max():.3f} m (max over robots), "
              f"tilt {tilt[k:].max():.1f} deg max")


def plot(log, dt, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, mode0 = log["t"], log["mode"][:, 0]
    ink, muted, hairline, band = "#0b0b0b", "#52514e", "#e6e5e0", "#f0efec"
    color = "#2a78d6"
    panels = (("base height [m]", log["z"]), ("base tilt [deg]", log["tilt"]),
              ("peak joint torque [% of limit]", 100 * log["tau"]))
    fig, axes = plt.subplots(3, 1, figsize=(9, 6.2), sharex=True)
    for ax, (label, y) in zip(axes, panels):
        for i, (a, b, m) in enumerate(phase_segments(t, mode0)):
            if i % 2:
                ax.axvspan(t[a], t[b - 1] + dt, color=band, linewidth=0)
        if y.shape[1] > 1:  # all robots: min-max band
            ax.fill_between(t, y.min(1), y.max(1), color=color, alpha=0.12, linewidth=0)
        ax.plot(t, y[:, 0], color=color, linewidth=1.6)
        ax.set_ylabel(label, fontsize=8, color=ink)
        ax.tick_params(labelsize=7, colors=muted)
        ax.grid(True, axis="y", color=hairline, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for a, b, m in phase_segments(t, mode0):
        axes[0].annotate(m.value.replace("_", " "), ((t[a] + t[b - 1]) / 2, 1.0), xycoords=("data", "axes fraction"),
                         xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7, color=muted)
    axes[-1].set_xlabel("time [s]", fontsize=8, color=ink)
    axes[-1].set_xlim(0, t[-1] + dt)
    fig.suptitle(title, fontsize=9, color=ink, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"plot saved to {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--exp_name", type=str, default=None, help="run with an exported policy (go2_export.py)")
    p.add_argument("--log_dir", type=str, default=None, help="default: logs/<exp_name>")
    p.add_argument("--scenario", choices=list(SCENARIOS), default="full")
    p.add_argument("-n", "--num_envs", type=int, default=1)
    p.add_argument("--dr", action="store_true", help="training randomization: friction, mass, CoM, motor strength")
    p.add_argument("--handover-s", type=float, default=FsmConfig.handover_s)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--out", type=str, default=None, help="plot path (default: fsm_<scenario>.png next to the run)")
    args = p.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, logging_level="warning", seed=args.seed)
    log_dir = args.log_dir or (os.path.join("logs", args.exp_name) if args.exp_name else None)
    if log_dir:
        with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
            env_cfg, obs_cfg, reward_cfg, command_cfg, _ = pickle.load(f)
    else:
        import go2_train

        env_cfg, obs_cfg, reward_cfg, command_cfg = go2_train.get_cfgs()
    reward_cfg["reward_scales"] = {}
    if not args.dr:
        nominal_world(env_cfg, obs_cfg)
    obs_cfg["add_noise"] = False  # the observation check compares noise-free observations
    # The bench runs its own control loop: no termination, no reset, no pushes, no fallen starts
    env_cfg.update(termination_if_roll_greater_than=1e9, termination_if_pitch_greater_than=1e9,
                   terminate_after_contacts_on=[], episode_length_s=1e6, recovery={})
    env = Go2Env(args.num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=args.viewer)
    N = env.num_envs

    export_dir = os.path.join(log_dir, "exported") if log_dir else None
    if export_dir and not os.path.isfile(os.path.join(export_dir, "policy.pt")):
        raise SystemExit(f"{export_dir}/policy.pt not found: run go2_export.py -e {args.exp_name} --ckpt N first")
    deploy_cfg = make_deploy_cfg(env)
    fsm_cfg = FsmConfig(handover_s=args.handover_s)
    fsms = [Go2StateMachine(Policy(export_dir) if export_dir else Policy(deploy_cfg=deploy_cfg), fsm_cfg,
                            verbose=(i == 0)) for i in range(N)]
    # The exported deploy_cfg.json must describe the same robot interface as this env
    exported = fsms[0].policy.cfg
    for key in ("joint_names", "default_joint_angles", "kp", "kd", "action_scale", "commands_scale", "obs_scales"):
        a, b = exported[key], deploy_cfg[key]
        same = a == b if isinstance(a, (str, dict)) or key == "joint_names" else np.allclose(a, b)
        if not same:
            raise SystemExit(f"deploy_cfg.json and the env disagree on {key}: {a} vs {b}")

    # Lying start: lying pose, base at LIE_BASE_HEIGHT, then damping mode settles it on the ground
    kinds = {"hip": 0, "thigh": 1, "calf": 2}
    lie_urdf = [LIE_POSE_LEG[kinds[j.name.split("_")[1]]] for j in env.robot.joints[1:]]  # URDF joint order
    qpos = env.init_qpos.clone()
    qpos[2] = LIE_BASE_HEIGHT
    qpos[7:] = torch.tensor(lie_urdf, dtype=gs.tc_float, device=gs.device)
    env.robot.set_qpos(qpos, zero_velocity=True)

    scn = SCENARIOS[args.scenario]
    dt, n_steps = env.sim_dt, round(scn["end"] / env.sim_dt)
    motors = env.motors_dof_idx
    log = {k: np.zeros((n_steps, N)) for k in ("z", "tilt", "tau", "err", "vz")}
    log["mode"] = np.zeros((n_steps, N), dtype=int)
    log["t"] = np.arange(n_steps) * dt
    obs_err, n_policy_steps, n_checked = 0.0, 0, 0
    for k in range(n_steps):
        t = k * dt
        q = env.robot.get_dofs_position(motors)
        dq = env.robot.get_dofs_velocity(motors)
        quat = env.robot.get_quat()
        inv = inv_quat(quat)
        gyro = transform_by_quat(env.robot.get_ang(), inv)
        qn, dqn, quatn, gyron = (x.cpu().numpy().astype(float) for x in (q, dq, quat, gyro))
        buttons, sticks = buttons_at(scn, t), sticks_at(scn, t)
        cmds = [f.update(t, qn[i], dqn[i], gyron[i], quatn[i], buttons, sticks) for i, f in enumerate(fsms)]
        q_des, dq_des, kp, kd, tau_ff = (torch.as_tensor(np.stack(c), dtype=gs.tc_float, device=gs.device)
                                         for c in zip(*cmds))
        tau = env.motor_strength * (kp * (q_des - q) + kd * (dq_des - dq)) + tau_ff
        tau = torch.clip(tau, -env.torque_limits, env.torque_limits)
        env.robot.control_dofs_force(tau[:, env.actions_dof_idx], env.motors_dof_slice)

        # Observation check on env 0: deploy-side observation vs Go2Env's, from the same state
        pol = fsms[0].policy
        if pol.steps != n_policy_steps:  # the policy ran during this update
            n_policy_steps = pol.steps
            ref = env_observation(env, gyron[0], quatn[0], qn[0], dqn[0], fsms[0].command, pol.last_obs[-12:])
            obs_err = max(obs_err, float(np.abs(ref - pol.last_obs).max()))
            n_checked += 1

        env.scene.step()

        g = transform_by_quat(env.global_gravity, inv)
        log["z"][k] = env.robot.get_pos()[:, 2].cpu().numpy()
        log["vz"][k] = env.robot.get_vel()[:, 2].cpu().numpy()
        log["tilt"][k] = np.degrees(np.arccos(np.clip(-g[:, 2].cpu().numpy(), -1.0, 1.0)))
        log["tau"][k] = (tau.abs() / env.torque_limits).max(-1).values.cpu().numpy()
        log["err"][k] = (q_des - q).abs().max(-1).values.cpu().numpy()
        log["mode"][k] = [MODES.index(f.mode) for f in fsms]

    print(f"\nobservation check: {n_checked} policy steps, max |deploy obs - Go2Env obs| = {obs_err:.2e}")
    policy_name = f"{args.exp_name} (exported)" if export_dir else "zero-action policy"
    summarize(log, dt, args.scenario, fsms[0])
    out = args.out or os.path.join(log_dir or ".", f"fsm_{args.scenario}.png")
    title = (f"{args.scenario} | {policy_name} | {N} robot{'s' if N > 1 else ''}"
             f"{' with training randomization' if args.dr else ''} | hand-over ramp {args.handover_s:.1f} s")
    plot(log, dt, out, title)


if __name__ == "__main__":
    main()
