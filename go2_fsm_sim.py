"""
go2_fsm_sim.py - [P1.5] Genesis test bench for the deploy-side state machine (go2_fsm.py).

The robot starts lying in damping mode, then a scripted gamepad sequence runs:
    full  (default)  L1+R2 stand up -> L1+A policy -> left stick forward -> D-pad down x3 (body height -3 cm, for
                     policies trained with the body-height command; others ignore it) -> L1+R1 sit down
    estop            L1+R2 stand up -> L1+A policy -> L1+Y damping (software e-stop) while standing
    [S2R] stand, policy, walk, estop_walk, restand, joint_test: the go2_inputs.SCENARIOS that go2_deploy_sdk2.py --sim
                     runs in unitree_mujoco, so the two simulators run the same sequence (walk needs --profile)
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
[S2R]
    python go2_fsm_sim.py -e go2-p15 --scenario walk --profile fwd_back --log     # same run as in unitree_mujoco
    python go2_fsm_sim.py --scenario joint_test --suspended 0.6 --log             # robot hung: actuator check
    python go2_fsm_sim.py -e go2-p15 --scenario policy --latency-ms 10 --log      # motor commands 10 ms late
    python go2_fsm_sim.py -e go2-p15 --scenario walk --profile fwd_back --friction 0.4 --log  # unitree_mujoco's feet
--log writes logs/deploy/genesis_<date>_<scenario>.npz (go2_log.py, env 0, with ground truth): go2_analyze.py compares
it with the unitree_mujoco and robot logs of the same scenario.
"""

import argparse
import os
import pickle
import time
from collections import deque

import numpy as np
import torch

import genesis as gs
from genesis.utils.geom import inv_quat, transform_by_quat

import go2_log
from go2_env import Go2Env, nominal_world
from go2_export import make_deploy_cfg
from go2_fsm import LIE_POSE_LEG, MODES, FsmConfig, Go2StateMachine, Mode, Policy
from go2_inputs import PROFILES, SCENARIOS, CommandProfile, Scenario, buttons_mask

# Gamepad script: (time [s], buttons, held for PRESS_S) and (start, end, sticks (lx, ly, rx, ry))
LOCAL_SCENARIOS = {
    "full": {"buttons": [(1.0, ("L1", "R2")), (4.0, ("L1", "A")),
                         (8.6, ("down",)), (8.75, ("down",)), (8.9, ("down",)),  # [P1.6] body height -3 cm
                         (10.0, ("L1", "R1"))],
             "sticks": [(6.0, 8.5, (0.0, 0.6, 0.0, 0.0))], "end": 13.5},
}
PRESS_S = 0.1
LIE_BASE_HEIGHT = 0.15  # [m] initial base height; the robot then settles in damping mode


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
    print(f"{'mode':10s} {'t [s]':>13s} {'z start':>8s} {'z end min/max':>14s} {'tilt max':>9s} "
          f"{'tau peak':>9s} {'|q*-q| max':>11s}")
    for a, b, m in phase_segments(t, mode[:, 0]):
        print(f"{m.value:10s} {t[a]:6.2f}-{t[b - 1] + dt:5.2f} {z[a].mean():8.3f} "
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
    if scn_name.startswith("estop"):
        t_stop = next((tt for tt, msg in fsm0.events if "e-stop" in msg), None)
        if t_stop is not None:
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
    p.add_argument("--scenario", choices=list(LOCAL_SCENARIOS) + list(SCENARIOS), default="full")
    p.add_argument("-n", "--num_envs", type=int, default=1)
    p.add_argument("--dr", action="store_true", help="training randomization: friction, mass, CoM, motor strength")
    p.add_argument("--handover-s", type=float, default=FsmConfig.handover_s)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--out", type=str, default=None, help="plot path (default: fsm_<scenario>.png next to the run)")
    # [S2R]
    p.add_argument("--profile", choices=list(PROFILES), default=None, help="command profile started by L1+X")
    p.add_argument("--suspended", type=float, default=None, metavar="Z",
                   help="base pinned at height Z [m], legs free (the stand); allows the joint test")
    p.add_argument("--latency-ms", type=float, default=0.0, help="motor commands applied this late (whole steps)")
    p.add_argument("--joint-damping", type=float, default=None, help="viscous joint damping [Nm s/rad] (go2.xml: 0.1)")
    p.add_argument("--joint-friction", type=float, default=None, help="joint Coulomb friction [Nm] (go2.xml: 0.2)")
    p.add_argument("--armature", type=float, default=None, help="rotor inertia [kg m2] (training: env_cfg armature)")
    p.add_argument("--friction", type=float, default=None, metavar="MU",
                   help="ground-foot friction (default 1.0; unitree_mujoco's feet: 0.4)")
    p.add_argument("--log", nargs="?", const="", default=None, metavar="FILE",
                   help="go2_log.py log of env 0 (default name: logs/deploy/genesis_<date>_<scenario>.npz)")
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
    if args.armature is not None:  # [S2R]
        env_cfg["armature"] = args.armature
    env = Go2Env(args.num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=args.viewer)
    N = env.num_envs
    # [S2R] joint losses that the training model does not have (unitree_mujoco's go2.xml: damping 0.1, friction 0.2)
    if args.joint_damping is not None:
        env.robot.set_dofs_damping(torch.full((env.num_actions,), args.joint_damping), env.motors_dof_idx)
    if args.joint_friction is not None:
        env.robot.set_dofs_frictionloss(torch.full((env.num_actions,), args.joint_friction), env.motors_dof_idx)
    if args.friction is not None:  # [S2R] as Go2Env's friction randomization: ground and robot get the same ratio
        n_links = env.robot.n_links
        env.friction[:] = args.friction
        env.robot.set_friction_ratio(env.friction[:, None].expand(N, n_links).contiguous(), list(range(n_links)))
        env.plane.set_friction_ratio(env.friction[:, None], [0])

    export_dir = os.path.join(log_dir, "exported") if log_dir else None
    if export_dir and not os.path.isfile(os.path.join(export_dir, "policy.pt")):
        raise SystemExit(f"{export_dir}/policy.pt not found: run go2_export.py -e {args.exp_name} --ckpt N first")
    deploy_cfg = make_deploy_cfg(env)
    fsm_cfg = FsmConfig(handover_s=args.handover_s, allow_joint_test=args.suspended is not None)
    fsms = [Go2StateMachine(Policy(export_dir) if export_dir else Policy(deploy_cfg=deploy_cfg), fsm_cfg,
                            verbose=(i == 0)) for i in range(N)]
    # The exported deploy_cfg.json must describe the same robot interface as this env
    exported = fsms[0].policy.cfg
    for key in ("joint_names", "default_joint_angles", "kp", "kd", "action_scale", "commands_scale", "obs_scales"):
        a, b = exported[key], deploy_cfg[key]
        same = a == b if isinstance(a, (str, dict)) or key == "joint_names" else np.allclose(a, b)
        if not same:
            raise SystemExit(f"deploy_cfg.json and the env disagree on {key}: {a} vs {b}")
    profile = CommandProfile(args.profile, fsms[0].num_commands) if args.profile else None  # [S2R]
    for f in fsms:
        f.load_profile(profile)

    # [S2R] the shared scenarios (go2_inputs.py) as the bench's button / stick script
    if args.scenario in LOCAL_SCENARIOS:
        scn = LOCAL_SCENARIOS[args.scenario]
    else:
        shared = Scenario(args.scenario, profile, fsms[0].joint_test_duration())
        scn = {"buttons": [(t0, fsm_cfg.combos[c]) for t0, c in shared.presses], "sticks": [],
               "end": shared.duration}
    if "joint_test" in args.scenario and args.suspended is None:
        p.error("the joint test needs the robot hung: add --suspended 0.6")

    # Lying start: lying pose, base at LIE_BASE_HEIGHT, then damping mode settles it on the ground
    kinds = {"hip": 0, "thigh": 1, "calf": 2}
    lie_urdf = [LIE_POSE_LEG[kinds[j.name.split("_")[1]]] for j in env.robot.joints[1:]]  # URDF joint order
    qpos = env.init_qpos.clone()
    qpos[2] = LIE_BASE_HEIGHT if args.suspended is None else args.suspended
    qpos[7:] = torch.tensor(lie_urdf, dtype=gs.tc_float, device=gs.device)
    env.robot.set_qpos(qpos, zero_velocity=True)
    # [S2R] --suspended: base pose and velocity reset before every physics step (pinned), legs free
    pin_pos = qpos[:3].repeat(N, 1)
    pin_quat = qpos[3:7].repeat(N, 1)
    base_dofs = list(range(env.robot.joints[0].dof_start, env.robot.joints[0].dof_start + 6))

    dt, n_steps = env.sim_dt, round(scn["end"] / env.sim_dt)
    motors = env.motors_dof_idx
    log = {k: np.zeros((n_steps, N)) for k in ("z", "tilt", "tau", "err", "vz")}
    log["mode"] = np.zeros((n_steps, N), dtype=int)
    log["t"] = np.arange(n_steps) * dt
    obs_err, n_policy_steps, n_checked = 0.0, 0, 0
    # [S2R] motor-command latency: a queue of whole physics steps
    delay = int(round(args.latency_ms / 1000.0 / dt))
    pending = deque()
    recorder = None
    if args.log is not None:
        path = args.log or go2_log.default_path("logs/deploy", "genesis", args.scenario)
        recorder = go2_log.Recorder(
            path, "genesis", max_s=scn["end"] + 5, ctrl_hz=1.0 / dt, pol_hz=1.0 / deploy_cfg["policy_dt"],
            truth_hz=1.0 / dt, n_obs=fsms[0].policy.n_obs, n_cmd=fsms[0].num_commands,
            meta={"deploy_cfg": exported, "fsm_cfg": fsms[0].config_dict(), "args": vars(args),
                  "policy_dir": os.path.abspath(export_dir) if export_dir else None, "zero_action": export_dir is None,
                  "control_dt": dt, "profile": profile.describe() if profile else None, "profile_name": args.profile,
                  "scenario": args.scenario, "joint_test": fsms[0].joint_test.describe(),
                  "modes": [m.value for m in MODES], "truth_kind": "full", "dr": args.dr,
                  "suspended": args.suspended, "latency_ms": delay * dt * 1000.0,
                  "joint_damping": args.joint_damping, "joint_friction": args.joint_friction,
                  "armature": env_cfg.get("armature"),
                  "friction": float(env.friction[0]), "added_mass": float(env.added_mass[0])})
    # [S2R] feet in the log's leg order (FR, FL, RR, RL, as the joints and the SDK), not Go2Env's FOOT_LINKS order
    feet = [env.robot.get_link(f"{leg}_foot").idx_local for leg in ("FR", "FL", "RR", "RL")]
    t_wall = time.monotonic()
    for k in range(n_steps):
        t = k * dt
        q = env.robot.get_dofs_position(motors)
        dq = env.robot.get_dofs_velocity(motors)
        quat = env.robot.get_quat()
        inv = inv_quat(quat)
        gyro = transform_by_quat(env.robot.get_ang(), inv)
        qn, dqn, quatn, gyron = (x.cpu().numpy().astype(float) for x in (q, dq, quat, gyro))
        buttons, sticks = buttons_at(scn, t), sticks_at(scn, t)
        t_cpu = time.perf_counter()
        cmds = [f.update(t, qn[i], dqn[i], gyron[i], quatn[i], buttons, sticks) for i, f in enumerate(fsms)]
        t_cpu = time.perf_counter() - t_cpu
        pending.append(cmds)
        applied = pending.popleft() if len(pending) > delay else pending[0]
        q_des, dq_des, kp, kd, tau_ff = (torch.as_tensor(np.stack(c), dtype=gs.tc_float, device=gs.device)
                                         for c in zip(*applied))
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

        if recorder is not None:  # [S2R] env 0, state before this physics step and the command it gets
            f0 = fsms[0]
            contact = env.robot.get_links_net_contact_force()[0, feet].cpu().numpy()  # [4, 3] world
            flags = (go2_log.FLAG_CLAMPED * f0.last_clamped) | (go2_log.FLAG_PROFILE * (f0.profile_t0 is not None)) \
                | (go2_log.FLAG_SCRIPTED * bool(buttons))
            tau0 = tau[0].cpu().numpy()
            recorder.append("ctrl", t=t, mode=MODES.index(f0.mode), q=qn[0], dq=dqn[0], tau_est=tau0, quat=quatn[0],
                            gyro=gyron[0], foot_force=contact[:, 2], q_des=cmds[0][0], dq_des=cmds[0][1],
                            kp=cmds[0][2], kd=cmds[0][3], tau_ff=cmds[0][4], tau_cmd=f0.last_tau_cmd, tick=k,
                            state_age=0.0, loop_dt=dt, compute=t_cpu / N, buttons=buttons_mask(buttons),
                            sticks=sticks, flags=flags)
            if f0.policy_stepped:
                recorder.append("pol", t=t, obs=pol.last_obs, action=pol.last_action, cmd=f0.command,
                                infer=pol.last_infer_s, seg=f0.profile_seg)
            recorder.append("truth", t=t, pos=env.robot.get_pos()[0].cpu().numpy(), quat=quatn[0],
                            lin_vel=env.robot.get_vel()[0].cpu().numpy(), ang_vel=gyron[0], q=qn[0], dq=dqn[0],
                            tau=tau0, foot_force=contact,
                            foot_pos=env.robot.get_links_pos()[0, feet].cpu().numpy())

        if args.suspended is not None:  # [S2R]
            env.robot.set_pos(pin_pos, zero_velocity=False)
            env.robot.set_quat(pin_quat, zero_velocity=False)
            env.robot.set_dofs_velocity(None, base_dofs)
        env.scene.step()

        g = transform_by_quat(env.global_gravity, inv)
        log["z"][k] = env.robot.get_pos()[:, 2].cpu().numpy()
        log["vz"][k] = env.robot.get_vel()[:, 2].cpu().numpy()
        log["tilt"][k] = np.degrees(np.arccos(np.clip(-g[:, 2].cpu().numpy(), -1.0, 1.0)))
        log["tau"][k] = (tau.abs() / env.torque_limits).max(-1).values.cpu().numpy()
        log["err"][k] = (q_des - q).abs().max(-1).values.cpu().numpy()
        log["mode"][k] = [MODES.index(f.mode) for f in fsms]

    print(f"\nobservation check: {n_checked} policy steps, max |deploy obs - Go2Env obs| = {obs_err:.2e}")
    print(f"simulated {n_steps * dt:.1f} s in {time.monotonic() - t_wall:.1f} s of wall time")
    policy_name = f"{args.exp_name} (exported)" if export_dir else "zero-action policy"
    summarize(log, dt, args.scenario, fsms[0])
    out = args.out or os.path.join(log_dir or ".", f"fsm_{args.scenario}.png")
    title = (f"{args.scenario} | {policy_name} | {N} robot{'s' if N > 1 else ''}"
             f"{' with training randomization' if args.dr else ''} | hand-over ramp {args.handover_s:.1f} s"
             + (f" | hung at {args.suspended} m" if args.suspended is not None else ""))
    plot(log, dt, out, title)
    if recorder is not None:
        recorder.events.extend([float(t), msg] for t, msg in fsms[0].events)
        print(f"log saved to {recorder.save(extra_meta={'obs_check_max_err': obs_err})}")


if __name__ == "__main__":
    main()
