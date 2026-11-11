"""
go2_export.py - [P1.5] Export a trained policy for deployment.

    python go2_export.py -e go2-p15 --ckpt 999
writes logs/<exp>/exported/
    policy.pt         TorchScript actor (mean action): obs [B, 45] -> action [B, 12] (46 inputs with the body-height
                      command, + 8 gait-clock inputs with --gait wtw [P2.4]). [P1.6] A recurrent actor exports for
                      batch 1 and keeps its memory between calls; reset() clears it (an MLP export has an empty reset(),
                      so the robot side can always call it).
    deploy_cfg.json   everything the robot side needs to build the observation and apply the action
                      (joint order, default pose, scales, gains, command ranges and deadzones, joint limits)
go2_fsm.py (robot side) only reads these two files, so the robot computer needs neither Genesis nor rsl_rl.
[P3.1] For runs trained with the command curriculum (go2_train.py --cmd-curriculum), command_ranges are the ranges the
curriculum had reached in that checkpoint (infos["curriculum"]), not the configured maximum: the policy never trained
beyond them, and go2_fsm.py scales the sticks to command_ranges. The configured ranges stay in
command_ranges_configured. Runs on terrain (--terrain) also get a training_terrain record (types, levels reached).
[P3.3] A privileged critic (go2_train.py --privileged critic) changes nothing here: only the actor is exported, and it
reads the policy observation; deploy_cfg["source"] records the critic's entries. A teacher run (--privileged teacher)
is refused: its actor needs the privileged group, which the robot cannot measure (distill a student first).
[P3.4] Baseline freeze (TEST_AND_TINKER_P3.md, section 7):
    python go2_export.py -e go2-v1.0 --ckpt 1999 --onnx
    policy.onnx       the same actor for ONNX runtimes (rsl_rl export_policy_to_onnx, opset 18): input "obs" [1, D],
                      output "actions" [1, 12]; a recurrent actor also takes h_in (and c_in for an LSTM) and returns
                      h_out (c_out): the caller carries them from one call to the next (zeros at a reset). One file:
                      the weights torch writes to policy.onnx.data by default are folded back in
    export_check.json every check of the exported files against the training policy, and their sha256
Both exports are checked against the training policy (deterministic mean), one observation at a time as on the robot,
on two input sets: 100 random observations, and the observations the policy itself meets in a rollout in the nominal
world (--check-envs envs x --check-steps steps, commands resampled as in training, resets included). Any difference
above EXPORT_TOL stops the script. deploy_cfg["source"] gets the files' sha256, the check results and the run's git
commit (manifest.json). Needs: pip install onnx onnxscript onnxruntime (onnxscript: torch >= 2.9 exports with it).
"""

import argparse
import json
import os
import pickle
from importlib import metadata

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

import genesis as gs
from go2_env import Go2Env, nominal_world
from go2_provenance import now, sha256_file  # [P3.4]

EXPORT_TOL = 1e-4  # [P3.4] largest |export - training policy| accepted, in every check (float32 round-off: ~1e-6)

def obs_layout(num_commands, num_clock=0):
    layout = ["base_ang_vel*ang_vel_scale (3)", "projected_gravity (3)", f"commands*commands_scale ({num_commands})",
              "(dof_pos-default)*dof_pos_scale (12)", "dof_vel*dof_vel_scale (12)", "last_action (12)"]
    if num_clock:  # [P2.4]
        layout.append(f"gait clock: sin of each foot's phase, then cos, feet in gait_clock.feet order ({num_clock})")
    return layout


def make_deploy_cfg(env, recurrent=False, curriculum=None):
    """Deployment parameters, read from a built Go2Env (so they are exactly what training used).
    [P3.1] curriculum: the checkpoint's infos["curriculum"] (Go2Env.curriculum_state()), or None."""
    q_lo, q_hi = env.robot.get_dofs_limit(env.motors_dof_idx)
    q_lo, q_hi = (q_lo[0], q_hi[0]) if q_lo.ndim == 2 else (q_lo, q_hi)
    cc = env.command_cfg
    command_ranges = {"lin_vel_x": list(cc["lin_vel_x_range"]), "lin_vel_y": list(cc["lin_vel_y_range"]),
                      "ang_vel": list(cc["ang_vel_range"])}
    if env.num_commands >= 4:  # [P1.6] body-height offset dh [m]
        command_ranges["body_height"] = list(cc["body_height_range"])
    extra = {}
    if curriculum and curriculum.get("command_curriculum"):  # [P3.1] the ranges the policy actually trained on
        extra["command_ranges_configured"] = {k: list(v) for k, v in command_ranges.items()}
        for name, r in curriculum.get("command_ranges", {}).items():
            command_ranges[name] = [float(r[0]), float(r[1])]
    if curriculum and curriculum.get("terrain"):  # [P3.1] for the record (the policy is blind: nothing to deploy)
        t = curriculum["terrain"]
        top = sum(c[-1] for c in t["level_counts"].values()) / max(1, sum(map(sum, t["level_counts"].values())))
        extra["training_terrain"] = {
            "types": t["types"], "num_levels": t["num_levels"], "max_slope_deg": t["max_slope_deg"], "key": t["key"],
            "mean_level": {name: round(float(np.dot(c, range(len(c))) / max(1, sum(c))), 2)
                           for name, c in t["level_counts"].items()},
            "top_level_share": round(float(top), 3), "env_steps": curriculum.get("env_steps")}
    if env.gait_clock:  # [P2.4] the robot side recomputes the clock (go2_fsm.gait_clock), so it needs its parameters
        extra["gait_clock"] = {
            "feet": [name.split("_")[0] for name in env.foot_links],  # clock order
            "offsets": env.gait_offsets.cpu().tolist(),  # same order
            "frequency": env.gait_frequency,  # [Hz]
            "duty": env.gait_duty,
        }
    return {
        "joint_names": list(env.env_cfg["joint_names"]),  # policy order = action order = observation order
        "default_joint_angles": env.default_dof_pos.cpu().tolist(),
        "kp": float(env.env_cfg["kp"]),
        "kd": float(env.env_cfg["kd"]),
        "action_scale": float(env.env_cfg["action_scale"]),
        "clip_actions": float(env.env_cfg["clip_actions"]),
        "obs_scales": {k: float(v) for k, v in env.obs_scales.items()},
        "commands_scale": env.commands_scale.cpu().tolist(),
        "num_commands": int(env.num_commands),  # [P1.6] 4 = (vx, vy, wz, dh)
        "command_ranges": command_ranges,
        "lin_deadzone": float(cc.get("lin_deadzone", 0.0)),
        "ang_deadzone": float(cc.get("ang_deadzone", 0.0)),
        "base_height_target": float(env.reward_cfg["base_height_target"]),  # [P1.6] dh is an offset from it
        "recurrent": bool(recurrent),  # [P1.6] the exported network keeps a memory (call reset() at each start)
        "policy_dt": float(env.dt),
        "torque_limits": env.torque_limits.cpu().tolist(),
        "dof_pos_limits": torch.stack((q_lo, q_hi), dim=-1).cpu().tolist(),
        "obs_layout": obs_layout(env.num_commands, env.num_clock_obs),  # [P2.4]
        # [P2] for the record: the passive joint model the policy was trained with ([S2R]; not used on the robot)
        "joint_friction": float(env.env_cfg.get("joint_friction", 0.0)),
        "joint_damping": float(env.env_cfg.get("joint_damping", 0.0)),
        **extra,
    }


class OnnxPolicy:
    """[P3.4] policy.onnx through onnxruntime, called like the TorchScript export: one observation [1, D] per call; a
    recurrent model's state (h, and c for an LSTM) is carried from one call to the next, and reset() zeroes it."""

    def __init__(self, path):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise SystemExit("--onnx: pip install onnxruntime (to check policy.onnx)") from e
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input_shapes = {i.name: list(i.shape) for i in self.sess.get_inputs()}
        self.outputs = [o.name for o in self.sess.get_outputs()]
        self.reset()

    def reset(self):
        self.state = {name: np.zeros(shape, dtype=np.float32) for name, shape in self.input_shapes.items()
                      if name != "obs"}

    def __call__(self, obs):
        out = dict(zip(self.outputs, self.sess.run(self.outputs, {"obs": np.asarray(obs, np.float32), **self.state})))
        for name in self.state:  # h_in <- h_out, c_in <- c_out
            self.state[name] = out[name.replace("_in", "_out")]
        return out["actions"]


def single_file_onnx(path):
    """[P3.4] torch 2.15 writes the weights next to the graph (torch.onnx.export's default external_data=True, which
    rsl_rl keeps): policy.onnx would only be the graph, and policy.onnx.data the weights. Fold them back into
    policy.onnx: one self-contained file for the robot side and the release, whose sha256 covers the weights."""
    import onnx

    graph_only = onnx.load(path, load_external_data=False)
    files = {e.value for t in graph_only.graph.initializer for e in t.external_data if e.key == "location"}
    if not files:
        return
    onnx.save_model(onnx.load(path), path, save_as_external_data=False)  # onnx.load reads the external files
    for name in files:
        os.remove(os.path.join(os.path.dirname(path), name))


def rollout_observations(env, policy, steps):
    """[P3.4] The observations the policy meets: `steps` policy steps of the training policy itself in the nominal env
    (commands resampled as in training). Returns obs [T, N, D], its actions [T, N, A] and dones [T, N] (dones[t]: the
    env was reset after step t, so obs[t + 1] starts a new episode)."""
    obs = env.reset()
    policy.reset()
    seq_obs, seq_act, seq_done = [], [], []
    with torch.no_grad():
        for _ in range(steps):
            actions = policy(obs)
            seq_obs.append(obs["policy"].clone())
            seq_act.append(actions.clone())
            obs, _, dones, _ = env.step(actions)
            policy.reset(dones)
            seq_done.append(dones.clone())
    return torch.stack(seq_obs), torch.stack(seq_act), torch.stack(seq_done)


def sequence_error(step, reset, obs, ref, dones=None):
    """[P3.4] max |step(obs[t, n]) - ref[t, n]|, feeding each env's sequence one observation [1, D] at a time (as on the
    robot), reset() at its start and after each of its resets."""
    err = 0.0
    for n in range(obs.shape[1]):
        reset()
        for t in range(obs.shape[0]):
            if t > 0 and dones is not None and bool(dones[t - 1, n]):
                reset()
            out = np.asarray(step(obs[t, n][None]))
            err = max(err, float(np.abs(out - ref[t, n][None].numpy()).max()))
    return err


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    p.add_argument("--ckpt", type=int, required=True)
    p.add_argument("--log_dir", type=str, default=None, help="default: logs/<exp_name>")
    # [P3.4] ONNX and the checks of the exported files
    p.add_argument("--onnx", action="store_true", help="also write policy.onnx (and check it with onnxruntime)")
    p.add_argument("--check-steps", type=int, default=200, help="rollout length of the real-observation check")
    p.add_argument("--check-envs", type=int, default=8, help="envs of the real-observation check")
    args = p.parse_args()

    log_dir = args.log_dir or os.path.join("logs", args.exp_name)
    out_dir = os.path.join(log_dir, "exported")
    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    actor_groups = train_cfg.get("obs_groups", {}).get("actor", ["policy"])
    if list(actor_groups) != ["policy"]:  # [P3.3] a teacher (or any actor reading more than the robot measures)
        raise SystemExit(f"{args.exp_name}: the actor reads the observation groups {actor_groups}, not only 'policy' "
                         f"(a --privileged teacher run). The robot cannot measure the privileged group: distill a "
                         f"student that reads 'policy' only, and export that one.")
    gs.init(backend=gs.cpu, logging_level="warning")
    reward_cfg["reward_scales"] = {}
    nominal_world(env_cfg, obs_cfg)
    # [P3.4] several envs: the real-observation check rolls them out (the deploy config does not depend on their number)
    env = Go2Env(num_envs=max(1, args.check_envs), env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg,
                 command_cfg=command_cfg)

    runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=gs.device)
    infos = runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt")) or {}
    runner.export_policy_to_jit(out_dir, "policy.pt")
    onnx_path = os.path.join(out_dir, "policy.onnx")
    if args.onnx:  # [P3.4]
        try:
            runner.export_policy_to_onnx(out_dir, "policy.onnx")
            single_file_onnx(onnx_path)
        except ImportError as e:
            raise SystemExit(f"--onnx: {e}\npip install onnx onnxscript onnxruntime") from e
    else:  # the policy.onnx of an earlier export would not match policy.pt any more
        stale = [f for f in (onnx_path, onnx_path + ".data") if os.path.exists(f)]
        for f in stale:
            os.remove(f)
        if stale:
            print(f"removed {', '.join(os.path.basename(f) for f in stale)} of an earlier export (no --onnx this time)")

    policy = runner.get_inference_policy(device="cpu")
    curriculum = infos.get("curriculum")  # [P3.1] None for runs from before P3.1
    deploy_cfg = make_deploy_cfg(env, recurrent=policy.is_recurrent, curriculum=curriculum)

    # Check: the exported network gives the same actions as the training policy (deterministic mean).
    # [P1.6] Over 100 consecutive steps from a reset, one observation at a time, as on the robot: for a recurrent
    # policy this also checks that the export carries its memory from one call to the next.
    # [P3.4] Same for policy.onnx, and both again on the observations of a rollout of the policy (a real state
    # distribution, with the resets of the episodes).
    exports = {"policy.pt": torch.jit.load(os.path.join(out_dir, "policy.pt"), map_location="cpu")}
    if args.onnx:
        exports["policy.onnx"] = OnnxPolicy(onnx_path)
    rollout_obs, rollout_act, rollout_done = rollout_observations(env, policy, args.check_steps)
    random_obs = torch.randn(100, 1, env.obs_buf.shape[-1])
    with torch.no_grad():
        policy.reset()
        random_ref = torch.stack([policy(TensorDict({"policy": o}, batch_size=[1])) for o in random_obs])
    checks = {}
    with torch.no_grad():
        for name, net in exports.items():
            step = (lambda o, net=net: net(o).numpy()) if name == "policy.pt" else (lambda o, net=net: net(o.numpy()))
            checks[name] = {
                "random_100": sequence_error(step, net.reset, random_obs, random_ref),
                f"rollout_{rollout_obs.shape[1]}x{rollout_obs.shape[0]}": sequence_error(
                    step, net.reset, rollout_obs, rollout_act, rollout_done),
            }
    worst = max(err for c in checks.values() for err in c.values())
    hashes = {name: sha256_file(os.path.join(out_dir, name)) for name in exports}
    manifest = {}
    if os.path.isfile(os.path.join(log_dir, "manifest.json")):
        with open(os.path.join(log_dir, "manifest.json")) as f:
            manifest = json.load(f)

    deploy_cfg["source"] = {
        "exp_name": args.exp_name, "ckpt": args.ckpt,
        "genesis": gs.__version__, "rsl_rl": metadata.version("rsl-rl-lib"), "torch": torch.__version__,
        # [P3.4] provenance and the checks of the exported files
        "git_commit": manifest.get("git_hash"), "git_dirty": manifest.get("git_dirty"),
        "exported": now(), "sha256": hashes,
        "max_abs_error_vs_training_policy": {name: max(c.values()) for name, c in checks.items()},
    }
    if env.privileged_layout:  # [P3.3] for the record: what the critic read during training (not deployed)
        deploy_cfg["source"]["critic_privileged"] = [name for name, _ in env.privileged_layout]
    with open(os.path.join(out_dir, "deploy_cfg.json"), "w") as f:
        json.dump(deploy_cfg, f, indent=2)
    report = {"exp_name": args.exp_name, "ckpt": args.ckpt, "exported": deploy_cfg["source"]["exported"],
              "tolerance": EXPORT_TOL, "passed": bool(np.isfinite(worst) and worst <= EXPORT_TOL),
              "max_abs_error": checks, "sha256": {**hashes, "deploy_cfg.json": sha256_file(
                  os.path.join(out_dir, "deploy_cfg.json"))},
              "recurrent": bool(policy.is_recurrent), "obs_dim": int(env.obs_buf.shape[-1]),
              "onnx_io": ({"inputs": exports["policy.onnx"].input_shapes, "outputs": exports["policy.onnx"].outputs}
                          if args.onnx else None),
              "git_commit": manifest.get("git_hash"), "git_dirty": manifest.get("git_dirty")}
    with open(os.path.join(out_dir, "export_check.json"), "w") as f:
        json.dump(report, f, indent=2)

    kind = "recurrent" if policy.is_recurrent else "MLP"
    files = ", ".join(exports) + ", deploy_cfg.json, export_check.json"
    print(f"exported to {out_dir}: {files} ({kind})")
    for name, c in checks.items():  # [P3.4]
        print(f"  {name:11s} max |export - policy|: " + ", ".join(f"{k} {v:.2e}" for k, v in c.items()))
    if "command_ranges_configured" in deploy_cfg:  # [P3.1]
        print(f"command ranges reached by the command curriculum (deployed): {deploy_cfg['command_ranges']}   "
              f"configured: {deploy_cfg['command_ranges_configured']}")
    if "training_terrain" in deploy_cfg:
        t = deploy_cfg["training_terrain"]
        print(f"trained on terrain {t['types']} ({t['num_levels']} levels, up to {t['max_slope_deg']:g} deg): "
              f"mean level {t['mean_level']}, top level share {t['top_level_share']:.0%}")
    if manifest.get("git_dirty"):  # [P3.4]
        print("WARNING: the run was trained from uncommitted code (manifest.json git_dirty): not a clean baseline")
    if not report["passed"]:
        raise SystemExit(f"exported network does not match the training policy (worst {worst:.2e} > {EXPORT_TOL:g})")


if __name__ == "__main__":
    main()
