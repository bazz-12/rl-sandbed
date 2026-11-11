"""
go2_export.py - [P1.5] Export a trained policy for deployment.

    python go2_export.py -e go2-p15 --ckpt 999
writes logs/<exp>/exported/
    policy.pt         TorchScript actor (mean action): obs [B, 45] -> action [B, 12] (46 inputs with the body-height
                      command). [P1.6] A recurrent actor exports for batch 1 and keeps its memory between calls;
                      reset() clears it (an MLP export has an empty reset(), so the robot side can always call it).
    deploy_cfg.json   everything the robot side needs to build the observation and apply the action
                      (joint order, default pose, scales, gains, command ranges and deadzones, joint limits)
go2_fsm.py (robot side) only reads these two files, so the robot computer needs neither Genesis nor rsl_rl.
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

def obs_layout(num_commands):
    return ["base_ang_vel*ang_vel_scale (3)", "projected_gravity (3)", f"commands*commands_scale ({num_commands})",
            "(dof_pos-default)*dof_pos_scale (12)", "dof_vel*dof_vel_scale (12)", "last_action (12)"]


def make_deploy_cfg(env, recurrent=False):
    """Deployment parameters, read from a built Go2Env (so they are exactly what training used)."""
    q_lo, q_hi = env.robot.get_dofs_limit(env.motors_dof_idx)
    q_lo, q_hi = (q_lo[0], q_hi[0]) if q_lo.ndim == 2 else (q_lo, q_hi)
    cc = env.command_cfg
    command_ranges = {"lin_vel_x": list(cc["lin_vel_x_range"]), "lin_vel_y": list(cc["lin_vel_y_range"]),
                      "ang_vel": list(cc["ang_vel_range"])}
    if env.num_commands >= 4:  # [P1.6] body-height offset dh [m]
        command_ranges["body_height"] = list(cc["body_height_range"])
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
        "obs_layout": obs_layout(env.num_commands),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    p.add_argument("--ckpt", type=int, required=True)
    p.add_argument("--log_dir", type=str, default=None, help="default: logs/<exp_name>")
    args = p.parse_args()

    log_dir = args.log_dir or os.path.join("logs", args.exp_name)
    out_dir = os.path.join(log_dir, "exported")
    gs.init(backend=gs.cpu, logging_level="warning")
    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}
    nominal_world(env_cfg, obs_cfg)
    env = Go2Env(num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg)

    runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
    runner.export_policy_to_jit(out_dir, "policy.pt")

    policy = runner.get_inference_policy(device="cpu")
    deploy_cfg = make_deploy_cfg(env, recurrent=policy.is_recurrent)
    deploy_cfg["source"] = {
        "exp_name": args.exp_name, "ckpt": args.ckpt,
        "genesis": gs.__version__, "rsl_rl": metadata.version("rsl-rl-lib"), "torch": torch.__version__,
    }
    with open(os.path.join(out_dir, "deploy_cfg.json"), "w") as f:
        json.dump(deploy_cfg, f, indent=2)

    # Check: the exported network gives the same actions as the training policy (deterministic mean).
    # [P1.6] Over 100 consecutive steps from a reset, one observation at a time, as on the robot: for a recurrent
    # policy this also checks that the export carries its memory from one call to the next.
    jit = torch.jit.load(os.path.join(out_dir, "policy.pt"), map_location="cpu")
    obs = torch.randn(100, 1, env.obs_buf.shape[-1])
    err = 0.0
    with torch.no_grad():
        policy.reset()
        jit.reset()
        for o in obs:
            ref = policy(TensorDict({"policy": o}, batch_size=[1]))
            err = max(err, (jit(o) - ref).abs().max().item())
    kind = "recurrent" if policy.is_recurrent else "MLP"
    print(f"exported to {out_dir}: policy.pt ({kind}), deploy_cfg.json   "
          f"(max |jit - policy| over 100 steps: {err:.2e})")
    if not np.isfinite(err) or err > 1e-4:
        raise SystemExit("exported network does not match the training policy")


if __name__ == "__main__":
    main()
