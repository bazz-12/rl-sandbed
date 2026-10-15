import argparse
import os
import pickle
from importlib import metadata

import torch

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError, ValueError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from go2_env import Go2Env, nominal_world


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp-name", type=str, default="go2-walking", help="Experiment name; also the log directory under logs/"
    )
    parser.add_argument("--ckpt", type=int, default=100, help="Checkpoint iteration to load")
    # [P1.1] pin the command, e.g. --cmd 0 0 0 to check standing still (default: random commands as in training)
    parser.add_argument("--cmd", type=float, nargs=3, metavar=("VX", "VY", "WZ"), default=None)
    # [P1.2] by default the viewer runs the nominal world; --dr keeps the training randomization (to watch pushes)
    parser.add_argument("--dr", action="store_true", help="keep domain randomization, noise and pushes")
    parser.add_argument("--latency-ms", type=float, default=20.0, help="action latency in the nominal world")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    log_dir = f"logs/{args.exp_name}"
    with open(f"logs/{args.exp_name}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}
    if args.cmd is not None:  # [P1.1] a [v, v] range pins the command; the training deadzone still applies
        for key, v in zip(("lin_vel_x_range", "lin_vel_y_range", "ang_vel_range"), args.cmd):
            command_cfg[key] = [v, v]
        command_cfg["rel_standing_envs"] = 0.0
    if not args.dr:
        nominal_world(env_cfg, obs_cfg, latency_s=args.latency_ms / 1000.0)  # [P1.2]

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    obs_dict = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs_dict)
            obs_dict, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking --ckpt 100
"""
