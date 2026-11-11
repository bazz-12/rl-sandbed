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


def pin_command(parser, cmd, command_cfg):
    """[P1.1] A [v, v] range pins a command; the training deadzone still applies.
    [P1.6] VX VY WZ [DH]: DH only for runs trained with the body-height command (dh = 0 when omitted)."""
    n_cmd = command_cfg["num_commands"]
    if len(cmd) not in (3, 4):
        parser.error("--cmd takes VX VY WZ, or VX VY WZ DH for runs trained with --height-cmd")
    if len(cmd) == 4 and n_cmd < 4:
        parser.error("--cmd: this run takes VX VY WZ (no DH: it was trained without --height-cmd)")
    keys = ["lin_vel_x_range", "lin_vel_y_range", "ang_vel_range"] + (["body_height_range"] if n_cmd == 4 else [])
    values = list(cmd) + [0.0] * (len(keys) - len(cmd))
    for key, v in zip(keys, values):
        command_cfg[key] = [v, v]
    command_cfg["rel_standing_envs"] = 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp-name", type=str, default="go2-walking", help="Experiment name; also the log directory under logs/"
    )
    parser.add_argument("--ckpt", type=int, default=100, help="Checkpoint iteration to load")
    # [P1.1] pin the command, e.g. --cmd 0 0 0 to check standing still (default: random commands as in training)
    # [P1.6] 4th value: body-height offset dh [m], for runs trained with --height-cmd
    parser.add_argument("--cmd", type=float, nargs="+", metavar="V", default=None, help="VX VY WZ [DH]")
    # [P1.2] by default the viewer runs the nominal world; --dr keeps the training randomization (to watch pushes)
    parser.add_argument("--dr", action="store_true", help="keep domain randomization, noise and pushes")
    parser.add_argument("--latency-ms", type=float, default=20.0, help="action latency in the nominal world")
    # [P1.6] watch a --recovery run (P1.5) get up: every episode starts from a random fallen pose
    parser.add_argument("--fallen", action="store_true", help="every episode starts from a random fallen pose")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    log_dir = f"logs/{args.exp_name}"
    with open(f"logs/{args.exp_name}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}
    if args.cmd is not None:
        pin_command(parser, args.cmd, command_cfg)
    if not args.dr:
        nominal_world(env_cfg, obs_cfg, latency_s=args.latency_ms / 1000.0)  # [P1.2]
    # [P1.6] fallen starts only with --fallen (before, --dr kept the 15 % fallen starts of a --recovery run)
    env_cfg["recovery"] = {"init_prob": 1.0, "spawn_height": 0.5} if args.fallen else {}

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
            policy.reset(dones)  # [P1.6] clear the memory of a reset env (recurrent policy; no-op for an MLP)


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking --ckpt 100
"""
