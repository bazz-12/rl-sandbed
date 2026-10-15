import argparse
import os
import pickle
import shutil
from importlib import metadata





try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from go2_env import Go2Env, FOOT_LINKS, GO2_TORQUE_LIMITS, GO2_DOF_VEL_LIMITS, obs_noise_scales
from go2_commands import LIN_DEADZONE, ANG_DEADZONE  # [P1.1] same deadzone as at deployment

import subprocess, json, sys, torch
from importlib.metadata import version

def run_manifest(log_dir, cfgs):
    meta = {
        "git_hash": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"]).strip()),
        "python": sys.version.split()[0],
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "genesis": gs.__version__, "rsl_rl": version("rsl-rl-lib"),
        "cfgs": cfgs,
    }
    with open(f"{log_dir}/manifest.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)



def get_train_cfg(exp_name):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "num_steps_per_env": 24,
        "save_interval": 100,
        "run_name": exp_name,
        "logger": "tensorboard",
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        # PD
        "kp": 20.0,
        "kd": 0.5,
        # termination
        # [P1.2] 10 -> 30 deg: a 1 m/s push tilts the robot past 10 deg, and terminating there would never let the
        # policy learn to recover (the orientation reward below keeps the base level in normal walking)
        "termination_if_roll_greater_than": 30,  # degree
        "termination_if_pitch_greater_than": 30,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "clip_actions": 100.0,
        # [P1.1] foot links kept separate from the calves (contact forces for the air-time reward)
        "foot_links": list(FOOT_LINKS),
        # [P1.2] actuation fidelity: 200 Hz physics + PD, 50 Hz policy, real rotor inertia, real torque limits
        "sim_dt": 0.005,
        "decimation": 4,
        "armature": 0.01,  # MuJoCo Menagerie go2.xml (Genesis URDF default: 0.1)
        "torque_limits": dict(GO2_TORQUE_LIMITS),  # clipping, from Unitree's go2_description URDF
        "latency_s": 0.02,  # used only when latency is not randomized (replaces simulate_action_latency)
        # [P1.2] domain randomization (legged_gym / Isaac Lab defaults, widened where the plan says so)
        "domain_rand": {
            "friction_range": [0.5, 1.25],  # contact friction coefficient (ground and robot)
            "added_mass_range": [-1.0, 2.0],  # [kg] on the base (Go2 ~15 kg)
            "com_shift_range": [-0.03, 0.03],  # [m] base CoM offset, each axis
            "kp_range": [0.85, 1.15],  # x Kp, per env and episode
            "kd_range": [0.85, 1.15],  # x Kd, per env and episode
            "motor_strength_range": [0.9, 1.1],  # x PD torque, per env, joint and episode
            "latency_range_s": [0.0, 0.02],  # action latency, per env and episode (5 ms resolution)
            "push_interval_s": 15.0,  # legged_gym
            "max_push_vel_xy": 1.0,  # [m/s] legged_gym
        },
        # [P1.3] contacts, by link-name substring (legged_gym / unitree_rl_gym go2_config)
        "penalize_contacts_on": ["thigh", "calf", "base"],
        "terminate_after_contacts_on": ["base"],
        "dof_vel_limits": dict(GO2_DOF_VEL_LIMITS),  # [rad/s] Unitree go2_description URDF
        # [P1.5] learned recovery, off: go2_train.py --recovery sets init_prob 0.15 (MetalHead recovery_init_prob)
        "recovery": {"init_prob": 0.0, "spawn_height": 0.5},
    }
    obs_cfg = {
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            # [P1.6] body-height command (used with --height-cmd only): dh x 5 -> about +-0.4, within the span of the
            # other command entries (+-0.25 to +-2). walk-these-ways scales its body_height_cmd by 2.
            "body_height": 5.0,
        },
        # [P1.2] observation noise, legged_gym noise_scales (physical units: rad/s, -, rad, rad/s)
        "add_noise": True,
        "noise_scales": {"ang_vel": 0.2, "gravity": 0.05, "dof_pos": 0.01, "dof_vel": 1.5},
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        # [P1.1] feet air time (legged_gym): contact threshold [N] and target air time [s]
        "contact_force_threshold": 1.0,
        "feet_air_time_target": 0.5,
        # [P1.1] at zero command and base speed < threshold, similar_to_default is multiplied by stand_still_scale
        # (Isaac Lab Spot: stand_still_scale 5.0, velocity_threshold 0.5 m/s)
        "stand_still_scale": 5.0,
        "stand_still_vel_threshold": 0.5,
        # [P1.3] limits and contacts (legged_gym; soft limits 0.9 as unitree_rl_gym go2_config)
        "soft_dof_pos_limit": 0.9,  # share of the URDF position range
        "soft_dof_vel_limit": 0.9,  # share of the motor velocity limit
        "collision_force_threshold": 0.1,  # [N]
        "termination_contact_force": 1.0,  # [N]
        "only_positive_rewards": True,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.5,  # [P1.1] 0.2 -> 0.5 (legged_gym): yaw commands are now non-zero
            "lin_vel_z": -2.0,  # [P1.3] -1 -> -2 (legged_gym)
            "base_height": -50.0,
            "action_rate": -0.01,  # [P1.3] -0.005 -> -0.01 (legged_gym)
            "similar_to_default": -0.1,
            "feet_air_time": 1.0,  # [P1.1] legged_gym default, gated off at zero command
            "orientation": -2.5,  # [P1.2] Isaac Lab Go2 flat (flat_orientation_l2); replaces the 10 deg termination
            # [P1.3] safety and limits, legged_gym defaults (dof_pos_limits: unitree_rl_gym go2_config)
            "ang_vel_xy": -0.05,
            "torques": -1e-5,
            "dof_acc": -2.5e-7,
            "collision": -1.0,
            "dof_pos_limits": -10.0,
            "dof_vel_limits": -1.0,  # no legged_gym default; at most -1 per joint and policy step (x dt)
            # [P1.4] second difference of the joint targets: walk-these-ways action_smoothness_2, same weight
            "action_smoothness": -0.1,
        },
    }
    command_cfg = {
        "num_commands": 3,  # [P1.6] 4 with --height-cmd: (vx, vy, wz, dh)
        # [P1.6] body-height offset dh from base_height_target [m], sampled with --height-cmd only (walk-these-ways
        # body_height_cmd). Reach with the feet at their default positions (IK): front calf at its soft limit at
        # +6.9 cm, at its URDF limit at +7.8 cm; -12 cm still leaves the calf 0.5 rad inside its range.
        "body_height_range": [-0.08, 0.04],
        # [P1.1] a real command space (was a single command, vx = 0.5); covers the go2_metrics.py grid
        "lin_vel_x_range": [-1.0, 1.0],
        "lin_vel_y_range": [-0.5, 0.5],
        "ang_vel_range": [-1.0, 1.0],
        # [P1.1] zero small commands (legged_gym) + exact-zero envs (Isaac Lab rel_standing_envs; Spot uses 0.1)
        "lin_deadzone": LIN_DEADZONE,
        "ang_deadzone": ANG_DEADZONE,
        "rel_standing_envs": 0.1,
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp-name", type=str, default="go2-walking", help="Experiment name; also the log directory under logs/"
    )
    parser.add_argument("-b", "--num-envs", type=int, default=4096, help="Number of parallel environments")
    # [P1.1] 101 iterations were enough for one fixed command; a 3-D command space with standing needs ~10x more
    parser.add_argument("--max-iterations", type=int, default=1000,
                        help="learning iterations (rsl_rl counts from 0: the last checkpoint is model_<N-1>.pt)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--cpu", action="store_true", help="CPU backend, for quick smoke tests only")
    # [P1.4] smoothness regularizers on the policy network (go2_smooth_ppo.py); all off by default. Use them only if
    # the action_smoothness reward is not enough (check the "Smoothness" table of go2_metrics.py first).
    parser.add_argument("--lcp", type=float, default=0.0, help="LCP gradient-penalty coefficient (paper: 0.002)")
    parser.add_argument("--caps-t", type=float, default=0.0, help="CAPS temporal coefficient (paper, Ant: 0.1)")
    parser.add_argument("--caps-s", type=float, default=0.0, help="CAPS spatial coefficient (paper, Ant: 0.5)")
    # [P1.5] learned recovery (optional, W12): 15 % of the episodes start fallen, with recovery rewards
    parser.add_argument("--recovery", action="store_true", help="train fall recovery as well")
    # [P1.6] body-height command and recurrent policy, both off by default (then a run is a P1.5 run)
    parser.add_argument("--height-cmd", action="store_true", help="4th command: body-height offset dh [m]")
    parser.add_argument("--rnn", choices=("lstm", "gru"), default=None, help="recurrent actor and critic (RNNModel)")
    parser.add_argument("--rnn-hidden", type=int, default=256, help="RNN hidden size (rsl_rl's default: 256)")
    parser.add_argument("--steps-per-env", type=int, default=None,
                        help="rollout length per env and iteration (default: 24 for an MLP, 48 with --rnn)")
    args = parser.parse_args()
    if args.rnn and (args.lcp > 0 or args.caps_t > 0 or args.caps_s > 0):
        # go2_smooth_ppo.py evaluates the policy on flat mini-batches of single observations; a recurrent version
        # would need rsl_rl's padded-sequence mini-batches with their hidden states
        parser.error("--lcp / --caps-* are implemented for MLP policies only (go2_smooth_ppo.py): drop --rnn or them")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name)
    if args.recovery:  # [P1.5] weights: my starting point, to be tuned (MetalHead uses AMP + task rewards instead)
        env_cfg["recovery"]["init_prob"] = 0.15
        reward_cfg["reward_scales"].update({"rec_upright": 1.0, "rec_height": 1.0, "rec_pose": 0.5})
    if args.height_cmd:  # [P1.6] dh sampled in command_cfg["body_height_range"]
        command_cfg["num_commands"] = 4
    if args.rnn:  # [P1.6] rsl_rl RNNModel: one RNN layer on the observation, then the same MLP; one memory each
        rnn_num_layers = 1
        for model in ("actor", "critic"):
            train_cfg[model].update(
                {"class_name": "RNNModel", "rnn_type": args.rnn, "rnn_hidden_dim": args.rnn_hidden,
                 "rnn_num_layers": rnn_num_layers}
            )
    # [P1.6] rsl_rl trains the RNN by backpropagation through time cut at the rollout: a mini-batch trajectory starts
    # from the memory saved during the rollout and lasts at most num_steps_per_env steps (it also ends at a reset).
    # So the rollout length bounds how far back a gradient reaches: 48 steps = 0.96 s.
    if args.steps_per_env is not None or args.rnn:
        train_cfg["num_steps_per_env"] = args.steps_per_env or 48
    if args.lcp > 0 or args.caps_t > 0 or args.caps_s > 0:  # [P1.4]
        train_cfg["algorithm"].update(
            {
                "class_name": "go2_smooth_ppo:SmoothPPO",
                "grad_penalty_coef": args.lcp,
                "caps_temporal_coef": args.caps_t,
                "caps_spatial_coef": args.caps_s,
                # CAPS spatial noise: the observation-noise scales of P1.2 (0 on commands and last actions)
                "caps_sigma": obs_noise_scales(obs_cfg, env_cfg["num_actions"], command_cfg["num_commands"]),
            }
        )

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    with open(f"{log_dir}/cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)
    # run_manifest() was defined but never called; best effort, so a missing git repo never blocks training
    try:
        run_manifest(log_dir, [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg])
    except Exception as e:  # noqa: BLE001
        print(f"manifest not written: {e}")

    # Training throughput comes from the 4096 parallel environments, which need a GPU.
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="warning", seed=args.seed,
            performance_mode=True)

    env = Go2Env(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training
python examples/locomotion/go2_train.py
"""
