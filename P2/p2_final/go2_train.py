import argparse
import ast
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

from go2_env import (Go2Env, FOOT_LINKS, GO2_TORQUE_LIMITS, GO2_DOF_VEL_LIMITS, GO2_JOINT_FRICTION, GO2_JOINT_DAMPING,
                     GO2_MJCF_LOWER_LEG, obs_noise_scales)
from go2_env import WTW_TROT_OFFSETS, num_clock_obs  # [P2.4]
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
        # [S2R] joint losses of unitree_mujoco's go2.xml. Without them go2-p16 froze at +0.25 m/s on the Genesis bench
        # with --joint-friction 0.2 --joint-damping 0.1 (and in unitree_mujoco): it had only learned lossless joints.
        "joint_friction": GO2_JOINT_FRICTION,  # [Nm] 0.2
        "joint_damping": GO2_JOINT_DAMPING,  # [Nm s/rad] 0.1
        # [S2R] unitree_mujoco's contact and lower legs. Together (not one by one) they reproduce in Genesis what
        # go2-p17h does in unitree_mujoco (bench, p17h: slow +0.11 / -0.10 m/s vs MuJoCo +0.12 / -0.10, yaw 0.21 vs
        # 0.29 rad/s, step frequency 0.83 Hz in both, against 0.18 / -0.19 m/s, 0.54 rad/s, 1.22 Hz in plain Genesis).
        # With Genesis' soft default friction (pyramidal cone, impratio 1) stance feet creep, the spin resistance with
        # them, and p17h learned a gait that relies on it. Costs solver time; None turns a term off.
        "foot_torsional_friction": 0.02,  # [m] go2.xml feet: condim="6" friction="0.4 0.02 0.01"
        "foot_rolling_friction": 0.01,  # [m]
        "friction_cone": "elliptic",  # go2.xml <option cone="elliptic" impratio="100"/>: feet hold, no creep
        "impratio": 100.0,
        "lower_leg_inertial": dict(GO2_MJCF_LOWER_LEG),  # go2.xml calf + foot: 0.241 kg, CoM 14.1 cm below the knee
        # [P1.2] domain randomization (legged_gym / Isaac Lab defaults, widened where the plan says so)
        "domain_rand": {
            # [S2R] 0.5 -> 0.4: unitree_mujoco's feet (go2.xml friction="0.4 ...", priority 1) were outside the range
            "friction_range": [0.4, 1.25],  # contact friction coefficient (ground and robot)
            "added_mass_range": [-1.0, 2.0],  # [kg] on the base (Go2 ~15 kg)
            "com_shift_range": [-0.03, 0.03],  # [m] base CoM offset, each axis
            "kp_range": [0.85, 1.15],  # x Kp, per env and episode
            "kd_range": [0.85, 1.15],  # x Kd, per env and episode
            "motor_strength_range": [0.9, 1.1],  # x PD torque, per env, joint and episode
            "latency_range_s": [0.0, 0.02],  # action latency, per env and episode (5 ms resolution)
            "push_interval_s": 15.0,  # legged_gym
            "max_push_vel_xy": 1.0,  # [m/s] legged_gym
            # [S2R] optional, off: joint losses sampled per env and joint around the go2.xml values (the real Go2's
            # are not known). Turns on batched dof info (a little slower). Uncomment both to use.
            # "joint_friction_range": [0.05, 0.4],  # [Nm]
            # "joint_damping_range": [0.02, 0.2],  # [Nm s/rad]
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
        "feet_height_target": 0.075,  # [m] [S2R] now used: swing clearance of the foot_clearance reward (go2-p16: 0.6 cm)
        # [S2R] foot_clearance kernel exp(-sum (z - z*)^2 tanh(2 |v_xy|) / std). Isaac Lab Spot uses std 0.05: two feet
        # dragged at 0.5 m/s then still score 0.84 (1.0 on target), too weak against go2-p16's shuffle. 0.02: 0.65.
        "foot_clearance_std": 0.02,  # [m^2] smaller = steeper; back to 0.05 if swing_h overshoots the target by > 3 cm
        "foot_clearance_tanh_mult": 2.0,  # [s/m] higher = slow feet (yaw, side steps) count fully
        # [P1.1] feet air time (legged_gym): contact threshold [N] and target air time [s]
        "contact_force_threshold": 1.0,
        "feet_air_time_target": 0.3,  # [fix] 0.5 is the ANYmal value, out of the Go2's reach at speed (P2 guide §5.1)
        "feet_air_time_max": None,  # [P2.1] optional cap [s] on the air time paid at touchdown (None = uncapped)
        # [P2.4] --gait spot: Spot GaitReward (std [s^2], max_err [s], velocity_threshold [m/s]), air_time_variance clip
        "gait_std": 0.1,
        "gait_max_err": 0.2,
        "gait_velocity_threshold": 0.5,
        "air_time_variance_clip": 0.5,
        # [P2.4] --gait wtw: walk-these-ways contact-schedule kernels (gait_force_sigma [N^2], gait_vel_sigma [m^2/s^2])
        "gait_force_sigma": 100.0,
        "gait_vel_sigma": 10.0,
        # [P1.1] at zero command and base speed < threshold, similar_to_default is multiplied by stand_still_scale
        # (Isaac Lab Spot: stand_still_scale 5.0, velocity_threshold 0.5 m/s)
        "stand_still_scale": 5.0,
        "stand_still_vel_threshold": 0.5,
        # [P1.3] limits and contacts (legged_gym; soft limits 0.9 as unitree_rl_gym go2_config)
        "soft_dof_pos_limit": 0.9,  # share of the URDF position range
        "soft_dof_vel_limit": 0.9,  # share of the motor velocity limit
        "collision_force_threshold": 0.1,  # [N]
        "termination_contact_force": 1.0,  # [N]
        "penalty_scale_init": 0.3,  # [fix] cost curriculum: penalties x 0.3 at the start, x 1 after 400 iterations
        "only_positive_rewards": True,
        "reward_scales": {
            "tracking_lin_vel": 1.5,   # [fix] 1.0 -> 1.5 (Isaac Lab Go2 flat, unitree_rl_lab Go2)
            "tracking_ang_vel": 0.75,  # [fix] 0.5 -> 0.75 (same sources)
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
            # [S2R] swing feet at feet_height_target, no dragged feet (Isaac Lab Spot foot_clearance_reward: 0.5 next to
            # 5 + 5 of tracking; stronger here because go2-p16 barely lifted its feet). Kernel in [0, 1]: dragging two
            # feet at ground level costs about 0.35 per second of reward, against 2.25 for perfect tracking.
            "foot_clearance": 1.0,
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
        # [S2R] single-axis commands (go2_env._single_axis_commands): what the gate profiles and a gamepad send, and
        # what uniform sampling of the 3-D box almost never gives (go2-p16: frozen at +-0.25 m/s on the bench).
        "small_cmd_prob": 0.25,  # share of resampled commands: one axis, magnitude deadzone .. small_*_max
        "small_lin_max": 0.4,  # [m/s]
        "small_ang_max": 0.8,  # [rad/s]
        "single_axis_prob": 0.15,  # share of resampled commands: one axis, magnitude over its full range
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


# [P2] Values the P2 switches use. P1 ([fix]) raised tracking from 1.0 + 0.5 to 1.5 + 0.75, so the P2 weights below are
# 1.5x those of the first P2 version: same strength relative to the tracking reward (TEST_AND_TINKER_P2.md 3.5).
MIRROR_LOSS_COEF = 1.0  # --symmetry loss/both (no published value; Mittal et al. 2024: too high hurts training)
ENERGY_WEIGHT = 1.5e-3  # --energy: x |tau qdot| [W]; at 75 W, 5 % of the 2.25/s tracking reward (unitree_rl_lab 2e-5)
SPOT_GAIT = {"gait": 1.5, "air_time_variance": -1.0}  # Spot: 10 next to tracking 5 + 5; variance: unitree_rl_lab Go2
WTW_GAIT = {"contact_schedule_force": -3.0, "contact_schedule_vel": -3.0}  # WTW: +4 each next to tracking 1 + 0.5
WTW_CLOCK = {"frequency": 2.0, "duty": 0.5, "kappa": 0.07, "offsets": dict(WTW_TROT_OFFSETS)}  # 2 Hz -> 0.25 s swings


def apply_overrides(cfgs, items):
    """[P2] --set group.key.subkey=value on the config dicts (value parsed as a Python literal, else kept as text).
    Groups: env, obs, reward, command, train. value None deletes the key (e.g. a reward term)."""
    for item in items:
        path, sep, text = item.partition("=")
        keys = path.split(".")
        if not sep or len(keys) < 2 or keys[0] not in cfgs:
            raise SystemExit(f"--set {item}: expected GROUP.KEY=VALUE with GROUP in {', '.join(cfgs)}")
        node = cfgs[keys[0]]
        for key in keys[1:-1]:
            if not isinstance(node.get(key), dict):
                if node.get(key) is not None:
                    raise SystemExit(f"--set {item}: {key} is not a dictionary")
                node[key] = {}  # e.g. env.gait_clock.frequency=2.5 without --gait wtw creates the entry
            node = node[key]
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            value = text
        if value is None and keys[0] == "reward" and keys[-2] == "reward_scales":
            node.pop(keys[-1], None)  # remove the term (also fine when it was not enabled)
        else:
            node[keys[-1]] = value
        print(f"--set {path} = {value!r}")
    sym = cfgs["train"]["algorithm"].get("symmetry_cfg")
    if sym is not None and "data_augmentation_func" not in sym:
        raise SystemExit("--set train.algorithm.symmetry_cfg...: give --symmetry (aug / loss / both / log) as well")


def p2_summary(env_cfg, reward_cfg, command_cfg, train_cfg):
    """[P2] One line per gait-quality part, printed at start (the full configuration is in cfgs.pkl)."""
    scales = reward_cfg["reward_scales"]
    sym = train_cfg["algorithm"].get("symmetry_cfg")
    sym_mode = "off" if not sym else "+".join(
        name for name, on in (("aug", sym["use_data_augmentation"]), ("loss", sym["use_mirror_loss"])) if on
    ) or "log only"
    gait = [name for name in ("gait", "air_time_variance", "contact_schedule_force", "contact_schedule_vel")
            if name in scales]
    clock = env_cfg.get("gait_clock")
    dr = env_cfg.get("domain_rand", {})
    joint_dr = {k: dr[k] for k in ("joint_friction_range", "joint_damping_range") if k in dr}
    lines = [
        f"P2.0 [S2R] joints {env_cfg.get('joint_friction', 0.0)} Nm / {env_cfg.get('joint_damping', 0.0)} Nm s/rad"
        + (f" {joint_dr}" if joint_dr else "") + f", feet torsion {env_cfg.get('foot_torsional_friction')} rolling "
        f"{env_cfg.get('foot_rolling_friction')}, cone {env_cfg.get('friction_cone')} impratio "
        f"{env_cfg.get('impratio')}, MJCF lower legs {bool(env_cfg.get('lower_leg_inertial'))}, contact friction "
        f"{dr.get('friction_range')}",
        f"P2.1 feet_air_time weight {scales.get('feet_air_time', 0.0)}, target {reward_cfg['feet_air_time_target']} s,"
        f" cap {reward_cfg.get('feet_air_time_max')}",
        f"P2.2 foot_clearance weight {scales.get('foot_clearance', 0.0)}, clearance target "
        f"{reward_cfg['feet_height_target']} m, std {reward_cfg.get('foot_clearance_std')}",
        f"P2.3 symmetry {sym_mode}" + (f", mirror loss coef {sym['mirror_loss_coeff']}" if sym else ""),
        f"P2.4 gait terms {gait or 'none'}" + (f", clock {clock}" if clock else ""),
        f"P2.5 energy weight {scales.get('energy', 0.0)}",
        f"penalty curriculum: negative terms x {reward_cfg.get('penalty_scale_init', 1.0)} -> 1 over "
        f"{reward_cfg.get('penalty_ramp_steps', 0) / train_cfg['num_steps_per_env']:g} iterations (P2.4 / P2.5 "
        f"penalties included)",
    ]
    print("\n".join(["--- gait quality (P2) ---"] + lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp-name", type=str, default="go2-walking", help="Experiment name; also the log directory under logs/"
    )
    parser.add_argument("-b", "--num-envs", type=int, default=4096, help="Number of parallel environments")
    # [P1.1] 101 iterations were enough for one fixed command; a 3-D command space with standing needs ~10x more
    parser.add_argument("--max-iterations", type=int, default=2000,
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
    # [P2] gait quality. P1 now trains with the air-time target 0.3 s and the foot-clearance reward ([fix], [S2R]):
    # --air-time and --clearance only change their targets (sweeps); --symmetry, --gait, --energy add what P1 lacks.
    # TEST_AND_TINKER_P2.md has one command line per part.
    parser.add_argument("--air-time", type=float, default=None, metavar="T",
                        help="[P2.1] feet air-time target [s] (P1 default: 0.3)")
    parser.add_argument("--air-time-max", type=float, default=None, metavar="C",
                        help="[P2.1] cap [s] on the air time paid at touchdown: the swing duration the term prefers "
                             "(P1: uncapped, so it always prefers fewer, longer steps)")
    parser.add_argument("--clearance", type=float, default=None, metavar="H",
                        help="[P2.2] foot clearance target [m] of the foot_clearance reward (P1 default: 0.075)")
    parser.add_argument("--symmetry", nargs="?", const="aug", default=None, choices=("aug", "loss", "both", "log"),
                        help="[P2.3] rsl_rl symmetry: data augmentation (default when given), mirror loss, both, or "
                             "log (Loss/symmetry is logged, training unchanged)")
    parser.add_argument("--gait", choices=("spot", "wtw"), default=None,
                        help="[P2.4] gait pattern: spot (GaitReward + air-time variance) or wtw (contact schedule + "
                             "clock inputs; drops feet_air_time)")
    parser.add_argument("--energy", type=float, nargs="?", const=ENERGY_WEIGHT, default=None, metavar="W",
                        help=f"[P2.5] mechanical power penalty -W * sum|tau qdot| (when given alone: {ENERGY_WEIGHT})")
    parser.add_argument("--set", nargs="+", default=[], metavar="GROUP.KEY=VALUE",
                        help="[P2] override any config entry last, e.g. reward.reward_scales.foot_clearance=0.5 "
                             "env.joint_friction=0.3 (groups: env, obs, reward, command, train)")
    args = parser.parse_args()
    if args.rnn and args.symmetry:  # [P2.3] rsl_rl 5.5.1 PPO raises "not supported for recurrent policies"
        parser.error("--symmetry is not supported by rsl_rl for recurrent policies: drop --rnn or --symmetry")
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
    # [fix] 400 iterations. [S2R] bug fix: this line was inside the if above, so MLP runs trained with it never had
    # the ramp; penalty_scale_init alone does nothing without it.
    reward_cfg["penalty_ramp_steps"] = 400 * train_cfg["num_steps_per_env"]
    # [P2.1] longer steps: the air-time target (P1 default 0.3 s; sweep it here)
    if args.air_time is not None:
        reward_cfg["feet_air_time_target"] = args.air_time
    if args.air_time_max is not None:
        reward_cfg["feet_air_time_max"] = args.air_time_max
    # [P2.2] foot clearance: target of the foot_clearance reward (P1 default 0.075 m, weight 1.0)
    if args.clearance is not None:
        reward_cfg["feet_height_target"] = args.clearance
    # [P2.3] symmetry: rsl_rl's extension with the Go2 left-right mirror (go2_symmetry.py)
    if args.symmetry:
        train_cfg["algorithm"]["symmetry_cfg"] = {
            "data_augmentation_func": "go2_symmetry:compute_symmetric_states",
            "use_data_augmentation": args.symmetry in ("aug", "both"),
            "use_mirror_loss": args.symmetry in ("loss", "both"),
            "mirror_loss_coeff": MIRROR_LOSS_COEF if args.symmetry in ("loss", "both") else 0.0,
        }
    # [P2.4] gait pattern (only if the gait is still irregular after P2.1-P2.3)
    if args.gait == "spot":
        reward_cfg["reward_scales"].update(SPOT_GAIT)
    elif args.gait == "wtw":
        env_cfg["gait_clock"] = {**WTW_CLOCK, "offsets": dict(WTW_CLOCK["offsets"])}
        reward_cfg["reward_scales"].update(WTW_GAIT)
        # walk-these-ways has no air-time term: the clock sets the swing time (0.25 s at 2 Hz), and P1's 0.3 s target
        # would make every scheduled step cost reward. Put it back with --set reward.reward_scales.feet_air_time=1.0
        # (and a target below the clock's swing, --air-time 0.2)
        reward_cfg["reward_scales"].pop("feet_air_time", None)
    # [P2.5] energy: mechanical power penalty (reported as power and CoT by go2_metrics.py)
    if args.energy is not None:
        reward_cfg["reward_scales"]["energy"] = -abs(args.energy)
    # [P2] --set overrides come last, so they win over every flag above
    apply_overrides({"env": env_cfg, "obs": obs_cfg, "reward": reward_cfg, "command": command_cfg, "train": train_cfg},
                    args.set)
    clock = env_cfg.get("gait_clock")
    if clock and reward_cfg["reward_scales"].get("feet_air_time", 0.0) > 0:
        swing = (1.0 - clock.get("duty", 0.5)) / clock.get("frequency", 2.0)
        if reward_cfg["feet_air_time_target"] > swing:
            print(f"WARNING: the clock asks for {swing:.2f} s swings but feet_air_time pays only above "
                  f"{reward_cfg['feet_air_time_target']} s: lower --air-time, or remove the term with "
                  f"--set reward.reward_scales.feet_air_time=None (walk-these-ways does not use it)")
    if args.lcp > 0 or args.caps_t > 0 or args.caps_s > 0:  # [P1.4]
        train_cfg["algorithm"].update(
            {
                "class_name": "go2_smooth_ppo:SmoothPPO",
                "grad_penalty_coef": args.lcp,
                "caps_temporal_coef": args.caps_t,
                "caps_spatial_coef": args.caps_s,
                # CAPS spatial noise: the observation-noise scales of P1.2 (0 on commands and last actions)
                # [P2.4] ... and on the gait clock
                "caps_sigma": obs_noise_scales(obs_cfg, env_cfg["num_actions"], command_cfg["num_commands"],
                                               num_clock_obs(env_cfg)),
            }
        )

    p2_summary(env_cfg, reward_cfg, command_cfg, train_cfg)  # [P2]

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
