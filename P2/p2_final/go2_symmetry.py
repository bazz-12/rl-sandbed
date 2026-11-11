"""
go2_symmetry.py - [P2.3] Left-right mirror of the Go2 observation and action, for rsl_rl's symmetry extension.

rsl-rl-lib 5.5.1 (rsl_rl/extensions/symmetry.py) calls
    compute_symmetric_states(env=env, obs=TensorDict [B] or None, actions=Tensor [B, 12] or None)
and expects [originals; mirrored] back (batch 2B, originals first). go2_train.py --symmetry puts
"go2_symmetry:compute_symmetric_states" in train_cfg["algorithm"]["symmetry_cfg"]; go2_metrics.py uses the same mirror
for its sym_err metric. Copy this file next to go2_env.py (rsl_rl imports it by name, also when a run is loaded by
go2_eval.py, go2_metrics.py, go2_export.py...).

The mirror is the reflection through the robot's sagittal (x-z) plane, y -> -y:
    legs                        FL <-> FR, RL <-> RR
    hip (abduction, x axis)     angle, velocity, action change sign; thigh and calf (y axis) do not
    base angular velocity       a pseudovector: (wx, wy, wz) -> (-wx, wy, -wz)
    projected gravity           a vector: (gx, gy, gz) -> (gx, -gy, gz)
    commands                    (vx, vy, wz[, dh]) -> (vx, -vy, -wz[, dh])
    [P2.4] gait clock           sin and cos per foot: FL <-> FR, RL <-> RR. For the trot schedule this is the same trot
                                half a period later, a state the clock produces too.
Every reward in go2_env.py is unchanged by the mirror (sums over feet and joints of mirrored quantities), which the
augmentation needs: mirrored samples reuse the advantages and returns of their originals.
Front-back is NOT a symmetry of the Go2 (all knees point backward, front and rear thighs have different default angles
and limits), unlike ANYmal in Isaac Lab's example: only left-right is used.

    python go2_symmetry.py          # checks: mirror twice = identity, and a mirrored Genesis rollout stays mirrored
"""

import torch

SWAP = {"FL": "FR", "FR": "FL", "RL": "RR", "RR": "RL"}


def joint_mirror(joint_names):
    """(perm, sign) for a vector in joint order: mirrored[i] = sign[i] * x[perm[i]]."""
    names = list(joint_names)
    perm, sign = [], []
    for name in names:
        leg, kind = name.split("_")[:2]
        perm.append(names.index(SWAP[leg] + name[len(leg):]))
        sign.append(-1.0 if kind == "hip" else 1.0)
    return perm, sign


def _spec(env, key, build):
    """Mirror (perm, sign) tensors on the env's device, built once per env."""
    cache = env.__dict__.setdefault("_mirror_cache", {})
    if key not in cache:
        perm, sign = build()
        cache[key] = (torch.tensor(perm, dtype=torch.long, device=env.device),
                      torch.tensor(sign, dtype=torch.float, device=env.device))
    return cache[key]


def _obs_layout(env):
    """(perm, sign) of the policy observation, block by block as Go2Env._update_observation() builds it."""
    joint_perm, joint_sign = joint_mirror(env.env_cfg["joint_names"])
    perm, sign = [], []

    def block(p, s):
        start = len(perm)
        perm.extend(start + i for i in p)
        sign.extend(s)

    block([0, 1, 2], [-1.0, 1.0, -1.0])  # base angular velocity
    block([0, 1, 2], [1.0, -1.0, 1.0])  # projected gravity
    block(list(range(env.num_commands)), [1.0, -1.0, -1.0, 1.0][: env.num_commands])  # vx, vy, wz[, dh]
    for _ in range(3):  # joint positions, joint velocities, last actions
        block(joint_perm, joint_sign)
    if getattr(env, "num_clock_obs", 0):  # [P2.4] sin of each foot, then cos of each foot
        feet = [name.split("_")[0] for name in env.foot_links]
        foot_perm = [feet.index(SWAP[f]) for f in feet]
        block(foot_perm, [1.0] * len(feet))
        block(foot_perm, [1.0] * len(feet))
    return perm, sign


def mirror_obs(env, obs):
    """Mirror a policy observation tensor [..., D]."""
    perm, sign = _spec(env, "obs", lambda: _obs_layout(env))
    if obs.shape[-1] != perm.numel():
        raise ValueError(f"go2_symmetry: observation of size {obs.shape[-1]}, the mirror knows the {perm.numel()} "
                         f"entries of Go2Env's policy observation. A new observation term needs its mirror in "
                         f"_obs_layout().")
    return obs[..., perm] * sign


def mirror_actions(env, actions):
    """Mirror an action tensor [..., 12] (policy joint order)."""
    perm, sign = _spec(env, "act", lambda: joint_mirror(env.env_cfg["joint_names"]))
    return actions[..., perm] * sign


@torch.no_grad()
def compute_symmetric_states(env, obs=None, actions=None):
    """rsl_rl symmetry function: originals followed by their left-right mirror (2 x the batch)."""
    obs_aug = None
    if obs is not None:
        batch = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        for key in obs.keys():  # every observation group ("policy"; the critic uses it too)
            obs_aug[key][batch:] = mirror_obs(env, obs[key])
    actions_aug = None
    if actions is not None:
        actions_aug = torch.cat((actions, mirror_actions(env, actions)), dim=0)
    return obs_aug, actions_aug


# ---------------------------------------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------------------------------------
def _check(steps=50, seed=0, gait_clock=False):
    """1) mirror twice = identity. 2) Physics: env 1 starts in the mirror of env 0's state (the default pose, which is
    its own mirror) and receives the mirrored command and mirrored actions; its observation must stay the mirror of
    env 0's. A sign error in the mirror shows up as a large difference in that block from the first steps."""
    import genesis as gs
    from tensordict import TensorDict

    import go2_train
    from go2_env import Go2Env, nominal_world

    gs.init(backend=gs.cpu, logging_level="warning", seed=seed)
    env_cfg, obs_cfg, reward_cfg, command_cfg = go2_train.get_cfgs()
    nominal_world(env_cfg, obs_cfg)
    reward_cfg["reward_scales"] = {}
    env_cfg.update(episode_length_s=1e3, resampling_time_s=1e3)
    if gait_clock:
        env_cfg["gait_clock"] = {"frequency": 2.0, "duty": 0.5}

    command = torch.tensor([0.5, 0.3, 0.4])

    class PinnedEnv(Go2Env):
        def _resample_commands(self, envs_idx):
            self.commands[0] = command.to(self.device)
            self.commands[1] = command.to(self.device) * torch.tensor([1.0, -1.0, -1.0], device=self.device)

    env = PinnedEnv(2, env_cfg, obs_cfg, reward_cfg, command_cfg)
    obs = env.reset()["policy"]
    if env.gait_clock:  # the mirror of a trot clock is the same clock half a period later
        env.gait_phase[1] = 0.5
        env._update_gait_targets()

    x = torch.randn(64, obs.shape[-1], device=env.device)
    a = torch.randn(64, env.num_actions, device=env.device)
    aug_obs, aug_act = compute_symmetric_states(env, TensorDict({"policy": x}, batch_size=[64]), a)
    twice = (mirror_obs(env, aug_obs["policy"][64:]) - x).abs().max().item()
    twice = max(twice, (mirror_actions(env, aug_act[64:]) - a).abs().max().item())
    print(f"mirror(mirror(x)) - x: {twice:.1e} (must be 0)")

    blocks = [("ang_vel", 3), ("gravity", 3), ("commands", env.num_commands), ("dof_pos", 12), ("dof_vel", 12),
              ("actions", 12)] + ([("clock", env.num_clock_obs)] if env.num_clock_obs else [])
    worst = {name: 0.0 for name, _ in blocks}
    gen = torch.Generator(device=env.device).manual_seed(seed)
    action = torch.zeros(env.num_actions, device=env.device)
    for _ in range(steps):  # smooth random actions, so the robot stays up for the whole check
        action = 0.8 * action + 0.2 * torch.randn(env.num_actions, generator=gen, device=env.device)
        actions = torch.stack((action, mirror_actions(env, action)))
        obs = env.step(actions)[0]["policy"]
        diff = (obs[1] - mirror_obs(env, obs[0])).abs()
        start = 0
        for name, size in blocks:
            worst[name] = max(worst[name], diff[start:start + size].max().item())
            start += size
    print(f"mirrored rollout, {steps} steps: max |obs_1 - mirror(obs_0)| per block")
    for name, value in worst.items():
        print(f"    {name:9s} {value:.1e}")
    print("Small values (contact solver round-off, ~1e-3 or below) = the mirror matches the physics. A block at the "
          "size of the signal itself (~0.1-1) = a wrong sign or permutation in that block.")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--clock", action="store_true", help="also check the [P2.4] gait-clock entries")
    args = p.parse_args()
    _check(args.steps, gait_clock=args.clock)
