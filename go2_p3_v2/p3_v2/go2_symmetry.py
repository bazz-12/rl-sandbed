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
    [P3.3] privileged group     entry by entry (Go2Env._privileged_obs): vectors in the base frame (velocity, CoM shift,
                                foot forces) flip y; per-joint and per-foot entries swap legs; scalars (friction, mass,
                                gains, latency) are unchanged; the height scan flips its y index.
Every reward in go2_env.py is unchanged by the mirror (sums over feet and joints of mirrored quantities), which the
augmentation needs: mirrored samples reuse the advantages and returns of their originals.
Front-back is NOT a symmetry of the Go2 (all knees point backward, front and rear thighs have different default angles
and limits), unlike ANYmal in Isaac Lab's example: only left-right is used.

    python go2_symmetry.py          # checks: mirror twice = identity, and a mirrored Genesis rollout stays mirrored
    python go2_symmetry.py --privileged             # [P3.3] the same with every privileged entry, randomized values
    python go2_symmetry.py --privileged --slope 10  # [P3.3] on a 10 deg slope (non-zero normals, height scan, slip)
[P3.3] compute_symmetric_states() mirrors every group of the TensorDict ("policy", and "privileged" for runs trained
with go2_train.py --privileged): rsl_rl augments the critic's inputs too, and the teacher's actor reads both groups.
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


def _privileged_mirror(env):
    """[P3.3] (perm, sign) of the privileged observation, entry by entry as Go2Env.privileged_layout lists them."""
    joint_perm, _ = joint_mirror(env.env_cfg["joint_names"])
    feet = [name.split("_")[0] for name in env.foot_links]
    foot_perm = [feet.index(SWAP[f]) for f in feet]
    nx, ny = env.height_scan_shape  # scan points x-major: index = ix * ny + iy, y symmetric about 0
    rules = {
        "base_lin_vel": ([0, 1, 2], [1.0, -1.0, 1.0]),  # a vector in the base frame: y flips
        "friction": ([0], [1.0]),
        "added_mass": ([0], [1.0]),
        "com_shift": ([0, 1, 2], [1.0, -1.0, 1.0]),  # base frame
        "motor_strength": (joint_perm, [1.0] * len(joint_perm)),  # a gain per joint: legs swap, no sign change
        "foot_forces": ([3 * foot_perm[f] + c for f in range(len(feet)) for c in range(3)],
                        [1.0, -1.0, 1.0] * len(feet)),  # base-frame vector per foot: feet swap, y flips
        "foot_slip": (foot_perm, [1.0] * len(feet)),
        "gains": ([0, 1], [1.0, 1.0]),
        "latency": ([0], [1.0]),
        "foot_clearance": (foot_perm, [1.0] * len(feet)),
        "height_scan": ([ix * ny + (ny - 1 - iy) for ix in range(nx) for iy in range(ny)],
                        [1.0] * (nx * ny)),  # the scan point at (x, y) becomes the one at (x, -y)
    }
    perm, sign = [], []
    for name, size in env.privileged_layout:
        if name not in rules:
            raise ValueError(f"go2_symmetry: no mirror rule for the privileged entry {name!r}: add it to "
                             f"_privileged_mirror()")
        p, s = rules[name]
        assert len(p) == size == len(s), (name, len(p), size)
        start = len(perm)
        perm.extend(start + i for i in p)
        sign.extend(s)
    return perm, sign


def mirror_obs(env, obs, group="policy"):
    """Mirror an observation tensor [..., D] of one group: "policy" (Go2Env._update_observation) or [P3.3]
    "privileged" (Go2Env._privileged_obs)."""
    if group == "policy":
        perm, sign = _spec(env, "obs", lambda: _obs_layout(env))
    elif group == "privileged":
        perm, sign = _spec(env, "privileged", lambda: _privileged_mirror(env))
    else:
        raise ValueError(f"go2_symmetry: no mirror for the observation group {group!r}")
    if obs.shape[-1] != perm.numel():
        raise ValueError(f"go2_symmetry: {group} observation of size {obs.shape[-1]}, the mirror knows "
                         f"{perm.numel()} entries of Go2Env's {group} observation. A new observation term needs its "
                         f"mirror in {'_obs_layout' if group == 'policy' else '_privileged_mirror'}().")
    return obs[..., perm] * sign


def mirror_observations(env, obs):
    """[P3.3] Mirror every group of an observation TensorDict (what a teacher's actor or a critic reads)."""
    out = obs.clone()
    for key in obs.keys():
        out[key] = mirror_obs(env, obs[key], key)
    return out


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
        for key in obs.keys():  # every observation group: "policy", [P3.3] "privileged"
            obs_aug[key][batch:] = mirror_obs(env, obs[key], key)
    actions_aug = None
    if actions is not None:
        actions_aug = torch.cat((actions, mirror_actions(env, actions)), dim=0)
    return obs_aug, actions_aug


# ---------------------------------------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------------------------------------
def _check(steps=50, seed=0, gait_clock=False, privileged=False, slope=0.0):
    """1) mirror twice = identity. 2) Physics: env 1 starts in the mirror of env 0's state (the default pose, which is
    its own mirror) and receives the mirrored command and mirrored actions; its observation must stay the mirror of
    env 0's. A sign error in the mirror shows up as a large difference in that block from the first steps.
    [P3.3] privileged: every privileged entry too, with randomized values that differ between left and right (env 0's
    base CoM shift and motor strengths are random, env 1 gets their mirror, so the physics stays mirrored only if the
    mirror rules are right). slope [deg]: on a plane rising along +x (the world is its own mirror about the robots' x-z
    plane), for non-trivial ground normals, foot slip and height scan."""
    import genesis as gs
    from tensordict import TensorDict

    import go2_train
    from go2_env import Go2Env, PRIVILEGED_SIZES, nominal_world

    gs.init(backend=gs.cpu, logging_level="warning", seed=seed)
    env_cfg, obs_cfg, reward_cfg, command_cfg = go2_train.get_cfgs()
    nominal_world(env_cfg, obs_cfg)
    reward_cfg["reward_scales"] = {}
    env_cfg.update(episode_length_s=1e3, resampling_time_s=1e3)
    if gait_clock:
        env_cfg["gait_clock"] = {"frequency": 2.0, "duty": 0.5}
    if privileged:  # [P3.3] every entry; equal friction and mass, a random CoM shift (needs batched link info)
        obs_cfg["privileged"] = list(PRIVILEGED_SIZES)
        env_cfg["domain_rand"] = {"friction_range": [0.7, 0.7], "added_mass_range": [1.0, 1.0],
                                  "com_shift_range": [-0.03, 0.03]}
    if slope:
        env_cfg["terrain"] = {"mode": "slope", "slope_deg": float(slope), "random_yaw": False, "spawn_xy_noise": 0.0}

    command = torch.tensor([0.5, 0.3, 0.4])

    class PinnedEnv(Go2Env):
        def _resample_commands(self, envs_idx):
            self.commands[0] = command.to(self.device)
            self.commands[1] = command.to(self.device) * torch.tensor([1.0, -1.0, -1.0], device=self.device)

    env = PinnedEnv(2, env_cfg, obs_cfg, reward_cfg, command_cfg)
    if privileged:  # env 1: the mirror of env 0's CoM shift (y flips) and of its motor strengths (legs swap)
        flip_y = torch.tensor([1.0, -1.0, 1.0], device=env.device)
        com = env.robot.get_links_COM([env.base_link_idx]) - env.com_shift[:, None, :]  # nominal CoM, both envs
        env.com_shift[1] = env.com_shift[0] * flip_y
        env.robot.set_links_COM(com + env.com_shift[:, None, :], [env.base_link_idx])
    obs = env.reset()
    if privileged:
        perm, _ = joint_mirror(env.env_cfg["joint_names"])
        gen0 = torch.Generator(device=env.device).manual_seed(seed + 1)
        env.motor_strength[0] = 0.9 + 0.2 * torch.rand(env.num_actions, generator=gen0, device=env.device)
        env.motor_strength[1] = env.motor_strength[0][perm]
    if env.gait_clock:  # the mirror of a trot clock is the same clock half a period later
        env.gait_phase[1] = 0.5
        env._update_gait_targets()

    groups = list(obs.keys())
    x = TensorDict({k: torch.randn(64, obs[k].shape[-1], device=env.device) for k in groups}, batch_size=[64])
    a = torch.randn(64, env.num_actions, device=env.device)
    aug_obs, aug_act = compute_symmetric_states(env, x, a)
    twice = max((mirror_obs(env, aug_obs[k][64:], k) - x[k]).abs().max().item() for k in groups)
    twice = max(twice, (mirror_actions(env, aug_act[64:]) - a).abs().max().item())
    print(f"mirror(mirror(x)) - x: {twice:.1e} (must be 0), groups {groups}")

    blocks = {"policy": [("ang_vel", 3), ("gravity", 3), ("commands", env.num_commands), ("dof_pos", 12),
                         ("dof_vel", 12), ("actions", 12)] + ([("clock", env.num_clock_obs)] if env.num_clock_obs
                                                              else [])}
    if "privileged" in groups:
        blocks["privileged"] = list(env.privileged_layout)
    worst = {(g, name): [0.0, 0.0] for g in blocks for name, _ in blocks[g]}  # max |difference|, max |signal|
    gen = torch.Generator(device=env.device).manual_seed(seed)
    action = torch.zeros(env.num_actions, device=env.device)
    for _ in range(steps):  # smooth random actions, so the robot stays up for the whole check
        action = 0.8 * action + 0.2 * torch.randn(env.num_actions, generator=gen, device=env.device)
        actions = torch.stack((action, mirror_actions(env, action)))
        obs = env.step(actions)[0]
        for g, layout in blocks.items():
            diff = (obs[g][1] - mirror_obs(env, obs[g][0], g)).abs()
            start = 0
            for name, size in layout:
                w = worst[(g, name)]
                w[0] = max(w[0], diff[start:start + size].max().item())
                w[1] = max(w[1], obs[g][0, start:start + size].abs().max().item())
                start += size
    world = f"{slope:g} deg slope" if slope else "flat ground"
    print(f"mirrored rollout, {steps} steps on {world}: max |obs_1 - mirror(obs_0)| per block "
          f"(max |obs_0| in brackets)")
    for (g, name), (value, signal) in worst.items():
        print(f"    {g:10s} {name:15s} {value:.1e}   ({signal:.2g})")
    print("Small values (contact solver round-off, ~1e-3 or below) = the mirror matches the physics. A block at the "
          "size of the signal itself (~0.1-1) = a wrong sign or permutation in that block. [P3.3] foot_forces and "
          "foot_slip follow the contacts, which can differ by one solver iteration between the two envs: up to ~1e-2 "
          "there is still round-off; a wrong rule gives the size of the signal.")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--clock", action="store_true", help="also check the [P2.4] gait-clock entries")
    p.add_argument("--privileged", action="store_true", help="[P3.3] also check every privileged entry")
    p.add_argument("--slope", type=float, default=0.0, metavar="DEG", help="[P3.3] on a plane tilted by DEG")
    args = p.parse_args()
    _check(args.steps, gait_clock=args.clock, privileged=args.privileged, slope=args.slope)
