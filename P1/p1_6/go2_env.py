import math

import torch
from tensordict import TensorDict

import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

from go2_commands import apply_command_deadzone  # [P1.1]

# [P1.1] Foot links of the Go2 URDF. They are fixed to the calves, so Genesis merges them into the calf
# links unless they are listed in `links_to_keep`; kept separate, their contact force can be read.
FOOT_LINKS = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")

# [P1.2] Motor limits of the real Go2, from Unitree's go2_description URDF (unitree_ros). The go2.urdf bundled
# with Genesis has Go1-like calf limits (35.55 Nm, 20.07 rad/s), so they are set explicitly here.
GO2_TORQUE_LIMITS = {"hip": 23.7, "thigh": 23.7, "calf": 45.43}  # [Nm]
GO2_DOF_VEL_LIMITS = {"hip": 30.1, "thigh": 30.1, "calf": 15.70}  # [rad/s] [P1.3] soft joint velocity limits

# [P1.6] Go2 leg segments (go2.urdf bundled with Genesis: calf and foot joints at z = -0.213 m from their parent).
# Used by nominal_dof_pos(), the leg pose of a commanded body height.
GO2_THIGH_LENGTH = 0.213  # [m] thigh joint -> calf joint
GO2_CALF_LENGTH = 0.213  # [m] calf joint -> foot sphere centre

# [P1.2] Physics used when a config does not say otherwise (runs trained before P1.2 included), so that every
# policy is evaluated in the same simulated world.
PHYSICS_DEFAULTS = {
    "sim_dt": 0.005,  # [s] physics step: 200 Hz, as legged_gym / Isaac Lab
    "decimation": 4,  # physics steps per policy step -> policy at 1 / (0.005 * 4) = 50 Hz
    "armature": 0.01,  # [kg m^2] reflected rotor inertia (MuJoCo Menagerie go2.xml); Genesis' URDF default is 0.1
    "latency_s": 0.02,  # [s] action latency without randomization (the example's "1 step latency")
}


def gs_rand(lower, upper, batch_shape):
    assert lower.shape == upper.shape
    return (upper - lower) * torch.rand(size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device) + lower


def uniform(lower, upper, shape):
    """[P1.2] Uniform samples in [lower, upper] (scalars) of the given shape."""
    return (upper - lower) * torch.rand(size=shape, dtype=gs.tc_float, device=gs.device) + lower


def obs_noise_scales(obs_cfg, num_actions, num_commands=3):
    """[P1.2] Half-width of the uniform observation noise, per observation dimension (legged_gym noise_scales, in
    physical units, scaled like the observation). [P1.4] Moved out of Go2Env so go2_train.py can reuse it.
    [P1.6] num_commands: 4 with the body-height command, so the vector keeps the size of the observation."""
    noise_scales, obs_scales = obs_cfg["noise_scales"], obs_cfg["obs_scales"]
    segments = (
        (3, noise_scales["ang_vel"] * obs_scales["ang_vel"]),
        (3, noise_scales["gravity"]),
        (num_commands, 0.0),  # commands
        (num_actions, noise_scales["dof_pos"] * obs_scales["dof_pos"]),
        (num_actions, noise_scales["dof_vel"] * obs_scales["dof_vel"]),
        (num_actions, 0.0),  # last actions
    )
    n_level = obs_cfg.get("noise_level", 1.0)
    return [n_level * s for n, s in segments for _ in range(n)]


def nominal_world(env_cfg, obs_cfg, latency_s=None):
    """[P1.2] Evaluation world: no domain randomization, no observation noise, no pushes, a fixed latency.

    Call it on the configs loaded from cfgs.pkl before building the env (go2_eval.py, go2_metrics.py).
    """
    env_cfg["domain_rand"] = {}
    obs_cfg["add_noise"] = False
    env_cfg["recovery"] = {}  # [P1.5] no fallen starts
    if latency_s is not None:
        env_cfg["latency_s"] = latency_s


class Go2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs: int = num_envs
        self.num_actions = env_cfg["num_actions"]
        self.cfg = env_cfg
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        # [P1.2] physics at 1 / sim_dt (200 Hz), policy at 1 / (sim_dt * decimation) (50 Hz). The PD controller runs
        # at the physics rate (see step()). self.dt stays the POLICY period: rewards, episode length, metrics use it.
        self.sim_dt = env_cfg.get("sim_dt", PHYSICS_DEFAULTS["sim_dt"])
        self.decimation = env_cfg.get("decimation", PHYSICS_DEFAULTS["decimation"])
        self.dt = self.sim_dt * self.decimation  # 0.02 s
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales: dict[str, float] = obs_cfg["obs_scales"]
        self.reward_scales: dict[str, float] = reward_cfg["reward_scales"]
        self.domain_rand: dict = env_cfg.get("domain_rand", {})  # [P1.2] empty = no randomization

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.sim_dt,  # [P1.2] was dt=0.02 with substeps=2 (100 Hz physics, PD inside the solver)
                substeps=1,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                # For this locomotion policy, there are usually no more than 20 collision pairs. Setting a low value
                # can save memory. Violating this condition will raise an exception.
                max_collision_pairs=20,
                tolerance=1e-5,
                # [P1.2] per-env link mass / center of mass (domain randomization) need batched link info
                batch_links_info=bool({"added_mass_range", "com_shift_range"} & self.domain_rand.keys()),
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=[0],
            ),
            viewer_options=gs.options.ViewerOptions(
                # [P1.6] configurable (go2_teleop.py: chase camera); the defaults are the former fixed values
                camera_pos=tuple(self.env_cfg.get("viewer_camera_pos", (2.0, 0.0, 2.5))),
                camera_lookat=tuple(self.env_cfg.get("viewer_camera_lookat", (0.0, 0.0, 0.5))),
                camera_fov=40,
            ),
            show_viewer=show_viewer,
        )

        # add plain
        self.plane = self.scene.add_entity(  # [P1.2] kept: its friction is randomized too
            gs.morphs.URDF(
                file="urdf/plane/plane.urdf",
                fixed=True,
            )
        )

        # [P1.1] always keep the foot links (contacts for the air-time reward), plus any extra link requested
        self.foot_links = tuple(self.env_cfg.get("foot_links", FOOT_LINKS))
        links_to_keep = tuple(dict.fromkeys((*self.foot_links, *self.env_cfg.get("links_to_keep", ()))))

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.env_cfg["base_init_pos"],
                quat=self.env_cfg["base_init_quat"],
                links_to_keep=links_to_keep,
                # [P1.2] rotor inertia added to every joint (Genesis' default for URDF is 0.1)
                default_armature=self.env_cfg.get("armature", PHYSICS_DEFAULTS["armature"]),
            ),
        )

        self._add_extra_entities()  # [P1.6] hook: cameras etc. must be added before build (go2_teleop.py --record)
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_int,
            device=gs.device,
        )
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)
        # [P1.2] the 12 motor dofs are contiguous (local dofs 6..17): written as a slice, in dof order
        self.motors_dof_slice = slice(int(self.motors_dof_idx.min()), int(self.motors_dof_idx.max()) + 1)
        assert self.motors_dof_slice.stop - self.motors_dof_slice.start == self.num_actions
        # [P1.2] world-frame linear x, y velocity dofs of the floating base (pushes)
        base_dof = self.robot.joints[0].dof_start
        self.base_lin_xy_dofs = [base_dof, base_dof + 1]
        self.base_link_idx = 0
        # [P1.1] foot link indices (local to the robot), same order as self.foot_links
        self.feet_link_idx = torch.tensor(
            [self.robot.get_link(name).idx_local for name in self.foot_links], dtype=gs.tc_int, device=gs.device
        )
        # [P1.3] contact-checked links, by name substring as legged_gym (penalize_contacts_on /
        # terminate_after_contacts_on): Go2 links are base, XX_hip, XX_thigh, XX_calf, XX_foot
        def links_matching(patterns):
            idx = [link.idx_local for link in self.robot.links if any(p in link.name for p in patterns)]
            return torch.tensor(idx, dtype=gs.tc_int, device=gs.device)

        self.penalised_contact_link_idx = links_matching(self.env_cfg.get("penalize_contacts_on", ()))
        self.termination_contact_link_idx = links_matching(self.env_cfg.get("terminate_after_contacts_on", ()))

        # [P1.2] PD control: computed in step() at the physics rate and applied as a torque, so that gains, motor
        # strength and torque clipping can differ per env without batched dof info (legged_gym _compute_torques).
        # The Genesis internal PD (set_dofs_kp / set_dofs_kv + control_dofs_position) is no longer used.
        self.kp_nominal = self.env_cfg["kp"]
        self.kd_nominal = self.env_cfg["kd"]
        joint_type = [name.split("_")[1] for name in self.env_cfg["joint_names"]]  # "hip" / "thigh" / "calf"
        torque_limits = self.env_cfg.get("torque_limits", GO2_TORQUE_LIMITS)
        self.torque_limits = torch.tensor([torque_limits[t] for t in joint_type], dtype=gs.tc_float, device=gs.device)
        # the solver clamps applied forces to its force range: make it match the clipping done here
        self.robot.set_dofs_force_range(-self.torque_limits, self.torque_limits, self.motors_dof_idx)

        # [P1.3] soft joint limits (legged_gym): a share of the URDF position range, centered, and of the motor
        # velocity limits; beyond them the dof_pos_limits / dof_vel_limits rewards penalize
        q_lo, q_hi = self.robot.get_dofs_limit(self.motors_dof_idx)
        q_lo, q_hi = (q_lo[0], q_hi[0]) if q_lo.ndim == 2 else (q_lo, q_hi)  # per-env copies if dof info is batched
        q_mid, q_half = 0.5 * (q_lo + q_hi), 0.5 * (q_hi - q_lo) * self.reward_cfg.get("soft_dof_pos_limit", 1.0)
        self.dof_pos_soft_limits = (q_mid - q_half, q_mid + q_half)
        dof_vel_limits = self.env_cfg.get("dof_vel_limits", GO2_DOF_VEL_LIMITS)
        self.dof_vel_soft_limits = self.reward_cfg.get("soft_dof_vel_limit", 1.0) * torch.tensor(
            [dof_vel_limits[t] for t in joint_type], dtype=gs.tc_float, device=gs.device
        )

        # Define global gravity direction vector
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device)

        # Initial state
        self.init_base_pos = torch.tensor(self.env_cfg["base_init_pos"], dtype=gs.tc_float, device=gs.device)
        self.init_base_quat = torch.tensor(self.env_cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device)
        self.inv_base_init_quat = inv_quat(self.init_base_quat)
        self.init_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][joint.name] for joint in self.robot.joints[1:]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.init_qpos = torch.concatenate((self.init_base_pos, self.init_base_quat, self.init_dof_pos))
        self.init_projected_gravity = transform_by_quat(self.global_gravity, self.inv_base_init_quat)

        # initialize buffers
        self.base_lin_vel = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_ang_vel = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.projected_gravity = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.rew_buf = torch.empty((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.reset_buf = torch.ones((self.num_envs,), dtype=gs.tc_bool, device=gs.device)
        self.episode_length_buf = torch.empty((self.num_envs,), dtype=gs.tc_int, device=gs.device)
        # [P1.6] commands = (vx, vy, wz), plus the body-height offset dh [m] when num_commands is 4. Scales and limits
        # are built from lists, so 3 and 4 commands share one code path.
        if self.num_commands not in (3, 4):
            raise ValueError(f"num_commands must be 3 (vx, vy, wz) or 4 (+ body height dh), not {self.num_commands}")
        self.height_cmd = self.num_commands == 4
        command_scales = [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]]
        command_ranges = [
            self.command_cfg["lin_vel_x_range"],
            self.command_cfg["lin_vel_y_range"],
            self.command_cfg["ang_vel_range"],
        ]
        if self.height_cmd:
            command_scales.append(self.obs_scales["body_height"])
            command_ranges.append(self.command_cfg["body_height_range"])
        self.commands = torch.empty((self.num_envs, self.num_commands), dtype=gs.tc_float, device=gs.device)
        self.commands_scale = torch.tensor(command_scales, device=gs.device, dtype=gs.tc_float)
        self.commands_limits: tuple[torch.Tensor, torch.Tensor] = tuple(
            torch.tensor(values, dtype=gs.tc_float, device=gs.device) for values in zip(*command_ranges)
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_last_actions = torch.zeros_like(self.actions)  # [P1.4] action of two policy steps ago
        self.dof_pos = torch.empty_like(self.actions)
        self.dof_vel = torch.empty_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.torques = torch.zeros_like(self.actions)  # [P1.2] torques applied at the last physics step [Nm]
        # [P1.4] torques of every physics step of the last policy step, for the 200 Hz analysis in go2_metrics.py
        self.torques_substeps = torch.zeros(
            (self.num_envs, self.decimation, self.num_actions), dtype=gs.tc_float, device=gs.device
        )
        self.base_pos = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_quat = torch.empty((self.num_envs, 4), dtype=gs.tc_float, device=gs.device)
        self.base_euler = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        # [P1.6] leg geometry for nominal_dof_pos(): per leg (FR, FL, RR, RL), the indices of its thigh and calf
        # joints (policy order), and where the default pose puts the foot centre in the leg's sagittal plane,
        # relative to the thigh joint (the thigh joints are at the height of the base origin, go2.urdf)
        names = self.env_cfg["joint_names"]
        thigh_idx = [i for i, name in enumerate(names) if name.endswith("_thigh_joint")]
        calf_idx = [names.index(names[i].replace("_thigh_joint", "_calf_joint")) for i in thigh_idx]
        self.leg_thigh_idx = torch.tensor(thigh_idx, dtype=torch.long, device=gs.device)
        self.leg_calf_idx = torch.tensor(calf_idx, dtype=torch.long, device=gs.device)
        th0, kn0 = self.default_dof_pos[self.leg_thigh_idx], self.default_dof_pos[self.leg_calf_idx]
        l1, l2 = GO2_THIGH_LENGTH, GO2_CALF_LENGTH
        self.leg_foot_x = -(l1 * torch.sin(th0) + l2 * torch.sin(th0 + kn0))  # [4] forward of the thigh joint [m]
        self.leg_foot_depth = l1 * torch.cos(th0) + l2 * torch.cos(th0 + kn0)  # [4] below the thigh joint [m]
        # [P1.1] foot contact state (legged_gym _reward_feet_air_time) and the zero-command flag
        n_feet = len(self.foot_links)
        self.last_feet_contact = torch.zeros((self.num_envs, n_feet), dtype=gs.tc_bool, device=gs.device)
        self.feet_contact = torch.zeros_like(self.last_feet_contact)  # filtered contact
        self.feet_first_contact = torch.zeros_like(self.last_feet_contact)  # touchdown at this step
        self.feet_air_time = torch.zeros((self.num_envs, n_feet), dtype=gs.tc_float, device=gs.device)
        self.feet_air_time_at_touchdown = torch.zeros_like(self.feet_air_time)
        self.zero_cmd = torch.zeros((self.num_envs,), dtype=gs.tc_bool, device=gs.device)
        # [P1.5] learned recovery (optional, off unless env_cfg["recovery"]["init_prob"] > 0): a share of the
        # episodes starts from a random fallen pose (MetalHead recovery_init_prob), with zero command and without
        # the tilt and base-contact terminations for the whole episode.
        recovery = self.env_cfg.get("recovery") or {}
        self.recovery_init_prob = recovery.get("init_prob", 0.0)
        self.recovery_spawn_height = recovery.get("spawn_height", 0.5)  # [m] base height of the fallen start
        self.recovering = torch.zeros((self.num_envs,), dtype=gs.tc_bool, device=gs.device)
        # [P1.6] recovery success counts (fix of the P1.5 log, see _reset_idx)
        self.rec_ended_ema = torch.zeros((), dtype=gs.tc_float, device=gs.device)
        self.rec_success_ema = torch.zeros((), dtype=gs.tc_float, device=gs.device)
        # [P1.3] |net contact force| per robot link, updated in step()
        self.link_contact_force = torch.zeros((self.num_envs, self.robot.n_links), dtype=gs.tc_float, device=gs.device)

        # [P1.2] per-episode actuator randomization (sampled in _reset_idx)
        self.kp = torch.full((self.num_envs, 1), self.kp_nominal, dtype=gs.tc_float, device=gs.device)
        self.kd = torch.full((self.num_envs, 1), self.kd_nominal, dtype=gs.tc_float, device=gs.device)
        self.motor_strength = torch.ones((self.num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device)
        self.latency_steps = torch.zeros((self.num_envs,), dtype=torch.long, device=gs.device)
        latency_s = self.env_cfg.get("latency_s", PHYSICS_DEFAULTS["latency_s"])
        lat_lo, lat_hi = self.domain_rand.get("latency_range_s", (latency_s, latency_s))
        # in physics steps, at most one policy step (the previous target is held for that many physics steps)
        self.latency_steps_range = (
            min(round(lat_lo / self.sim_dt), self.decimation),
            min(round(lat_hi / self.sim_dt), self.decimation),
        )

        # [P1.2] pushes (legged_gym _push_robots): all envs every push_interval_s, sync-free
        self.common_step_counter = 0
        push_interval_s = self.domain_rand.get("push_interval_s", 0.0)
        self.push_interval = round(push_interval_s / self.dt) if push_interval_s > 0 else 0
        self.max_push_vel_xy = self.domain_rand.get("max_push_vel_xy", 0.0)

        # [P1.2] observation noise (legged_gym noise_scales, in physical units, scaled like the observation)
        noise_scales = self.obs_cfg.get("noise_scales", {})
        self.add_noise = bool(self.obs_cfg.get("add_noise", False) and noise_scales)
        if self.add_noise:
            # [P1.4] computed by obs_noise_scales(), also used for the CAPS sigma in go2_train.py
            self.noise_scale_vec = torch.tensor(
                obs_noise_scales(self.obs_cfg, self.num_actions, self.num_commands), dtype=gs.tc_float, device=gs.device
            )

        # [P1.2] friction, base mass and CoM: sampled once per env, after build (Genesis DR guide; legged_gym also
        # samples friction and mass at env creation)
        self._randomize_physics()

        self.extras = dict()  # extra information for logging

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)

        self.reset()

    def _add_extra_entities(self):
        """[P1.6] Called just before scene.build(): a subclass adds cameras or objects here (go2_teleop.py)."""

    def nominal_dof_pos(self, dh):
        """[P1.6] Joint pose (policy order) that moves the base by dh [m] (tensor [N]) with each foot where the default
        pose puts it: planar 2-link IK on thigh and calf, hips at their default angle. nominal_dof_pos(0) is the
        default pose. Out of reach, the calf angle saturates (straight leg) instead of returning NaN."""
        l1, l2 = GO2_THIGH_LENGTH, GO2_CALF_LENGTH
        x = self.leg_foot_x  # [4]
        depth = self.leg_foot_depth + dh[:, None]  # [N, 4]
        cos_knee = ((x * x + depth * depth - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)).clamp(-1.0, 1.0)
        knee = -torch.acos(cos_knee)  # calf angle: negative = bent, knee behind the hip
        thigh = torch.atan2(-x, depth) - torch.atan2(l2 * torch.sin(knee), l1 + l2 * torch.cos(knee))
        q = self.default_dof_pos.repeat(dh.shape[0], 1)
        q[:, self.leg_thigh_idx] = thigh
        q[:, self.leg_calf_idx] = knee
        return q

    def _randomize_physics(self):
        # [P1.2] Values are kept (self.friction, ...) for logging, or for a privileged critic later.
        dr, N = self.domain_rand, self.num_envs
        self.friction = torch.ones((N,), dtype=gs.tc_float, device=gs.device)
        self.added_mass = torch.zeros((N,), dtype=gs.tc_float, device=gs.device)
        self.com_shift = torch.zeros((N, 3), dtype=gs.tc_float, device=gs.device)
        if "friction_range" in dr:
            # Genesis takes max(mu_a, mu_b) for a contact pair and every geom starts at mu = 1, so scaling the robot
            # alone would never make the ground slipperier: ground and robot get the same ratio -> contact mu = ratio.
            self.friction = uniform(*dr["friction_range"], (N,))
            n_links = self.robot.n_links
            self.robot.set_friction_ratio(self.friction[:, None].expand(N, n_links).contiguous(), list(range(n_links)))
            self.plane.set_friction_ratio(self.friction[:, None], [0])
        if "added_mass_range" in dr:
            # set_mass_shift() is gone in Genesis 1.4: set_links_mass() takes the absolute mass (batched link info)
            self.added_mass = uniform(*dr["added_mass_range"], (N,))
            mass = self.robot.get_links_mass([self.base_link_idx])  # [N, 1]
            self.robot.set_links_mass(mass + self.added_mass[:, None], [self.base_link_idx])
        if "com_shift_range" in dr:
            # set_COM_shift() is gone too: set_links_COM() takes the CoM position in the link frame
            self.com_shift = uniform(*dr["com_shift_range"], (N, 3))
            com = self.robot.get_links_COM([self.base_link_idx])  # [N, 1, 3]
            self.robot.set_links_COM(com + self.com_shift[:, None, :], [self.base_link_idx])

    def _fallen_starts(self, envs_idx):
        # [P1.5] A share recovery_init_prob of the reset envs restarts from a random fallen state, dropped from
        # spawn_height: uniformly random base orientation (MetalHead samples random Euler angles) and joint angles
        # uniform within the soft limits. The flag stays for the whole episode.
        N = self.num_envs
        reset = torch.ones((N,), dtype=gs.tc_bool, device=gs.device) if envs_idx is None else envs_idx
        fallen = reset & (torch.rand((N,), device=gs.device) < self.recovery_init_prob)
        self.recovering = torch.where(reset, fallen, self.recovering)

        quat = torch.randn((N, 4), dtype=gs.tc_float, device=gs.device)
        quat = quat / torch.linalg.norm(quat, dim=1, keepdim=True)  # uniform random rotation (w, x, y, z)
        lower, upper = self.dof_pos_soft_limits
        q = lower + (upper - lower) * torch.rand((N, self.num_actions), dtype=gs.tc_float, device=gs.device)
        qpos = self.init_qpos.repeat(N, 1)
        qpos[:, 2] = self.recovery_spawn_height
        qpos[:, 3:7] = quat
        qpos[:, self.motors_dof_idx + 1] = q  # qpos = [pos 3, quat 4, joints]: dof d is qpos entry d + 1
        # full [N, 19] tensor + boolean mask: written with torch.where (Genesis zero-copy, default on CPU and CUDA)
        self.robot.set_qpos(qpos, envs_idx=fallen, zero_velocity=True, skip_forward=True)

        torch.where(fallen[:, None], qpos[:, :3], self.base_pos, out=self.base_pos)
        torch.where(fallen[:, None], quat, self.base_quat, out=self.base_quat)
        gravity = transform_by_quat(self.global_gravity, inv_quat(quat))
        torch.where(fallen[:, None], gravity, self.projected_gravity, out=self.projected_gravity)
        torch.where(fallen[:, None], q, self.dof_pos, out=self.dof_pos)

    def _resample_actuation(self, envs_idx):
        # [P1.2] Per-episode actuator randomization: Kp, Kd (per env), motor strength (per env and joint),
        # action latency (per env, in physics steps). Drawn for all envs, applied to the ones being reset.
        dr, N = self.domain_rand, self.num_envs
        kp = self.kp_nominal * uniform(*dr.get("kp_range", (1.0, 1.0)), (N, 1))
        kd = self.kd_nominal * uniform(*dr.get("kd_range", (1.0, 1.0)), (N, 1))
        strength = uniform(*dr.get("motor_strength_range", (1.0, 1.0)), (N, self.num_actions))
        lo, hi = self.latency_steps_range
        latency = torch.randint(lo, hi + 1, (N,), dtype=torch.long, device=gs.device)
        if envs_idx is None:
            self.kp.copy_(kp)
            self.kd.copy_(kd)
            self.motor_strength.copy_(strength)
            self.latency_steps.copy_(latency)
        else:
            torch.where(envs_idx[:, None], kp, self.kp, out=self.kp)
            torch.where(envs_idx[:, None], kd, self.kd, out=self.kd)
            torch.where(envs_idx[:, None], strength, self.motor_strength, out=self.motor_strength)
            torch.where(envs_idx, latency, self.latency_steps, out=self.latency_steps)

    def _push_robots(self):
        # [P1.2] legged_gym _push_robots: a random planar velocity kick for all envs every push_interval_s.
        # Added to the current velocity (as Isaac Lab push_by_setting_velocity), i.e. an impulse of up to
        # max_push_vel_xy per axis. Free-joint dofs 0 and 1 are the base velocity along world x and y.
        vel = self.robot.get_dofs_velocity(self.base_lin_xy_dofs)
        vel += uniform(-self.max_push_vel_xy, self.max_push_vel_xy, (self.num_envs, 2))
        self.robot.set_dofs_velocity(vel, self.base_lin_xy_dofs)

    def _resample_commands(self, envs_idx):
        commands = gs_rand(*self.commands_limits, (self.num_envs,))
        # [P1.1] Zero small commands (legged_gym _resample_commands), with the same rule as at deployment
        # (go2_commands.py), so that "stop" is an exact command the policy sees often.
        commands = apply_command_deadzone(
            commands, self.command_cfg.get("lin_deadzone", 0.0), self.command_cfg.get("ang_deadzone", 0.0)
        )
        # [P1.1] A share of the envs gets an exact zero command (Isaac Lab rel_standing_envs).
        standing = torch.rand((self.num_envs,), dtype=gs.tc_float, device=gs.device) < self.command_cfg.get(
            "rel_standing_envs", 0.0
        )
        # [P1.6] velocities only: a standing env keeps its body-height offset (standing still while crouched is a
        # valid command). With 3 commands, [:, :3] is the whole tensor.
        commands[:, :3].masked_fill_(standing[:, None], 0.0)
        if self.recovery_init_prob > 0:  # [P1.5] recovery episodes: get up and stand still (dh = 0 too)
            commands.masked_fill_(self.recovering[:, None], 0.0)
        if envs_idx is None:
            self.commands.copy_(commands)
        else:
            torch.where(envs_idx[:, None], commands, self.commands, out=self.commands)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])

        # [P1.2] pushes (the policy only sees their effect at its next observation)
        self.common_step_counter += 1
        if self.push_interval > 0 and self.common_step_counter % self.push_interval == 0:
            self._push_robots()

        # [P1.2] decimation loop: physics and PD at 1 / sim_dt, policy at 1 / dt. Action latency: the previous
        # target is held for the first `latency_steps` physics steps (0 .. decimation, sampled per episode),
        # instead of the example's fixed one-policy-step delay (= latency_steps == decimation).
        target_new = self.actions * self.env_cfg["action_scale"] + self.default_dof_pos
        target_old = self.last_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        for i in range(self.decimation):
            target = torch.where((i < self.latency_steps)[:, None], target_old, target_new)
            q = self.robot.get_dofs_position(self.motors_dof_idx)
            qd = self.robot.get_dofs_velocity(self.motors_dof_idx)
            torques = self.motor_strength * (self.kp * (target - q) - self.kd * qd)
            self.torques = torch.clip(torques, -self.torque_limits, self.torque_limits)  # URDF effort limits
            self.torques_substeps[:, i] = self.torques  # [P1.4]
            self.robot.control_dofs_force(self.torques[:, self.actions_dof_idx], self.motors_dof_slice)
            self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat), rpy=True, degrees=True
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)
        # [P1.3] |net contact force| of every robot link, read once and shared by feet, collisions, termination
        self.link_contact_force = torch.linalg.norm(self.robot.get_links_net_contact_force(), dim=-1)  # [N, n_links]
        self._update_feet_contacts()  # [P1.1]
        # [P1.1] "zero command" = all three components zero (the deadzone makes it exact)
        # [P1.6] velocities only: the body-height offset is not a motion command
        self.zero_cmd = torch.linalg.norm(self.commands[:, :3], dim=1) < 0.1

        # compute reward
        self.rew_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        # [P1.3] legged_gym only_positive_rewards: with many penalties and contact termination, a negative total
        # would make ending the episode early attractive; clipping at 0 removes that incentive
        # [P1.5] ... except in recovery episodes, which cannot end early: their penalties must stay informative
        if self.reward_cfg.get("only_positive_rewards", False):
            self.rew_buf = torch.where(self.recovering, self.rew_buf, self.rew_buf.clip(min=0.0))

        self._resample_commands(self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        fallen = torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        fallen |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        # [P1.3] base contact termination (legged_gym check_termination: |F| > 1 N on terminate_after_contacts_on)
        if len(self.termination_contact_link_idx) > 0:
            contact_force = self.link_contact_force[:, self.termination_contact_link_idx]
            fallen |= torch.any(contact_force > self.reward_cfg.get("termination_contact_force", 1.0), dim=1)
        self.reset_buf |= fallen & ~self.recovering  # [P1.5] recovery episodes are not ended by a fall
        self.reset_buf |= self.scene.rigid_solver.get_error_envs_mask()

        # Compute timeout
        self.extras["time_outs"] = (self.episode_length_buf > self.max_episode_length).to(dtype=gs.tc_float)

        # Reset environment if necessary
        self._reset_idx(self.reset_buf)

        # update observations
        self._update_observation()

        self.last_last_actions.copy_(self.last_actions)  # [P1.4]
        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)

        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def _update_feet_contacts(self):
        # [P1.1] Foot contact and air time, as in legged_gym _reward_feet_air_time: a foot is in contact when its
        # contact force exceeds a threshold; the contact is OR-ed with the previous step to filter flicker.
        force = self.link_contact_force[:, self.feet_link_idx]  # [N, 4]  (|force|, read in step(), [P1.3])
        contact = force > self.reward_cfg.get("contact_force_threshold", 1.0)
        contact_filt = contact | self.last_feet_contact
        self.last_feet_contact.copy_(contact)
        self.feet_first_contact = (self.feet_air_time > 0.0) & contact_filt
        self.feet_air_time += self.dt
        self.feet_air_time_at_touchdown = self.feet_air_time * self.feet_first_contact
        self.feet_air_time.masked_fill_(contact_filt, 0.0)
        self.feet_contact = contact_filt

    def get_observations(self):
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    def _reset_idx(self, envs_idx=None):
        if self.recovery_init_prob > 0:  # [P1.5] success of the recovery episodes that end here (logged below)
            ended = self.recovering if envs_idx is None else self.recovering & envs_idx
            upright = (self.projected_gravity[:, 2] < -0.9) & (self.base_pos[:, 2] > 0.2)
            # [P1.6] fix: rsl_rl averages a logged value over every step of the iteration, and P1.5 logged 0 at the
            # steps where no recovery episode ended, so the curve stayed far below the true rate. Now the counts
            # decay by 0.99 per step (a window of about 100 steps) and their ratio is logged at every step.
            self.rec_ended_ema.mul_(0.99).add_(ended.sum())
            self.rec_success_ema.mul_(0.99).add_((ended & upright).sum())
            recovery_success = self.rec_success_ema / self.rec_ended_ema.clamp(min=1e-6)

        # reset state
        self.robot.set_qpos(self.init_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True)

        # reset buffers
        if envs_idx is None:
            self.base_pos.copy_(self.init_base_pos)
            self.base_quat.copy_(self.init_base_quat)
            self.projected_gravity.copy_(self.init_projected_gravity)
            self.dof_pos.copy_(self.init_dof_pos)
            self.base_lin_vel.zero_()
            self.base_ang_vel.zero_()
            self.dof_vel.zero_()
            self.actions.zero_()
            self.last_actions.zero_()
            self.last_last_actions.zero_()  # [P1.4]
            self.last_dof_vel.zero_()
            self.feet_air_time.zero_()  # [P1.1]
            self.last_feet_contact.zero_()  # [P1.1]
            self.episode_length_buf.zero_()
            self.reset_buf.fill_(True)
        else:
            torch.where(envs_idx[:, None], self.init_base_pos, self.base_pos, out=self.base_pos)
            torch.where(envs_idx[:, None], self.init_base_quat, self.base_quat, out=self.base_quat)
            torch.where(
                envs_idx[:, None], self.init_projected_gravity, self.projected_gravity, out=self.projected_gravity
            )
            torch.where(envs_idx[:, None], self.init_dof_pos, self.dof_pos, out=self.dof_pos)
            self.base_lin_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.base_ang_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_last_actions.masked_fill_(envs_idx[:, None], 0.0)  # [P1.4]
            self.last_dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.feet_air_time.masked_fill_(envs_idx[:, None], 0.0)  # [P1.1]
            self.last_feet_contact.masked_fill_(envs_idx[:, None], False)  # [P1.1]
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)

        self._resample_actuation(envs_idx)  # [P1.2]
        if self.recovery_init_prob > 0:  # [P1.5]
            self._fallen_starts(envs_idx)

        # fill extras
        n_envs = envs_idx.sum() if envs_idx is not None else self.num_envs
        self.extras["episode"] = {}
        if self.recovery_init_prob > 0:  # [P1.5] share of recovery episodes ending upright, base above 0.2 m
            self.extras["episode"]["recovery_success"] = recovery_success
        for key, value in self.episode_sums.items():
            if envs_idx is None:
                mean = value.mean()
            else:
                mean = torch.where(n_envs > 0, value[envs_idx].sum() / n_envs, 0.0)
            self.extras["episode"]["rew_" + key] = mean / self.env_cfg["episode_length_s"]
            if envs_idx is None:
                value.zero_()
            else:
                value.masked_fill_(envs_idx, 0.0)

        # random sample command upon reset
        self._resample_commands(envs_idx)

    def _update_observation(self):
        self.obs_buf = torch.concatenate(
            (
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3, or 4 with the body-height command [P1.6]
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
            ),
            dim=-1,
        )
        if self.add_noise:  # [P1.2] uniform noise, as legged_gym
            self.obs_buf += (2.0 * torch.rand_like(self.obs_buf) - 1.0) * self.noise_scale_vec

    def reset(self):
        self._reset_idx()
        self._update_observation()
        return self.get_observations()

    # Reward functions.
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_action_smoothness(self):
        # [P1.4] Second difference of the joint position targets (walk-these-ways action_smoothness_2), i.e. their
        # discrete acceleration. For a sinusoid, its square is 4 sin^2(pi f dt) times that of the first difference:
        # 4x at 25 Hz (an action flipping every step), ~1/16 at a 2 Hz gait. So it targets jitter, not the gait.
        # Masked on the first two steps of an episode, where the zeroed action history is not a real change.
        second_diff = self.actions - 2.0 * self.last_actions + self.last_last_actions
        second_diff = second_diff * self.env_cfg["action_scale"]  # actions -> joint targets [rad]
        return torch.sum(torch.square(second_diff), dim=1) * (self.episode_length_buf > 2)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        # [P1.6] with the body-height command, from the leg pose of the commanded height (= default pose at dh = 0)
        reference = self.nominal_dof_pos(self.commands[:, 3]) if self.height_cmd else self.default_dof_pos
        deviation = torch.sum(torch.abs(self.dof_pos - reference), dim=1)
        # [P1.1] x stand_still_scale at zero command while the base is (nearly) at rest (Isaac Lab Spot
        # joint_position_penalty). The speed gate keeps the normal weight while the base moves (e.g. after a
        # push), so the robot may still step to recover.
        base_speed = torch.linalg.norm(self.base_lin_vel[:, :2], dim=1)
        at_rest = base_speed < self.reward_cfg.get("stand_still_vel_threshold", 0.5)
        scale = self.reward_cfg.get("stand_still_scale", 1.0)
        return torch.where(self.zero_cmd & at_rest, scale * deviation, deviation)

    def _reward_base_height(self):
        # Penalize base height away from target
        target = self.reward_cfg["base_height_target"]
        if self.height_cmd:  # [P1.6] target + commanded offset (walk-these-ways _reward_jump)
            target = target + self.commands[:, 3]
        return torch.square(self.base_pos[:, 2] - target)

    def _reward_feet_air_time(self):
        # [P1.1] legged_gym: on each touchdown, reward (air time - target), i.e. long steps; this also stops the
        # feet from shuffling. Paid only for a non-zero command, otherwise it would reward stepping in place.
        # (legged_gym gates on |cmd_xy| > 0.1; the yaw command counts here too, so turning in place is stepped.)
        rew = torch.sum(
            (self.feet_air_time_at_touchdown - self.reward_cfg["feet_air_time_target"]) * self.feet_first_contact, dim=1
        )
        return rew * ~self.zero_cmd

    def _reward_orientation(self):
        # [P1.2] legged_gym: penalize a non-flat base. Takes over from the 10 deg tilt termination, which is
        # relaxed so that the robot can lean and recover from pushes.
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    # [P1.3] Safety and limits, legged_gym reward functions (legged_robot.py)
    def _reward_ang_vel_xy(self):
        # Penalize roll and pitch rates of the base
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_torques(self):
        # Penalize torques (those applied at the last physics step)
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_acc(self):
        # Penalize joint accelerations (finite difference over one policy step)
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_dof_vel(self):
        # Penalize joint velocities (legged_gym default weight 0: not enabled in go2_train.py)
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_pos_limits(self):
        # Penalize joint positions beyond the soft limits [rad]
        lower, upper = self.dof_pos_soft_limits
        out_of_limits = (lower - self.dof_pos).clip(min=0.0) + (self.dof_pos - upper).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize joint velocities beyond the soft limits, clipped at 1 rad/s per joint
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_soft_limits).clip(min=0.0, max=1.0), dim=1)

    def _reward_collision(self):
        # Penalize contacts on thighs, calves and base: number of such links touching (|F| > 0.1 N)
        contact_force = self.link_contact_force[:, self.penalised_contact_link_idx]
        return torch.sum((contact_force > self.reward_cfg.get("collision_force_threshold", 0.1)).float(), dim=1)

    # [P1.5] Learned recovery rewards (optional): positive, bounded, and paid in recovery episodes only.
    def _reward_rec_upright(self):
        # 1 upright, 0.5 on the side, 0 upside down
        return 0.5 * (1.0 - self.projected_gravity[:, 2]) * self.recovering

    def _reward_rec_height(self):
        # base height up to the target, counted only when upright (a robot on its back is not "high")
        height = (self.base_pos[:, 2] / self.reward_cfg["base_height_target"]).clip(0.0, 1.0)
        return height * self._reward_rec_upright()

    def _reward_rec_pose(self):
        # close to the default pose, once upright
        pose = torch.exp(-torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1))
        return pose * self._reward_rec_upright()
