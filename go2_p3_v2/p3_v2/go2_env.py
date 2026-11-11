import hashlib
import inspect
import math

import numpy as np
import torch
from tensordict import TensorDict

import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

from go2_commands import apply_command_deadzone  # [P1.1]
from go2_terrain import make_ground, terrain_config  # [P3.1]
from go2_terrain import contact_slip_speed  # [P3.2]

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

# [S2R] Joint losses of unitree_mujoco's go2.xml (<joint damping="0.1" frictionloss="0.2"/>). The go2.urdf bundled
# with Genesis has no <dynamics> tag, so its joints are lossless unless env_cfg sets these. A config without the keys
# (runs trained before [S2R]) keeps lossless joints, so old runs are evaluated as they were trained.
GO2_JOINT_FRICTION = 0.2  # [Nm] Coulomb (dry) friction, solved as a constraint by Genesis, not part of self.torques
GO2_JOINT_DAMPING = 0.1  # [Nm s/rad] viscous damping, on top of the PD's kd
GO2_FOOT_RADIUS = 0.023  # [m] [S2R] foot sphere: its centre is this high when the foot stands on flat ground
# [S2R] lower leg of unitree_mujoco's go2.xml: calf body with the foot sphere, mass [kg] and CoM in the calf frame [m]
# (|y| mirrored per side). go2.urdf has a lighter calf (0.154 kg, CoM at z = -0.115) plus a 0.04 kg foot link at -0.213.
GO2_MJCF_LOWER_LEG = {"mass": 0.241352, "com": (0.00629595, 0.000622121, -0.141417)}
URDF_FOOT_MASS, URDF_FOOT_POS = 0.04, (-0.002, 0.0, -0.213)  # go2.urdf foot link, in the calf frame

# [P2.4] Trot: feet that move together, as indices in FOOT_LINKS order (FL, FR, RL, RR): FL-RR and FR-RL
# (go2_metrics.py; Go2Env builds its own self.trot_pairs from env_cfg["foot_links"]). The same indices hold in the
# SDK leg order FR, FL, RR, RL (go2_kinematics.LEGS, go2_analyze.py): FR-RL and FL-RR.
TROT_PAIRS = ((0, 3), (1, 2))
# [P2.4] Walk-these-ways trot phase offsets ("phases" = 0.5, offsets = bounds = 0): FL and RR half a cycle after FR, RL
WTW_TROT_OFFSETS = {"FL": 0.5, "FR": 0.0, "RL": 0.0, "RR": 0.5}

# [P3.3] legged_gym measured_points_x / _y (LeggedRobotCfg.terrain): a 1.6 m x 1.0 m grid every 0.1 m, in the base's
# heading frame (x forward, y left; rotated by the base yaw only), points ordered x-major (index = ix * 11 + iy)
HEIGHT_SCAN_X = (-0.8, 0.8, 17)
HEIGHT_SCAN_Y = (-0.5, 0.5, 11)
HEIGHT_SCAN_OFFSET = 0.3  # [m] legged_gym subtracts 0.5 (ANYmal's base height); the Go2 stands at 0.3 m

# [P3.3] Privileged observation group (obs_cfg["privileged"], go2_train.py --privileged): what the simulator knows and
# the robot cannot measure, for the critic (asymmetric actor-critic) or a teacher actor. Entry -> size. Every entry is
# noise-free and scaled to about +-1 by FIXED constants (in the comments, used in Go2Env._privileged_obs()), so a value
# means the same thing in every run, whatever its randomization range. The vector holds the entries in list order;
# go2_symmetry.py has the left-right mirror of each one. Feet in foot_links order (FL, FR, RL, RR), joints in policy
# order (env_cfg["joint_names"]).
PRIVILEGED_SIZES = {
    "base_lin_vel": 3,  # true base linear velocity, base frame [m/s]: x obs_scales.lin_vel (2.0)
    "friction": 1,  # contact friction coefficient of the env (P1.2 randomization): (mu - 0.8) / 0.4
    "added_mass": 1,  # mass added to the base [kg] (P1.2): (m - 0.5) / 1.5
    "com_shift": 3,  # base CoM offset, base frame [m] (P1.2): / 0.03
    "motor_strength": 12,  # PD torque factor per joint (P1.2): (k - 1) / 0.1
    "foot_forces": 12,  # net contact force on each foot, base frame [N]: / 100, clipped to +-3
    "foot_slip": 4,  # [P3.2] contact-point slip speed per foot, 0 when not in contact [m/s]: / 0.2, clipped at 5
    # optional: not in go2_train.PRIVILEGED_DEFAULT, add them with --privileged-entries
    "gains": 2,  # Kp and Kd factors of the episode (P1.2): (k - 1) / 0.15
    "latency": 1,  # action latency of the episode, as a share of the policy step (P1.2): 0 .. 1
    "foot_clearance": 4,  # height of each foot-sphere bottom above the ground under it [m]: / 0.1, clipped -1 .. 3
    "height_scan": HEIGHT_SCAN_X[2] * HEIGHT_SCAN_Y[2],  # ground around the base (187 points): clip(base z - 0.3 - h,
    # -1, 1) x 5
}


def privileged_layout(obs_cfg):
    """[P3.3] [(entry, size), ...] of obs_cfg["privileged"] in vector order; [] when it lists none (runs before P3.3).
    Used by Go2Env, go2_train.py (sizes before any env exists) and go2_symmetry.py."""
    entries = list(obs_cfg.get("privileged") or ())
    unknown = [e for e in entries if e not in PRIVILEGED_SIZES]
    if unknown:
        raise ValueError(f"obs_cfg['privileged']: unknown entries {unknown}; known: {list(PRIVILEGED_SIZES)}")
    if len(set(entries)) != len(entries):
        raise ValueError(f"obs_cfg['privileged']: an entry is listed twice: {entries}")
    return [(e, PRIVILEGED_SIZES[e]) for e in entries]


def height_scan_points():
    """[P3.3] The 187 scan points (x, y) in the heading frame [m], x-major, as a [187, 2] tensor on gs.device."""
    x = torch.linspace(*HEIGHT_SCAN_X, dtype=gs.tc_float, device=gs.device)
    y = torch.linspace(*HEIGHT_SCAN_Y, dtype=gs.tc_float, device=gs.device)
    gx, gy = torch.meshgrid(x, y, indexing="ij")
    return torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)


def num_clock_obs(env_cfg):
    """[P2.4] Number of gait-clock observations: sin and cos of each foot's phase with --gait wtw, else 0."""
    return 2 * len(env_cfg.get("foot_links", FOOT_LINKS)) if env_cfg.get("gait_clock") else 0


def gait_targets(phase, offsets, duty, kappa):
    """[P2.4] Walk-these-ways contact schedule (go1_gym legged_robot.py _step_contact_targets), as a function.

    phase [N] in [0, 1): global gait phase. offsets [F]: per-foot phase offsets. duty: stance share of the cycle.
    Returns (foot_phase [N, F], desired_contact [N, F]). foot_phase is warped so that stance is [0, 0.5) and swing
    [0.5, 1) whatever the duty; desired_contact is ~1 in stance and ~0 in swing, smoothed by normal CDFs of width kappa
    at the transitions (WTW kappa_gait_probs = 0.07). go2_fsm.py gait_clock() is the numpy twin of the clock part.
    """
    raw = torch.remainder(phase[:, None] + offsets, 1.0)
    warped = torch.where(raw < duty, raw * (0.5 / duty), 0.5 + (raw - duty) * (0.5 / (1.0 - duty)))
    cdf = lambda x: 0.5 * (1.0 + torch.erf(x / (kappa * math.sqrt(2.0))))  # normal CDF, mean 0, std kappa
    desired = cdf(warped) * (1.0 - cdf(warped - 0.5)) + cdf(warped - 1.0) * (1.0 - cdf(warped - 1.5))
    return warped, desired


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


def obs_noise_scales(obs_cfg, num_actions, num_commands=3, num_clock=0):
    """[P1.2] Half-width of the uniform observation noise, per observation dimension (legged_gym noise_scales, in
    physical units, scaled like the observation). [P1.4] Moved out of Go2Env so go2_train.py can reuse it.
    [P1.6] num_commands: 4 with the body-height command, so the vector keeps the size of the observation.
    [P2.4] num_clock: gait-clock entries at the end of the observation (no noise: computed, not measured)."""
    noise_scales, obs_scales = obs_cfg["noise_scales"], obs_cfg["obs_scales"]
    segments = (
        (3, noise_scales["ang_vel"] * obs_scales["ang_vel"]),
        (3, noise_scales["gravity"]),
        (num_commands, 0.0),  # commands
        (num_actions, noise_scales["dof_pos"] * obs_scales["dof_pos"]),
        (num_actions, noise_scales["dof_vel"] * obs_scales["dof_vel"]),
        (num_actions, 0.0),  # last actions
        (num_clock, 0.0),  # [P2.4] gait clock
    )
    n_level = obs_cfg.get("noise_level", 1.0)
    return [n_level * s for n, s in segments for _ in range(n)]


def nominal_world(env_cfg, obs_cfg, latency_s=None):
    """[P1.2] Evaluation world: no domain randomization, no observation noise, no pushes, a fixed latency.

    Call it on the configs loaded from cfgs.pkl before building the env (go2_eval.py, go2_metrics.py).
    [P3.1] ... and flat ground: a run trained on terrain is measured on the same plane as every other run. The terrain
    evaluations (go2_metrics.py --terrain grid / --slope) put their ground back after this call.
    """
    env_cfg["domain_rand"] = {}
    obs_cfg["add_noise"] = False
    env_cfg["recovery"] = {}  # [P1.5] no fallen starts
    env_cfg["terrain"] = None  # [P3.1]
    if latency_s is not None:
        env_cfg["latency_s"] = latency_s


# [P3.1] Genesis versions whose SupportField.activate() source was checked against the replacement below (sha1 prefix
# of inspect.getsource). Any other source: the replacement is not used, and Genesis builds the terrain its own way.
_SUPPORT_FIELD_SOURCES = {"eb0321f73f3db143": "1.4.2"}


def skip_terrain_support_field():
    """[P3.1] Build-time fix for terrains. At scene.build(), Genesis 1.4.2 precomputes for every collision geom the
    furthest vertex in each of 33480 directions (collider/support_field.py, SupportField.activate): for the terrain
    geom that is 33480 x every heightfield vertex, in chunks of 5e8 float64 (4 GB). Measured on 2 CPU cores: 35 s of a
    45 s build for a 245k-vertex grid (TEST_AND_TINKER_P3.md 4.8: build cost). Terrain contacts never read that table:
    the narrow phase builds a prism per heightfield cell and takes its support from the prism (_func_support_prism in
    mpr.py and gjk_support.py). This replacement fills the terrain's block with vertex 0 and computes every other geom
    exactly as Genesis does. Genesis 1.4.3 made the same change upstream ("The support of a terrain is read off the
    prism built from its height field, so its table would never be read"), so this is that fix for 1.4.2: it is
    installed only when Genesis' function is the 1.4.2 one (source hash), and does nothing on 1.4.3+. Returns True when
    installed (go2_env.TERRAIN_FAST_BUILD = False disables it). Checked here: same trajectories with and without it."""
    from genesis.engine.solvers.rigid.collider import support_field as sf
    import genesis.utils.array_class as array_class

    if getattr(sf.SupportField.activate, "_go2_terrain_skip", False):
        return True
    source = hashlib.sha1(inspect.getsource(sf.SupportField.activate).encode()).hexdigest()[:16]
    if source not in _SUPPORT_FIELD_SOURCES:
        if "TERRAIN" not in inspect.getsource(sf.SupportField.activate):  # 1.4.3+ skip terrains themselves
            print(f"[go2_env] Genesis {gs.__version__}: SupportField.activate differs from 1.4.2; terrain built the "
                  f"default way (slow, several GB of RAM for big grids)")
        return False

    def activate(self):
        if self.is_active:
            return
        v1 = self._get_direction_grid().reshape([-1, 3])
        num_v = len(v1)
        support_v, support_vid, support_cell_start, n_support_cells = [], [], [], 0
        if self.solver.n_geoms > 0:
            init_pos = self.solver.dyn_info.verts.init_pos.to_numpy()
            geoms_vert_start = self.solver.dyn_info.geoms.vert_start.to_numpy()
            geoms_vert_end = self.solver.dyn_info.geoms.vert_end.to_numpy()
            geoms_type = self.solver.dyn_info.geoms.type.to_numpy()
            for i_g in range(self.solver.n_geoms):
                this_pos = init_pos[geoms_vert_start[i_g] : geoms_vert_end[i_g]]
                max_indices = np.zeros(num_v, dtype=np.intp)
                if geoms_type[i_g] != gs.GEOM_TYPE.TERRAIN:  # the only change: terrain supports come from prisms
                    window_size = int(5e8 // this_pos.shape[0])
                    for i in range(0, num_v, window_size):
                        end = min(i + window_size, num_v)
                        max_indices[i:end] = np.argmax(v1[i:end] @ this_pos.T, axis=1)
                support = this_pos[max_indices]
                support_cell_start.append(n_support_cells)
                support_v.append(support)
                support_vid.append(max_indices)
                n_support_cells += support.shape[0]
            support_v = np.concatenate(support_v)
            support_vid = np.concatenate(support_vid, dtype=gs.np_int)
            support_cell_start = np.array(support_cell_start, dtype=gs.np_int)
        else:
            support_v = np.zeros((1, 3), dtype=gs.np_float)
            support_vid = np.zeros((1,), dtype=gs.np_int)
            support_cell_start = np.zeros((1,), dtype=gs.np_int)
        self._support_field_info = array_class.get_support_field_info(
            self.solver.n_geoms, n_support_cells, self._support_res
        )
        sf._kernel_init_support(
            support_cell_start, support_v, support_vid, self._support_field_info, self.solver.rigid_config
        )
        self._is_active = True

    activate._go2_terrain_skip = True
    sf.SupportField.activate = activate
    return True


TERRAIN_FAST_BUILD = True  # [P3.1] skip_terrain_support_field() when a terrain is built (see its docstring)


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
        # [P3.1] ground: flat plane (no env_cfg["terrain"]: every run before P3), a tilted plane (mode "slope") or the
        # curriculum grid of tiles (mode "grid"), see go2_terrain.py. None of P3.1 runs on flat ground, so flat runs
        # are the P2 runs, bit for bit.
        self.terrain_cfg = terrain_config(env_cfg.get("terrain"))
        self.ground = make_ground(self.terrain_cfg)
        self.on_terrain = self.ground.kind != "flat"
        # [P3.1] Reset with forward kinematics. With skip_forward=True (the Genesis example's reset, kept on flat ground
        # so default runs stay identical to P2), get_dofs_position() and the link poses keep their pre-reset values until
        # the next physics step, so the explicit PD of the first substep (P1.2) pulls from the old joint angles: up to the
        # torque limit for 5 ms after every reset. Harmless in the air, but not the clean start an evaluation wants.
        # Default: on for terrain (new runs), off on flat ground; --set env.reset_forward=True turns it on there too.
        self.reset_forward = bool(env_cfg.get("reset_forward", self.on_terrain))

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
                # [S2R] per-env joint friction / damping (domain randomization) need batched dof info
                batch_dofs_info=self._joint_losses_randomized(),
                # [S2R] spin and rolling resistance at contacts (MuJoCo condim 6). Off unless env_cfg sets the foot
                # coefficients; costs solver time on every contact.
                enable_torsional_friction=self._contact_spin_friction(),
                enable_rolling_friction=self.env_cfg.get("foot_rolling_friction") is not None,
                # [S2R] friction cone and impratio. unitree_mujoco's go2.xml: <option cone="elliptic" impratio="100"/>,
                # i.e. stiff friction where stance feet do not creep. Genesis' default (pyramidal, impratio 1) lets
                # them creep. Configs without the keys keep Genesis' defaults.
                **self._contact_solver_options(),
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
        # [P3.1] self.plane is the ground entity whatever its shape (friction randomization, go2_fsm_sim.py --friction)
        if self.ground.kind == "grid":
            if TERRAIN_FAST_BUILD:
                skip_terrain_support_field()
            vs = self.terrain_cfg["vertical_scale"]
            self.plane = self.scene.add_entity(
                gs.morphs.Terrain(
                    height_field=self.ground.height_field / vs,  # Genesis multiplies by vertical_scale
                    horizontal_scale=self.ground.hs,
                    vertical_scale=vs,
                    pos=(float(self.ground.origin[0]), float(self.ground.origin[1]), 0.0),
                    name=f"go2_terrain_{self.ground.key}",  # identifier only: Genesis caches its own generator
                )
            )
        elif self.ground.kind == "slope":
            self.plane = self.scene.add_entity(gs.morphs.Plane(normal=self.ground.normal()))
        else:
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
            **self._robot_material(),  # [S2R] torsional / rolling friction coefficients, when set
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
        # [S2R] one row per env when dof info is batched (joint-loss randomization)
        self.robot.set_dofs_force_range(
            self._per_dof(-self.torque_limits), self._per_dof(self.torque_limits), self.motors_dof_idx
        )

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
        # [S2R] foot-sphere centre position and velocity, world frame, same foot order as self.foot_links
        self.feet_pos = torch.zeros((self.num_envs, n_feet, 3), dtype=gs.tc_float, device=gs.device)
        self.feet_vel = torch.zeros_like(self.feet_pos)
        # [P3.2] per foot, updated in step(): angular velocity (world), ground normal under the foot (constant (0, 0, 1)
        # on flat ground), net contact force (world, [N]) and the stance slip speed: speed of the contact point along
        # the ground while |F| > contact_force_threshold, 0 otherwise (go2_terrain.contact_slip_speed, as the harness)
        self.feet_ang = torch.zeros_like(self.feet_pos)
        self.feet_normal = torch.zeros_like(self.feet_pos)
        self.feet_normal[..., 2] = 1.0
        self.feet_force = torch.zeros_like(self.feet_pos)
        self.foot_slip = torch.zeros((self.num_envs, n_feet), dtype=gs.tc_float, device=gs.device)
        self.foot_slip_mode = self.reward_cfg.get("foot_slip_mode", "xy")  # [P3.2] _reward_foot_slip()
        if self.foot_slip_mode not in ("xy", "tangent"):
            raise ValueError(f"reward_cfg['foot_slip_mode'] must be 'xy' (Isaac Lab Spot) or 'tangent', "
                             f"not {self.foot_slip_mode!r}")
        # [P2.4] contact timers (Isaac Lab ContactSensor track_air_time) on the same filtered contact: feet_air_time is
        # the current air time, feet_contact_time the current contact time, and the last complete swing / stance
        # durations are kept for the air-time variance penalty
        self.feet_contact_time = torch.zeros_like(self.feet_air_time)
        self.feet_last_air_time = torch.zeros_like(self.feet_air_time)
        self.feet_last_contact_time = torch.zeros_like(self.feet_air_time)
        # [P2.5] mechanical power of the 12 motors, sum over joints of |tau * qdot|, averaged over the physics steps of
        # the last policy step [W] (energy reward; power and CoT in go2_metrics.py)
        self.mech_power = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        # [P2.4] optional walk-these-ways gait clock (env_cfg["gait_clock"], go2_train.py --gait wtw): a periodic
        # contact schedule whose per-foot phases are appended to the observation as sin / cos
        clock = self.env_cfg.get("gait_clock") or {}
        self.gait_clock = bool(clock)
        self.num_clock_obs = num_clock_obs(self.env_cfg)
        self.gait_frequency = float(clock.get("frequency", 2.0))  # [Hz]
        self.gait_duty = float(clock.get("duty", 0.5))  # stance share of the cycle
        self.gait_kappa = float(clock.get("kappa", 0.07))
        offsets = clock.get("offsets", WTW_TROT_OFFSETS)
        self.gait_offsets = torch.tensor(
            [offsets[name.split("_")[0]] for name in self.foot_links], dtype=gs.tc_float, device=gs.device
        )
        self.gait_phase = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)  # global phase in [0, 1)
        self.foot_phase = torch.zeros((self.num_envs, n_feet), dtype=gs.tc_float, device=gs.device)  # warped phase
        self.desired_contact = torch.ones_like(self.foot_phase)  # 1 = stance wanted, 0 = swing wanted
        # [P2.4] trot pairs as indices into self.foot_links (FL-RR, FR-RL), whatever the order of foot_links
        feet = [name.split("_")[0] for name in self.foot_links]
        self.trot_pairs = ((feet.index("FL"), feet.index("RR")), (feet.index("FR"), feet.index("RL")))
        # [P3.1] ground under the robot: height of the ground under each foot [N, 4] and base height above the mean of
        # those [N] (what base_height, foot_clearance, rec_height and the metrics measure). Flat ground: 0 and base z.
        self.feet_ground = torch.zeros((self.num_envs, n_feet), dtype=gs.tc_float, device=gs.device)
        self.base_height = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        # [P3.1] tilt limits per env [deg] (go2_metrics.py raises them on slopes: a robot on a 20 deg slope is not
        # fallen at 10 deg of pitch), and the flag of the episodes that ended by a fall (terrain curriculum, metrics)
        self.max_pitch_deg = torch.full((self.num_envs,), float(self.env_cfg["termination_if_pitch_greater_than"]),
                                        dtype=gs.tc_float, device=gs.device)
        self.max_roll_deg = torch.full((self.num_envs,), float(self.env_cfg["termination_if_roll_greater_than"]),
                                       dtype=gs.tc_float, device=gs.device)
        self.fell = torch.zeros((self.num_envs,), dtype=gs.tc_bool, device=gs.device)
        self._init_terrain_state()  # [P3.1]
        self._init_command_curriculum()  # [P3.1]

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
                obs_noise_scales(self.obs_cfg, self.num_actions, self.num_commands, self.num_clock_obs),  # [P2.4]
                dtype=gs.tc_float,
                device=gs.device,
            )
        # [P3.3] privileged observation group "privileged", next to "policy" (none unless obs_cfg["privileged"] lists
        # entries: every run before P3.3 returns "policy" only). Noise-free (add_noise applies to "policy" only).
        self.privileged_layout = privileged_layout(self.obs_cfg)
        self.num_privileged_obs = sum(size for _, size in self.privileged_layout)
        if self.privileged_layout and (self.num_actions != 12 or n_feet != 4):
            raise ValueError("the privileged entries are laid out for the Go2: 12 joints, 4 feet")
        self.privileged_buf = torch.zeros((self.num_envs, self.num_privileged_obs), dtype=gs.tc_float,
                                          device=gs.device)
        self.height_scan_points = height_scan_points() if "height_scan" in dict(self.privileged_layout) else None
        self.height_scan_shape = (HEIGHT_SCAN_X[2], HEIGHT_SCAN_Y[2])  # go2_symmetry.py: x-major, y index flips

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
        if {"contact_schedule_force", "contact_schedule_vel"} & self.reward_scales.keys() and not self.gait_clock:
            raise ValueError("the contact_schedule_* rewards need env_cfg['gait_clock'] (go2_train.py --gait wtw)")

        self.reset()
        self._curriculum_live = True  # [P3.1] the curricula judge episodes from now on

    def _add_extra_entities(self):
        """[P1.6] Called just before scene.build(): a subclass adds cameras or objects here (go2_teleop.py)."""

    def _contact_solver_options(self):
        """[S2R] RigidOptions for env_cfg["friction_cone"] ("pyramidal" / "elliptic") and env_cfg["impratio"]."""
        options = {}
        if self.env_cfg.get("friction_cone"):
            options["friction_cone"] = getattr(gs.friction_cone, self.env_cfg["friction_cone"])
        if self.env_cfg.get("impratio"):
            options["impratio"] = float(self.env_cfg["impratio"])
        return options

    def _contact_spin_friction(self):
        """[S2R] True when env_cfg asks for torsional friction at the robot's contacts (needed for rolling too)."""
        return self.env_cfg.get("foot_torsional_friction") is not None or \
            self.env_cfg.get("foot_rolling_friction") is not None

    def _robot_material(self):
        """[S2R] unitree_mujoco's go2.xml feet use condim="6" friction="0.4 0.02 0.01": besides sliding friction, a foot
        resists spinning about the contact normal (0.02 m) and rolling (0.01 m). A turn in place twists every stance
        foot by the yaw angle, so in MuJoCo (and on rubber feet) yawing costs torque that a point contact in Genesis
        never asks for. Set on the whole robot entity (only the feet touch the ground in normal walking)."""
        if not self._contact_spin_friction():
            return {}
        material = {"friction_torsional": self.env_cfg.get("foot_torsional_friction") or 0.0}
        if self.env_cfg.get("foot_rolling_friction") is not None:
            material["friction_rolling"] = self.env_cfg["foot_rolling_friction"]
        return {"material": gs.materials.Rigid(**material)}

    def _joint_losses_randomized(self):
        """[S2R] True when joint friction or damping is sampled per env (needs batched dof info)."""
        return bool({"joint_friction_range", "joint_damping_range"} & self.domain_rand.keys())

    def _per_dof(self, values):
        """[S2R] A per-joint tensor [12] as Genesis expects it for the motor dofs: [N, 12] (one row per env) when dof
        info is batched, [12] otherwise. Values that already have one row per env are passed through."""
        if not self._joint_losses_randomized() or values.ndim == 2:
            return values
        return values.expand(self.num_envs, -1).contiguous()

    def set_joint_losses(self, friction=None, damping=None):
        """[S2R] Set joint Coulomb friction [Nm] and viscous damping [Nm s/rad] of the 12 motor joints: a scalar, a [12]
        tensor or (with batched dof info) a [N, 12] tensor. None leaves a value unchanged. Also used by go2_fsm_sim.py
        (--joint-friction / --joint-damping), so the bench overrides the run's values in the same way."""
        for value, current, setter in (
            (friction, self.joint_friction, self.robot.set_dofs_frictionloss),
            (damping, self.joint_damping, self.robot.set_dofs_damping),
        ):
            if value is None:
                continue
            current[:] = torch.as_tensor(value, dtype=gs.tc_float, device=gs.device)  # broadcasts to [N, 12]
            setter(current if self._joint_losses_randomized() else current[0], self.motors_dof_idx)

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

    # ------------------------------------------------------------------------------------------------------------------
    # [P3.1] Terrain: tiles, spawning and the game-inspired curriculum (Rudin et al. 2022; legged_gym
    # _update_terrain_curriculum, Isaac Lab terrain_levels_vel); command curriculum (legged_gym update_command_curriculum)
    # ------------------------------------------------------------------------------------------------------------------
    def _init_terrain_state(self):
        N, dev = self.num_envs, gs.device
        self.spawn_pos = self.init_base_pos.repeat(N, 1)  # base position and orientation of the last reset
        self.spawn_quat = self.init_base_quat.repeat(N, 1)
        self.cmd_dist = torch.zeros((N,), dtype=gs.tc_float, device=dev)  # commanded planar distance, this episode [m]
        self.progress = torch.zeros((N,), dtype=gs.tc_float, device=dev)  # distance covered along the commands [m]
        self.terrain_levels = torch.zeros((N,), dtype=torch.long, device=dev)
        self.terrain_cols = torch.zeros((N,), dtype=torch.long, device=dev)
        self.terrain_types = torch.zeros((N,), dtype=torch.long, device=dev)
        self.terrain_curriculum = False
        self._curriculum_live = False  # off during the reset of __init__ (legged_gym init_done)
        if not self.on_terrain:
            return
        self.ground.to(dev)
        if self.ground.kind != "grid":
            return
        g, tc = self.ground, self.terrain_cfg
        self.tile_origins = torch.as_tensor(g.tile_origins, dtype=gs.tc_float, device=dev)  # [L, C, 3]
        self.col_type = torch.as_tensor(g.col_type, dtype=torch.long, device=dev)  # [C] type of each column
        self.num_levels = g.num_levels
        self.ground_min = float(g.height_field.min())  # below this - 1 m: fell off the grid
        self.terrain_cols, self.terrain_levels = self._terrain_assignment()
        self.terrain_types = self.col_type[self.terrain_cols]
        self.terrain_curriculum = bool(tc["curriculum"]) and not tc.get("spawn_tile")
        self.up_distance = 0.5 * g.size if tc["up_distance"] is None else float(tc["up_distance"])

    def _terrain_assignment(self):
        """[P3.1] Column and initial level of every env, [N] each (go2_metrics.py overrides it: one tile per bin).
        Columns follow the type shares deterministically (env i takes the column at quantile (i + 0.5) / N); initial
        levels are uniform in 0 .. max_init_level (legged_gym max_init_terrain_level)."""
        g, tc, N, dev = self.ground, self.terrain_cfg, self.num_envs, gs.device
        if tc.get("spawn_tile"):
            kind, level = tc["spawn_tile"]
            col = g.column_of_type(kind)
            level = int(min(max(int(level), 0), g.num_levels - 1))
            return (torch.full((N,), col, dtype=torch.long, device=dev),
                    torch.full((N,), level, dtype=torch.long, device=dev))
        cols = np.searchsorted(np.cumsum(g.col_share), (np.arange(N) + 0.5) / N).clip(max=g.num_cols - 1)
        top = min(int(tc["max_init_level"]), g.num_levels - 1)
        levels = torch.randint(0, top + 1, (N,), device=dev)
        return torch.as_tensor(cols, dtype=torch.long, device=dev), levels

    def _spawn_poses(self):
        """[P3.1] Base position and orientation [N, 3], [N, 4] for a reset, drawn for every env (fixed shapes), with a
        random heading when random_yaw:
        - grid: around the tile origin, level, just above the highest ground under the footprint;
        - slope: at the origin, standing on the plane: base parallel to it, at the initial height along its normal
          (a level start above a slope would drop the downhill legs onto the plane at the first steps);
        - flat ground: the configured initial pose."""
        N = self.num_envs
        pos, quat = self.init_base_pos.repeat(N, 1), self.init_base_quat.repeat(N, 1)
        if not self.on_terrain:
            return pos, quat
        tc = self.terrain_cfg
        if self.ground.kind == "grid":
            xy = self.tile_origins[self.terrain_levels, self.terrain_cols, :2]
            if tc["spawn_xy_noise"] > 0:
                xy = xy + uniform(-tc["spawn_xy_noise"], tc["spawn_xy_noise"], (N, 2))
        else:
            xy = pos[:, :2]
        if tc["random_yaw"]:
            yaw = uniform(-math.pi, math.pi, (N,))
            zero = torch.zeros_like(yaw)
            q_yaw = torch.stack((torch.cos(0.5 * yaw), zero, zero, torch.sin(0.5 * yaw)), dim=1)
            quat = transform_quat_by_quat(quat, q_yaw)  # yaw (world z) after the initial orientation
        if self.ground.kind == "slope":
            a = math.radians(self.ground.slope_deg)
            z = self.ground.height(xy) + self.init_base_pos[2] / math.cos(a)
            q_tilt = torch.tensor([[math.cos(0.5 * a), 0.0, -math.sin(0.5 * a), 0.0]], dtype=gs.tc_float,
                                  device=gs.device).repeat(N, 1)  # about world y: body up -> the plane's normal
            quat = transform_quat_by_quat(quat, q_tilt)  # after the heading
        else:
            z = self.ground.spawn_height(xy) + self.init_base_pos[2]
        return torch.cat((xy, z[:, None]), dim=1), quat

    def _update_terrain_curriculum(self, envs_idx):
        """[P3.1] Game-inspired curriculum, judged on the episodes that end now (before their reset):
        - a robot that walked farther than up_distance from its spawn point (half a tile: out of its tile), without
          falling, moves to the next level;
        - one that fell (when fall_moves_down), or that covered less than down_ratio of the distance its commands asked
          for (when they asked for at least down_min_cmd_dist), moves to the previous level;
        - one that moves up from the top level goes to a random level (keeps every level in the mix, legged_gym).
        legged_gym compares the distance walked with the LAST command x the episode length, which with commands
        resampled every 4 s judges a robot on a command it had for a few seconds; here the commanded distance is
        integrated over the episode, and the progress is measured along the command (step()), as go2_metrics.py's
        slip_ratio does."""
        if not (self.terrain_curriculum and self._curriculum_live):
            return
        tc, N = self.terrain_cfg, self.num_envs
        reset = torch.ones((N,), dtype=gs.tc_bool, device=gs.device) if envs_idx is None else envs_idx
        walked = torch.linalg.norm(self.base_pos[:, :2] - self.spawn_pos[:, :2], dim=1)
        up = reset & (walked > self.up_distance) & ~self.fell
        short = (self.cmd_dist >= tc["down_min_cmd_dist"]) & (self.progress < tc["down_ratio"] * self.cmd_dist)
        down = reset & ~up & (short | (self.fell & bool(tc["fall_moves_down"])))
        level = self.terrain_levels + up.long() - down.long()
        beyond = level >= self.num_levels
        level = torch.where(beyond, torch.randint_like(level, 0, self.num_levels), level.clamp(min=0))
        torch.where(reset, level, self.terrain_levels, out=self.terrain_levels)

    def _init_command_curriculum(self):
        """[P3.1] Command curriculum (legged_gym update_command_curriculum, extended to the three velocity axes): the
        sampling ranges start at command_cfg["curriculum"]["start"] and grow by "step" per side, up to the configured
        ranges (lin_vel_x_range, ...), each time the mean tracking reward of the episodes that ended since the last
        check exceeds "threshold" x its maximum (linear score for vx and vy, yaw score for wz). Checked every
        "interval_s" of simulated time (default: one episode). Off when command_cfg has no "curriculum"."""
        lo, hi = (t.clone() for t in self.commands_limits)
        self.cmd_limits = (lo, hi)  # current sampling limits; commands_limits stays the configured maximum
        cc = self.command_cfg.get("curriculum") or {}
        # it scores the two tracking rewards: evaluation scripts, which clear reward_scales, run without it
        self.cmd_curriculum = bool(cc) and {"tracking_lin_vel", "tracking_ang_vel"} <= self.reward_scales.keys()
        self.cmd_score = torch.zeros((2,), dtype=gs.tc_float, device=gs.device)  # summed linear, yaw scores
        self.cmd_score_n = torch.zeros((), dtype=gs.tc_float, device=gs.device)
        self.cmd_last_score = torch.zeros((2,), dtype=gs.tc_float, device=gs.device)
        if not self.cmd_curriculum:
            return
        names = ("lin_vel_x", "lin_vel_y", "ang_vel")
        start = cc.get("start", {})
        for k, name in enumerate(names):
            if name in start:
                lo[k] = max(float(min(start[name])), float(self.commands_limits[0][k]))
                hi[k] = min(float(max(start[name])), float(self.commands_limits[1][k]))
        step = cc.get("step", {})
        self.cmd_step = torch.tensor([float(step.get(n, d)) for n, d in zip(names, (0.1, 0.05, 0.1))],
                                     dtype=gs.tc_float, device=gs.device)
        self.cmd_threshold = float(cc.get("threshold", 0.8))
        self.cmd_interval = max(1, round(float(cc.get("interval_s", self.env_cfg["episode_length_s"])) / self.dt))

    def _score_command_curriculum(self, envs_idx):
        """[P3.1] Adds the mean tracking rewards (per step, in [0, 1]) of the episodes that end now and lasted at least
        1 s (the first, randomly shortened episodes of rsl_rl's init_at_random_ep_len are left out)."""
        if not (self.cmd_curriculum and self._curriculum_live):
            return
        reset = torch.ones((self.num_envs,), dtype=gs.tc_bool, device=gs.device) if envs_idx is None else envs_idx
        steps = self.episode_length_buf.clamp(min=1).to(gs.tc_float)
        full = reset & (self.episode_length_buf >= round(1.0 / self.dt))
        lin = self.episode_sums["tracking_lin_vel"] / (self.reward_scales["tracking_lin_vel"] * steps)
        ang = self.episode_sums["tracking_ang_vel"] / (self.reward_scales["tracking_ang_vel"] * steps)
        self.cmd_score += torch.stack(((lin * full).sum(), (ang * full).sum()))
        self.cmd_score_n += full.sum()

    def _update_command_curriculum(self):
        """[P3.1] Grow the ranges whose score passed the threshold (no GPU-CPU sync: tensor ops only)."""
        score = self.cmd_score / self.cmd_score_n.clamp(min=1.0)
        grow = (score > self.cmd_threshold) & (self.cmd_score_n > 0)
        amount = torch.stack((grow[0], grow[0], grow[1])).to(gs.tc_float) * self.cmd_step
        lo, hi = self.cmd_limits
        lo[:3] = torch.maximum(lo[:3] - amount, self.commands_limits[0][:3])
        hi[:3] = torch.minimum(hi[:3] + amount, self.commands_limits[1][:3])
        self.cmd_last_score = torch.where(self.cmd_score_n > 0, score, self.cmd_last_score)
        self.cmd_score.zero_()
        self.cmd_score_n.zero_()

    def _curriculum_extras(self, extras):
        """[P3.1] TensorBoard scalars (a "/" in the key: rsl_rl logs it under that name, not under Episode/)."""
        if self.ground.kind == "grid":
            level = self.terrain_levels.to(gs.tc_float)
            extras["Curriculum/terrain_level"] = level.mean()
            extras["Curriculum/top_level_share"] = (self.terrain_levels == self.num_levels - 1).to(gs.tc_float).mean()
            per_level = self.terrain_cfg["max_slope_deg"] / max(self.num_levels - 1, 1)
            for t, name in enumerate(self.ground.types):
                mask = (self.terrain_types == t).to(gs.tc_float)
                mean = (level * mask).sum() / mask.sum().clamp(min=1.0)
                extras[f"Curriculum/level_{name}"] = mean
                if name in ("hill", "pit"):
                    extras[f"Curriculum/slope_{name}_deg"] = mean * per_level
        if self.cmd_curriculum:
            lo, hi = self.cmd_limits
            extras["Curriculum/cmd_vx_max"] = torch.maximum(-lo[0], hi[0])
            extras["Curriculum/cmd_vy_max"] = torch.maximum(-lo[1], hi[1])
            extras["Curriculum/cmd_wz_max"] = torch.maximum(-lo[2], hi[2])
            extras["Curriculum/cmd_score_lin"] = self.cmd_last_score[0]
            extras["Curriculum/cmd_score_yaw"] = self.cmd_last_score[1]

    def curriculum_state(self):
        """[P3.1] Plain-python state of both curricula. go2_train.py saves it in every checkpoint (infos["curriculum"]),
        go2_export.py writes the command ranges the policy actually reached into deploy_cfg.json, go2_metrics.py
        flags commands outside them, and load_curriculum_state() restores it (go2_train.py --init-from)."""
        lo, hi = ([round(v, 4) for v in t.cpu().tolist()] for t in self.cmd_limits)
        state = {"env_steps": int(self.common_step_counter), "command_curriculum": self.cmd_curriculum,
                 "command_ranges": {"lin_vel_x": [lo[0], hi[0]], "lin_vel_y": [lo[1], hi[1]], "ang_vel": [lo[2], hi[2]]}}
        if self.ground.kind == "grid":
            levels, types = self.terrain_levels.cpu().numpy(), self.terrain_types.cpu().numpy()
            state["terrain"] = {
                "types": list(self.ground.types), "num_levels": int(self.num_levels),
                "max_slope_deg": float(self.terrain_cfg["max_slope_deg"]), "key": self.ground.key,
                "level_counts": {name: np.bincount(levels[types == t], minlength=self.num_levels).tolist()
                                 for t, name in enumerate(self.ground.types)},
            }
        return state

    def load_curriculum_state(self, state):
        """[P3.1] Restore the command ranges (within the configured maximum) and, on the same grid, the level
        distribution of each type (levels drawn from the saved counts)."""
        if not state:
            return
        lo, hi = self.cmd_limits
        for k, name in enumerate(("lin_vel_x", "lin_vel_y", "ang_vel")):
            r = state.get("command_ranges", {}).get(name)
            if r is not None and self.cmd_curriculum:
                lo[k] = max(float(r[0]), float(self.commands_limits[0][k]))
                hi[k] = min(float(r[1]), float(self.commands_limits[1][k]))
        saved = state.get("terrain")
        if self.terrain_curriculum and saved and saved.get("num_levels") == self.num_levels:
            for t, name in enumerate(self.ground.types):
                counts = np.asarray(saved["level_counts"].get(name, []), dtype=np.float64)
                mask = self.terrain_types == t
                if counts.sum() <= 0 or not bool(mask.any()):
                    continue
                draw = np.random.default_rng(0).choice(self.num_levels, size=int(mask.sum()), p=counts / counts.sum())
                self.terrain_levels[mask] = torch.as_tensor(draw, dtype=torch.long, device=gs.device)
            self.reset()  # respawn everyone on their restored level

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
        # [S2R] joint friction and damping: env_cfg values (go2.xml: 0.2 Nm, 0.1 Nm s/rad), or sampled per env and
        # joint when a range is given (the real Go2's values are not known; go2.xml's are themselves a model)
        J = self.num_actions
        self.joint_friction = torch.full((N, J), self.env_cfg.get("joint_friction", 0.0), dtype=gs.tc_float,
                                         device=gs.device)
        self.joint_damping = torch.full((N, J), self.env_cfg.get("joint_damping", 0.0), dtype=gs.tc_float,
                                        device=gs.device)
        friction = uniform(*dr["joint_friction_range"], (N, J)) if "joint_friction_range" in dr else self.joint_friction
        damping = uniform(*dr["joint_damping_range"], (N, J)) if "joint_damping_range" in dr else self.joint_damping
        if self._joint_losses_randomized() or self.joint_friction.any() or self.joint_damping.any():
            self.set_joint_losses(friction.clone(), damping.clone())
        if self.env_cfg.get("lower_leg_inertial"):  # [S2R]
            self.set_lower_leg_inertial(self.env_cfg["lower_leg_inertial"])

    def set_lower_leg_inertial(self, lower_leg):
        """[S2R] Give each calf link the mass and CoM that make calf + foot link equal lower_leg = {"mass": kg (calf
        with foot), "com": (x, y, z) in the calf frame}, e.g. GO2_MJCF_LOWER_LEG. The foot link keeps its 0.04 kg.
        Rotational inertia about the CoM is left as in go2.urdf (it is the small part: most of the difference is the
        mass sitting lower). Also used by go2_fsm_sim.py --mjcf-legs."""
        calves = [self.robot.get_link(f"{leg}_calf").idx_local for leg in ("FL", "FR", "RL", "RR")]
        m_leg, (x, y, z) = float(lower_leg["mass"]), lower_leg["com"]
        m_calf = m_leg - URDF_FOOT_MASS
        com = self.robot.get_links_COM(calves).clone()  # [4, 3], or [N, 4, 3] with batched link info
        com[..., 0] = (m_leg * x - URDF_FOOT_MASS * URDF_FOOT_POS[0]) / m_calf
        com[..., 1] = torch.sign(com[..., 1]) * abs(y) * m_leg / m_calf  # foot link at y = 0
        com[..., 2] = (m_leg * z - URDF_FOOT_MASS * URDF_FOOT_POS[2]) / m_calf
        self.robot.set_links_mass(torch.full(com.shape[:-1], m_calf, dtype=com.dtype, device=com.device), calves)
        self.robot.set_links_COM(com, calves)

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
        if self.on_terrain:  # [P3.1] above the env's spawn point, spawn_height above the ground there
            qpos[:, :2] = self.spawn_pos[:, :2]
            qpos[:, 2] = self.spawn_pos[:, 2] - self.init_base_pos[2] + self.recovery_spawn_height
        else:
            qpos[:, 2] = self.recovery_spawn_height
        qpos[:, 3:7] = quat
        qpos[:, self.motors_dof_idx + 1] = q  # qpos = [pos 3, quat 4, joints]: dof d is qpos entry d + 1
        # full [N, 19] tensor + boolean mask: written with torch.where (Genesis zero-copy, default on CPU and CUDA)
        self.robot.set_qpos(qpos, envs_idx=fallen, zero_velocity=True, skip_forward=not self.reset_forward)

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

    def _single_axis_commands(self, commands):
        """[S2R] Overwrite (vx, vy, wz) of a share of the envs, in place, with a command along ONE axis:
        - small_cmd_prob: magnitude in [deadzone, small_*_max], e.g. 0.2-0.4 m/s or 0.2-0.8 rad/s;
        - single_axis_prob: magnitude anywhere in that axis' training range.
        Uniform sampling of the 3-D box almost never gives such commands (pure vx needs vy = wz = 0 exactly), yet the
        deploy gate and a gamepad mostly send them: go2-p16 never learned -0.25 m/s and froze at +0.25 m/s with joint
        friction. Sign and axis are uniform. The body-height offset (4th command) is not touched. Off when both
        probabilities are 0 (configs without the keys)."""
        cc = self.command_cfg
        p_small, p_axis = cc.get("small_cmd_prob", 0.0), cc.get("single_axis_prob", 0.0)
        if p_small <= 0.0 and p_axis <= 0.0:
            return
        N = self.num_envs
        u = torch.rand((N,), dtype=gs.tc_float, device=gs.device)
        small, full = u < p_small, (u >= p_small) & (u < p_small + p_axis)
        lo, hi = self.cmd_limits[0][:3], self.cmd_limits[1][:3]  # [P3.1] current command-curriculum ranges
        mag_max = torch.maximum(lo.abs(), hi.abs())  # [3] full-range magnitude per axis
        dz = torch.tensor([cc.get("lin_deadzone", 0.0)] * 2 + [cc.get("ang_deadzone", 0.0)], dtype=gs.tc_float,
                          device=gs.device)
        small_max = torch.tensor([cc.get("small_lin_max", 0.4)] * 2 + [cc.get("small_ang_max", 0.8)],
                                 dtype=gs.tc_float, device=gs.device)
        top = torch.where(small[:, None], torch.minimum(small_max, mag_max), mag_max)  # [N, 3]
        mag = dz + (top - dz) * torch.rand((N, 3), dtype=gs.tc_float, device=gs.device)
        sign = torch.where(torch.rand((N, 3), device=gs.device) < 0.5, -1.0, 1.0)
        value = (sign * mag).clamp(lo, hi)  # stays inside the training range on asymmetric ranges
        axis = torch.randint(0, 3, (N,), device=gs.device)
        one_axis = torch.nn.functional.one_hot(axis, 3).to(gs.tc_float) * value  # [N, 3]
        commands[:, :3] = torch.where((small | full)[:, None], one_axis, commands[:, :3])

    def _resample_commands(self, envs_idx):
        # [P3.1] cmd_limits: the command curriculum's current ranges (= commands_limits without the curriculum)
        commands = gs_rand(*self.cmd_limits, (self.num_envs,))
        self._single_axis_commands(commands)  # [S2R]
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
        self.mech_power.zero_()  # [P2.5]
        for i in range(self.decimation):
            target = torch.where((i < self.latency_steps)[:, None], target_old, target_new)
            q = self.robot.get_dofs_position(self.motors_dof_idx)
            qd = self.robot.get_dofs_velocity(self.motors_dof_idx)
            torques = self.motor_strength * (self.kp * (target - q) - self.kd * qd)
            self.torques = torch.clip(torques, -self.torque_limits, self.torque_limits)  # URDF effort limits
            self.torques_substeps[:, i] = self.torques  # [P1.4]
            # [P2.5] motor power of this physics step: |tau * qdot| per joint, negative work not recovered. The joint
            # losses ([S2R] friction, damping) act outside the motor, so they show up here only through the torque the
            # PD needs to overcome them (as tau_est * dq on the robot, go2_analyze.py power[W])
            self.mech_power += torch.sum(torch.abs(self.torques * qd), dim=1)
            self.robot.control_dofs_force(self.torques[:, self.actions_dof_idx], self.motors_dof_slice)
            self.scene.step()
        self.mech_power /= self.decimation  # [P2.5] mean over the policy step [W]

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
        contact_force = self.robot.get_links_net_contact_force()  # [N, n_links, 3] world frame
        self.link_contact_force = torch.linalg.norm(contact_force, dim=-1)  # [N, n_links]
        self.feet_force = contact_force[:, self.feet_link_idx]  # [P3.2] [N, 4, 3]
        # [S2R] foot heights and speeds for the foot-clearance reward (all links read, then the feet kept, as
        # go2_metrics.py does: independent of the optional link-index arguments of the Genesis version)
        self.feet_pos = self.robot.get_links_pos()[:, self.feet_link_idx]  # [N, 4, 3]
        self.feet_vel = self.robot.get_links_vel()[:, self.feet_link_idx]  # [N, 4, 3]
        self.feet_ang = self.robot.get_links_ang()[:, self.feet_link_idx]  # [P3.2] [N, 4, 3]
        if self.on_terrain:  # [P3.1] ground under each foot, base height above their mean (flat: base z itself)
            self.feet_ground = self.ground.height(self.feet_pos[:, :, :2])
            self.base_height = self.base_pos[:, 2] - self.feet_ground.mean(dim=1)
            self.feet_normal = self.ground.normal_at(self.feet_pos[:, :, :2])  # [P3.2] face under each foot
        else:
            self.base_height = self.base_pos[:, 2]
        # [P3.2] stance slip per foot: contact-point speed along the ground, in contact at this step (unfiltered: the
        # filtered contact of _update_feet_contacts() also counts the first step of each swing, the fastest one)
        threshold = self.reward_cfg.get("contact_force_threshold", 1.0)
        in_contact = self.link_contact_force[:, self.feet_link_idx] > threshold
        self.foot_slip = contact_slip_speed(self.feet_vel, self.feet_ang, self.feet_normal, GO2_FOOT_RADIUS) * in_contact
        if self.terrain_curriculum:  # [P3.1] commanded distance and progress along the command, this episode
            cmd_xy = self.commands[:, :2]
            cmd_speed = torch.linalg.norm(cmd_xy, dim=1)
            self.cmd_dist += cmd_speed * self.dt
            along = torch.sum(self.base_lin_vel[:, :2] * cmd_xy, dim=1) / cmd_speed.clamp(min=1e-6)
            self.progress += along * self.dt
        self._update_feet_contacts()  # [P1.1]
        # [P1.1] "zero command" = all three components zero (the deadzone makes it exact)
        # [P1.6] velocities only: the body-height offset is not a motion command
        self.zero_cmd = torch.linalg.norm(self.commands[:, :3], dim=1) < 0.1
        # [P2.4] walk-these-ways order: the clock advances after the physics step, so the reward compares the contacts
        # just produced with the schedule at the new phase, and the next observation carries that phase
        if self.gait_clock:
            self.gait_phase = torch.remainder(self.gait_phase + self.dt * self.gait_frequency, 1.0)
            self._update_gait_targets()

        # compute reward
        self.rew_buf.zero_()
        # [fix] cost curriculum: negative-weight terms x k, k from penalty_scale_init to 1 over penalty_ramp_steps
        ramp = self.reward_cfg.get("penalty_ramp_steps", 0)
        k0 = self.reward_cfg.get("penalty_scale_init", 1.0)
        k = 1.0 if ramp <= 0 else min(1.0, k0 + (1.0 - k0) * self.common_step_counter / ramp)
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            if self.reward_scales[name] < 0:
                rew = rew * k
            self.rew_buf += rew
            self.episode_sums[name] += rew
        # [P1.3] legged_gym only_positive_rewards: with many penalties and contact termination, a negative total
        # would make ending the episode early attractive; clipping at 0 removes that incentive
        # [P1.5] ... except in recovery episodes, which cannot end early: their penalties must stay informative
        if self.reward_cfg.get("only_positive_rewards", False):
            self.rew_buf = torch.where(self.recovering, self.rew_buf, self.rew_buf.clip(min=0.0))

        if self.cmd_curriculum and self.common_step_counter % self.cmd_interval == 0:  # [P3.1]
            self._update_command_curriculum()
        self._resample_commands(self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        # [P3.1] per-env limits (same values as env_cfg unless go2_metrics.py raised them for a slope)
        fallen = torch.abs(self.base_euler[:, 1]) > self.max_pitch_deg
        fallen |= torch.abs(self.base_euler[:, 0]) > self.max_roll_deg
        # [P1.3] base contact termination (legged_gym check_termination: |F| > 1 N on terminate_after_contacts_on)
        if len(self.termination_contact_link_idx) > 0:
            contact_force = self.link_contact_force[:, self.termination_contact_link_idx]
            fallen |= torch.any(contact_force > self.reward_cfg.get("termination_contact_force", 1.0), dim=1)
        if self.ground.kind == "grid":  # [P3.1] walked off the edge of the grid's border
            fallen |= self.base_pos[:, 2] < self.ground_min - 1.0
        self.reset_buf |= fallen & ~self.recovering  # [P1.5] recovery episodes are not ended by a fall
        self.reset_buf |= self.scene.rigid_solver.get_error_envs_mask()

        # Compute timeout
        self.extras["time_outs"] = (self.episode_length_buf > self.max_episode_length).to(dtype=gs.tc_float)
        self.fell = self.reset_buf & (self.episode_length_buf <= self.max_episode_length)  # [P3.1] ended, not timed out

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
        # [P2.4] last complete swing (at touchdown) and stance (at lift-off), current contact time (Isaac Lab
        # ContactSensor: last_air_time, last_contact_time, current_contact_time)
        first_detach = (self.feet_contact_time > 0.0) & ~contact_filt
        self.feet_contact_time += self.dt
        torch.where(self.feet_first_contact, self.feet_air_time, self.feet_last_air_time, out=self.feet_last_air_time)
        torch.where(first_detach, self.feet_contact_time, self.feet_last_contact_time, out=self.feet_last_contact_time)
        self.feet_contact_time.masked_fill_(~contact_filt, 0.0)
        self.feet_air_time.masked_fill_(contact_filt, 0.0)
        self.feet_contact = contact_filt

    def _update_gait_targets(self):
        """[P2.4] Per-foot phase and desired contact of the walk-these-ways schedule at the current gait phase."""
        self.foot_phase, self.desired_contact = gait_targets(
            self.gait_phase, self.gait_offsets, self.gait_duty, self.gait_kappa
        )

    def clock_obs(self):
        """[P2.4] Gait-clock observation [N, 8]: sin and cos of each foot's warped phase, feet in foot_links order.
        go2_fsm.py gait_clock() computes the same values on the robot."""
        angle = 2.0 * math.pi * self.foot_phase
        return torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1)

    def get_observations(self):
        obs = {"policy": self.obs_buf}
        if self.privileged_layout:  # [P3.3] read by the networks whose obs_groups list it (go2_train.py --privileged)
            obs["privileged"] = self.privileged_buf
        return TensorDict(obs, batch_size=[self.num_envs])

    def _reset_idx(self, envs_idx=None):
        if self.recovery_init_prob > 0:  # [P1.5] success of the recovery episodes that end here (logged below)
            ended = self.recovering if envs_idx is None else self.recovering & envs_idx
            # [P3.1] base height above the ground (= base z on flat ground)
            upright = (self.projected_gravity[:, 2] < -0.9) & (self.base_height > 0.2)
            # [P1.6] fix: rsl_rl averages a logged value over every step of the iteration, and P1.5 logged 0 at the
            # steps where no recovery episode ended, so the curve stayed far below the true rate. Now the counts
            # decay by 0.99 per step (a window of about 100 steps) and their ratio is logged at every step.
            self.rec_ended_ema.mul_(0.99).add_(ended.sum())
            self.rec_success_ema.mul_(0.99).add_((ended & upright).sum())
            recovery_success = self.rec_success_ema / self.rec_ended_ema.clamp(min=1e-6)

        # [P3.1] the curricula judge the episodes that end here (before their state and episode sums are reset)
        self._update_terrain_curriculum(envs_idx)
        self._score_command_curriculum(envs_idx)

        # reset state
        if self.on_terrain:  # [P3.1] a new spawn pose, on the env's (possibly new) tile
            pos, quat = self._spawn_poses()
            if envs_idx is None:
                self.spawn_pos.copy_(pos)
                self.spawn_quat.copy_(quat)
            else:
                torch.where(envs_idx[:, None], pos, self.spawn_pos, out=self.spawn_pos)
                torch.where(envs_idx[:, None], quat, self.spawn_quat, out=self.spawn_quat)
            qpos = self.init_qpos.repeat(self.num_envs, 1)
            qpos[:, :3] = self.spawn_pos
            qpos[:, 3:7] = self.spawn_quat
            self.robot.set_qpos(qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=not self.reset_forward)
        else:
            self.robot.set_qpos(self.init_qpos, envs_idx=envs_idx, zero_velocity=True,
                                skip_forward=not self.reset_forward)

        # reset buffers
        if envs_idx is None:
            self.base_pos.copy_(self.spawn_pos)  # [P3.1] spawn pose (= the initial pose on flat ground)
            self.base_quat.copy_(self.spawn_quat)
            self.cmd_dist.zero_()  # [P3.1]
            self.progress.zero_()
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
            for timer in (self.feet_contact_time, self.feet_last_air_time, self.feet_last_contact_time):  # [P2.4]
                timer.zero_()
            self.gait_phase.zero_()  # [P2.4]
            self.mech_power.zero_()  # [P2.5]
            self.feet_force.zero_()  # [P3.2] the robot respawns in the air: no contact, no slip
            self.foot_slip.zero_()
            self.episode_length_buf.zero_()
            self.reset_buf.fill_(True)
        else:
            torch.where(envs_idx[:, None], self.spawn_pos, self.base_pos, out=self.base_pos)  # [P3.1] spawn pose
            torch.where(envs_idx[:, None], self.spawn_quat, self.base_quat, out=self.base_quat)
            self.cmd_dist.masked_fill_(envs_idx, 0.0)  # [P3.1]
            self.progress.masked_fill_(envs_idx, 0.0)
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
            for timer in (self.feet_contact_time, self.feet_last_air_time, self.feet_last_contact_time):  # [P2.4]
                timer.masked_fill_(envs_idx[:, None], 0.0)
            self.gait_phase.masked_fill_(envs_idx, 0.0)  # [P2.4]
            self.mech_power.masked_fill_(envs_idx, 0.0)  # [P2.5]
            self.feet_force.masked_fill_(envs_idx[:, None, None], 0.0)  # [P3.2] respawn in the air
            self.foot_slip.masked_fill_(envs_idx[:, None], 0.0)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)
        if self.gait_clock:  # [P2.4] schedule at phase 0 for the first observation of the new episode
            self._update_gait_targets()

        self._resample_actuation(envs_idx)  # [P1.2]
        if self.recovery_init_prob > 0:  # [P1.5]
            self._fallen_starts(envs_idx)

        # fill extras
        n_envs = envs_idx.sum() if envs_idx is not None else self.num_envs
        self.extras["episode"] = {}
        if self.recovery_init_prob > 0:  # [P1.5] share of recovery episodes ending upright, base above 0.2 m
            self.extras["episode"]["recovery_success"] = recovery_success
        self._curriculum_extras(self.extras["episode"])  # [P3.1]
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
        parts = [
            self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
            self.projected_gravity,  # 3
            self.commands * self.commands_scale,  # 3, or 4 with the body-height command [P1.6]
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
            self.dof_vel * self.obs_scales["dof_vel"],  # 12
            self.actions,  # 12
        ]
        if self.gait_clock:  # [P2.4] 8: appended at the end, so the first 45 (46) entries keep the P1 layout
            parts.append(self.clock_obs())
        self.obs_buf = torch.concatenate(parts, dim=-1)
        if self.add_noise:  # [P1.2] uniform noise, as legged_gym
            self.obs_buf += (2.0 * torch.rand_like(self.obs_buf) - 1.0) * self.noise_scale_vec
        if self.privileged_layout:  # [P3.3] after the noise draw: the random stream of "policy" is unchanged
            self.privileged_buf = self._privileged_obs()

    def _privileged_obs(self):
        """[P3.3] The privileged vector [N, num_privileged_obs]: the entries of obs_cfg["privileged"] in order, scaled
        as PRIVILEGED_SIZES says. At the first observation of an episode the velocities are 0 and the feet in the air
        (the reset state); the randomized parameters are those of the new episode."""
        parts = []
        for name, _ in self.privileged_layout:
            if name == "base_lin_vel":
                x = self.base_lin_vel * self.obs_scales["lin_vel"]
            elif name == "friction":
                x = ((self.friction - 0.8) / 0.4)[:, None]
            elif name == "added_mass":
                x = ((self.added_mass - 0.5) / 1.5)[:, None]
            elif name == "com_shift":
                x = self.com_shift / 0.03
            elif name == "motor_strength":
                x = (self.motor_strength - 1.0) / 0.1
            elif name == "foot_forces":  # base frame, so that the left-right mirror is a fixed permutation and sign
                inv_q = inv_quat(self.base_quat)[:, None, :].expand(-1, self.feet_force.shape[1], -1)
                x = (transform_by_quat(self.feet_force, inv_q) / 100.0).clip(-3.0, 3.0).flatten(1)
            elif name == "foot_slip":
                x = (self.foot_slip / 0.2).clip(max=5.0)
            elif name == "gains":
                x = (torch.cat((self.kp / self.kp_nominal, self.kd / self.kd_nominal), dim=1) - 1.0) / 0.15
            elif name == "latency":
                x = (self.latency_steps.to(gs.tc_float) / self.decimation)[:, None]
            elif name == "foot_clearance":  # read again: with reset_forward, the respawned feet (else the last step's)
                pos = self.robot.get_links_pos()[:, self.feet_link_idx]
                z = pos[..., 2] - self.ground.height(pos[..., :2]) - GO2_FOOT_RADIUS
                x = (z / 0.1).clip(-1.0, 3.0)
            else:  # "height_scan"
                x = self._height_scan()
            parts.append(x)
        return torch.cat(parts, dim=-1)

    def _height_scan(self):
        """[P3.3] legged_gym _get_heights() and compute_observations(): the ground height at the 17 x 11 scan points
        around the base (heading frame: rotated by the base yaw only), as clip(base z - 0.3 - h, -1, 1) x 5. On flat
        ground every point reads (base z - 0.3) x 5."""
        w, x, y, z = self.base_quat.unbind(-1)  # Genesis quaternions are (w, x, y, z)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
        px, py = self.height_scan_points[:, 0], self.height_scan_points[:, 1]  # [P]
        xy = torch.stack((self.base_pos[:, 0:1] + c * px - s * py, self.base_pos[:, 1:2] + s * px + c * py), dim=-1)
        h = self.ground.height(xy)  # [N, P]
        return (self.base_pos[:, 2:3] - HEIGHT_SCAN_OFFSET - h).clip(-1.0, 1.0) * 5.0

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
        # [P3.1] height above the ground under the feet (legged_gym: above the measured heights); base z on flat ground
        return torch.square(self.base_height - target)

    def _reward_feet_air_time(self):
        # [P1.1] legged_gym: on each touchdown, reward (air time - target), i.e. long steps; this also stops the
        # feet from shuffling. Paid only for a non-zero command, otherwise it would reward stepping in place.
        # (legged_gym gates on |cmd_xy| > 0.1; the yaw command counts here too, so turning in place is stepped.)
        # [P2.1] Summed over touchdowns this is (total airborne time) - target x (number of touchdowns): the target is
        # the price of one step, in seconds of swing (TEST_AND_TINKER_P2.md 5.1). Optional cap feet_air_time_max (Isaac
        # Lab Spot caps at mode_time): a foot held in the air earns at most (max - target) per touchdown, which removes
        # the "lame leg" incentive. None (default) = legged_gym, uncapped.
        air_time = self.feet_air_time_at_touchdown
        air_time_max = self.reward_cfg.get("feet_air_time_max")
        if air_time_max is not None:
            air_time = air_time.clip(max=air_time_max)
        rew = torch.sum((air_time - self.reward_cfg["feet_air_time_target"]) * self.feet_first_contact, dim=1)
        return rew * ~self.zero_cmd

    def _reward_foot_clearance(self):
        # [S2R] Isaac Lab Spot foot_clearance_reward: exp(-sum_feet (z - z*)^2 * tanh(k |v_xy|) / std). Only feet that
        # move horizontally count: swinging feet, and also SLIDING feet, so a foot dragged along the ground scores as
        # badly as a low swing. z* is the sphere-centre height for a clearance of feet_height_target (the quantity
        # swing_h measures). go2-p16 shuffled with 0.5-0.7 cm swings, which joint friction and torsional foot friction
        # (unitree_mujoco, real rubber feet) then stop.
        # [P3.1] clearance above the ground under each foot (was: above world z = 0, flat ground only)
        target = GO2_FOOT_RADIUS + self.reward_cfg["feet_height_target"]
        foot_z = self.feet_pos[:, :, 2] - self.feet_ground if self.on_terrain else self.feet_pos[:, :, 2]
        height_err = torch.square(foot_z - target)  # [N, 4]
        speed = torch.linalg.norm(self.feet_vel[:, :, :2], dim=-1)  # [N, 4]
        moving = torch.tanh(self.reward_cfg.get("foot_clearance_tanh_mult", 2.0) * speed)
        return torch.exp(-torch.sum(height_err * moving, dim=1) / self.reward_cfg.get("foot_clearance_std", 0.05))

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
        # [P3.1] above the ground (= base z on flat ground)
        height = (self.base_height / self.reward_cfg["base_height_target"]).clip(0.0, 1.0)
        return height * self._reward_rec_upright()

    def _reward_rec_pose(self):
        # close to the default pose, once upright
        pose = torch.exp(-torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1))
        return pose * self._reward_rec_upright()

    # [P2.4] Gait pattern, option A (go2_train.py --gait spot): Isaac Lab Spot GaitReward + air-time variance penalty
    def _gait_active(self):
        # as Spot: enforce the gait when a motion is commanded or the base moves (e.g. pushed) faster than a threshold
        base_speed = torch.linalg.norm(self.base_lin_vel[:, :2], dim=1)
        return ~self.zero_cmd | (base_speed > self.reward_cfg.get("gait_velocity_threshold", 0.5))

    def _reward_gait(self):
        # Spot GaitReward for a trot (synced pairs FL-RR and FR-RL). Synced feet: same current air and contact times.
        # Feet of different pairs: the air time of one equals the contact time of the other. Squared errors [s^2]
        # clipped at max_err^2, exp kernels of width std, product of the 6 terms -> 1 for a perfect 50 % duty trot.
        std = self.reward_cfg.get("gait_std", 0.1)
        max_sq = self.reward_cfg.get("gait_max_err", 0.2) ** 2
        air, con = self.feet_air_time, self.feet_contact_time  # current air / contact time [N, 4]

        def err(a, b):
            return torch.clip(torch.square(a - b), max=max_sq)

        def sync(i, j):
            return torch.exp(-(err(air[:, i], air[:, j]) + err(con[:, i], con[:, j])) / std)

        def anti(i, j):
            return torch.exp(-(err(air[:, i], con[:, j]) + err(con[:, i], air[:, j])) / std)

        (a0, a1), (b0, b1) = self.trot_pairs
        rew = sync(a0, a1) * sync(b0, b1) * anti(a0, b0) * anti(a1, b1) * anti(a0, b1) * anti(b0, a1)
        return rew * self._gait_active()

    def _reward_air_time_variance(self):
        # Spot air_time_variance_penalty (also in unitree_rl_lab's Go2 config, weight -1): variance over the 4 feet of
        # the last swing and last stance durations, clipped at 0.5 s. Gated like the gait reward (a standing robot's
        # timers are stale, the term would be a constant there).
        clip = self.reward_cfg.get("air_time_variance_clip", 0.5)
        variance = torch.var(self.feet_last_air_time.clip(max=clip), dim=1) + torch.var(
            self.feet_last_contact_time.clip(max=clip), dim=1
        )
        return variance * self._gait_active()

    # [P2.4] Gait pattern, option B (go2_train.py --gait wtw): walk-these-ways periodic contact schedule
    def _schedule(self):
        # desired contact of the clock; at zero command all four feet stand (WTW keeps stepping in place instead)
        return torch.where(self.zero_cmd[:, None], torch.ones_like(self.desired_contact), self.desired_contact)

    def _reward_contact_schedule_force(self):
        # WTW tracking_contacts_shaped_force, written as a penalty: contact force on a foot that should swing
        force = self.link_contact_force[:, self.feet_link_idx]
        sigma = self.reward_cfg.get("gait_force_sigma", 100.0)  # [N^2]
        return torch.sum((1.0 - self._schedule()) * (1.0 - torch.exp(-torch.square(force) / sigma)), dim=1) / 4.0

    def _reward_contact_schedule_vel(self):
        # WTW tracking_contacts_shaped_vel, written as a penalty: speed of a foot that should stand
        speed = torch.linalg.norm(self.feet_vel, dim=-1)
        sigma = self.reward_cfg.get("gait_vel_sigma", 10.0)  # [m^2/s^2]
        return torch.sum(self._schedule() * (1.0 - torch.exp(-torch.square(speed) / sigma)), dim=1) / 4.0

    # [P2.5] Energy (go2_train.py --energy)
    def _reward_energy(self):
        # mechanical power of the motors [W], sum over joints of |tau * qdot| (see step()); the same quantity as
        # power[W] in go2_metrics.py, whose CoT = power / (m g v)
        return self.mech_power

    # [P3.2] Foot slip (go2_train.py --foot-slip)
    def _reward_foot_slip(self):
        # Isaac Lab Spot foot_slip_penalty (spot_mdp/rewards.py; the velocity task's feet_slide computes the same): sum
        # over the feet in contact (|F| > contact_force_threshold, 1 N as Spot) of the planar world-frame speed of the
        # foot body [m/s]. Mode "xy" (default) is that term. On the Go2 the foot body is the sphere CENTRE, which moves
        # at R x omega even when the foot rolls without slipping (R = 2.3 cm: 4.6 cm/s for a calf turning at 2 rad/s in
        # stance), so "xy" also prices the leg's rotation over a planted foot, and on a slope it sees the slip x cos.
        # Mode "tangent" (reward_cfg["foot_slip_mode"]) sums self.foot_slip instead: the contact point's speed along the
        # ground, slip only (the harness's slip_v). go2_metrics.py reports both sums: pen_xy and pen_t [cm/s].
        if self.foot_slip_mode == "tangent":
            return torch.sum(self.foot_slip, dim=1)
        threshold = self.reward_cfg.get("contact_force_threshold", 1.0)
        in_contact = self.link_contact_force[:, self.feet_link_idx] > threshold
        return torch.sum(torch.linalg.norm(self.feet_vel[:, :, :2], dim=-1) * in_contact, dim=1)
