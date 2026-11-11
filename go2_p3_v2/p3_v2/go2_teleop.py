"""
go2_teleop.py - [P1.6] Drive a trained policy with the keyboard in the Genesis viewer, optionally recording a video.

    python go2_teleop.py -e go2-p16h --ckpt 999                    # click the viewer window, then use the keys
    python go2_teleop.py -e go2-p16h --ckpt 999 --dr               # with the training randomization and pushes
    python go2_teleop.py -e go2-p16h --ckpt 999 --cmd 0.3 0 0 -0.03   # start from this command
    python go2_teleop.py -e go2-p16h --ckpt 999 --record drive.mp4    # + video from a camera following the robot
    PYOPENGL_PLATFORM=egl python go2_teleop.py -e go2-p16h --ckpt 999 --headless --cmd 0.5 0 0 -0.04 \\
        --duration 10 --record crouch.mp4                          # no display: fixed command, video only

Keys (click the viewer window first; the list is also printed at start):
    Up / Down      vx +-0.1 m/s
    Left / Right   vy +-0.1 m/s (left = +)
    Q / E          wz +-0.1 rad/s (Q = turn left)
    T / G          dh +-1 cm (body height; runs trained with --height-cmd)
    Space          vx = vy = wz = 0 (keeps dh)
    Backspace      reset the robot (and the policy's memory)
    J / K          terrain level -1 / +1, U next terrain type ([P3.1] --terrain; each respawns at the tile centre)
Genesis' own viewer keys keep working (A C D F H I L O P R S V W Z F11; I shows them in the window). Close the window
or press Ctrl+C in the terminal to stop.

[P3.1] Terrain (TEST_AND_TINKER_P3.md, section 4.6):
    python go2_teleop.py -e go2-p31 --ckpt 2999 --terrain                  # the run's grid (logs/<exp>/terrain.npz)
    python go2_teleop.py -e go2-p31 --ckpt 2999 --terrain --tile pit 7     # start on this tile (J / K / U change it)
    python go2_teleop.py -e go2-p31 --ckpt 2999 --slope 20                 # plane at 20 deg (forward = uphill)
A run trained on flat ground gets the default grid (go2_terrain.TERRAIN_DEFAULTS), so P2 and P3 policies can be driven
over the same tiles. The robot always starts facing +x at the tile centre (hill: on top; pit: at the bottom). For a run
trained on flat ground, the tilt termination is raised by the ground's own steepest tilt (as go2_metrics.py does), so a
robot standing on a 25 deg face does not count as fallen.
For runs trained with the command curriculum (--cmd-curriculum), the keys stop at the ranges the curriculum had reached
in this checkpoint (what go2_export.py deploys); --configured_ranges goes up to the configured maximum instead.

The command is clamped to the run's training ranges, then the training deadzone applies, as on the robot
(go2_commands.py): the terminal shows both. It is pinned inside the env (never resampled, also after a fall), and the
observation is recomputed as soon as it changes, so the policy sees a key press at its next step.
Nominal world by default (no randomization, no noise, no pushes, 20 ms latency), no time-out resets, never a fallen
start (also with --dr and a run trained with --recovery). A fall resets the robot and the policy's memory; the fall
count is printed at the end.
"""

import argparse
import os
import pickle
import threading

import numpy as np
import torch

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind
from rsl_rl.runners import OnPolicyRunner

from go2_commands import apply_command_deadzone
from go2_env import Go2Env, nominal_world
from go2_terrain import make_ground  # [P3.1]

LIN_STEP, ANG_STEP, HEIGHT_STEP = 0.1, 0.1, 0.01  # change per key press [m/s, rad/s, m]
CHASE_OFFSET = (-1.6, -1.2, 0.9)  # [m] camera position relative to the robot (viewer and recording)
FOLLOW_SMOOTHING = 0.9  # camera low-pass per physics step: closer to 1 = smoother, more lag
NAMES = ("vx", "vy", "wz", "dh")


def key_help(height_cmd, tiles=False):
    """The key list printed at start (values from the constants above)."""
    t_g = f"dh +-{100 * HEIGHT_STEP:.0f} cm" if height_cmd else "unused (this run was trained without --height-cmd)"
    return "\n".join((
        "Keys (click the viewer window first):",
        f"    Up / Down      vx +-{LIN_STEP} m/s",
        f"    Left / Right   vy +-{LIN_STEP} m/s (left = +)",
        f"    Q / E          wz +-{ANG_STEP} rad/s (Q = turn left)",
        f"    T / G          {t_g}",
        "    Space          vx = vy = wz = 0 (keeps dh)",
        "    Backspace      reset the robot (and the policy's memory)",
        *(["    J / K          terrain level -1 / +1 (respawn)", "    U              next terrain type (respawn)"]
          if tiles else []),  # [P3.1]
        "Close the window or press Ctrl+C here to stop.",
    ))


class KeyboardCommand:
    """Command setpoint changed by key presses (from the viewer thread), read by the control loop."""

    def __init__(self, command_cfg, initial=None, reached=None):
        self.n = command_cfg["num_commands"]
        ranges = [command_cfg["lin_vel_x_range"], command_cfg["lin_vel_y_range"], command_cfg["ang_vel_range"]]
        for i, name in enumerate(("lin_vel_x", "lin_vel_y", "ang_vel")):  # [P3.1] the command curriculum's ranges
            if reached and name in reached:
                ranges[i] = reached[name]
        if self.n >= 4:
            ranges.append(command_cfg["body_height_range"])
        self.lo = [float(min(r)) for r in ranges]
        self.hi = [float(max(r)) for r in ranges]
        self.deadzones = (command_cfg.get("lin_deadzone", 0.0), command_cfg.get("ang_deadzone", 0.0))
        self.setpoint = [0.0] * self.n
        self.notes = []  # messages for the terminal (e.g. a key this run does not use)
        for i, v in enumerate((initial or [])[: self.n]):
            self.setpoint[i] = self._clamp(i, v)
            if self.setpoint[i] != round(v, 6):
                self.notes.append(f"--cmd {NAMES[i]} {v:g} is outside the training range: clamped to "
                                  f"{self.setpoint[i]:g}")
        self.lock = threading.Lock()
        self.changed = True  # the env takes the first command at start
        self.reset_requested = False

    def _clamp(self, i, v):
        return round(min(max(v, self.lo[i]), self.hi[i]), 6)

    def nudge(self, i, step):
        with self.lock:
            if i >= self.n:
                self.notes.append("T / G do nothing: this run was trained without --height-cmd")
                return
            value = self._clamp(i, self.setpoint[i] + step)
            if value == self.setpoint[i]:
                limit = f"{100 * value:+.0f} cm" if i == 3 else f"{value:+.2f}"
                self.notes.append(f"{NAMES[i]} is at its training limit ({limit})")
                return
            self.setpoint[i] = value
            self.changed = True

    def stop(self):
        with self.lock:
            self.setpoint[:3] = [0.0, 0.0, 0.0]
            self.changed = True

    def request_reset(self):
        with self.lock:
            self.reset_requested = True

    def take(self):
        """(command changed?, reset requested?, notes) since the last call; clears them."""
        with self.lock:
            out = (self.changed, self.reset_requested, self.notes)
            self.changed, self.reset_requested, self.notes = False, False, []
        return out

    def policy_command(self):
        """What the policy receives: the setpoint after the training deadzone."""
        with self.lock:
            setpoint = list(self.setpoint)
        return self._deadzone(setpoint)

    def _deadzone(self, setpoint):
        return apply_command_deadzone(torch.tensor(setpoint, dtype=torch.float32), *self.deadzones)

    def status(self):
        def fmt(values):
            parts = [f"{name} {v + 0.0:+.2f}" for name, v in zip(NAMES[:3], values[:3])]  # + 0.0: no "-0.00"
            if self.n >= 4:
                parts.append(f"dh {100 * values[3] + 0.0:+.0f} cm")
            return " ".join(parts)

        with self.lock:
            setpoint = list(self.setpoint)
        return f"set: {fmt(setpoint)}  -> policy gets: {fmt(self._deadzone(setpoint).tolist())}"

    def keybinds(self):
        press = KeyAction.PRESS
        return [
            Keybind("teleop_vx_up", Key.UP, press, callback=self.nudge, args=(0, LIN_STEP)),
            Keybind("teleop_vx_down", Key.DOWN, press, callback=self.nudge, args=(0, -LIN_STEP)),
            Keybind("teleop_vy_left", Key.LEFT, press, callback=self.nudge, args=(1, LIN_STEP)),
            Keybind("teleop_vy_right", Key.RIGHT, press, callback=self.nudge, args=(1, -LIN_STEP)),
            Keybind("teleop_turn_left", Key.Q, press, callback=self.nudge, args=(2, ANG_STEP)),
            Keybind("teleop_turn_right", Key.E, press, callback=self.nudge, args=(2, -ANG_STEP)),
            Keybind("teleop_height_up", Key.T, press, callback=self.nudge, args=(3, HEIGHT_STEP)),
            Keybind("teleop_height_down", Key.G, press, callback=self.nudge, args=(3, -HEIGHT_STEP)),
            Keybind("teleop_stop", Key.SPACE, press, callback=self.stop),
            Keybind("teleop_reset", Key.BACKSPACE, press, callback=self.request_reset),
        ]


class TileKeys:
    """[P3.1] --terrain: the tile the robot drives on, changed by J / K (level) and U (type) from the viewer thread."""

    def __init__(self, ground, kind, level):
        self.ground, self.kind, self.level = ground, kind, int(level)
        self.lock = threading.Lock()
        self.changed = False

    def step_level(self, delta):
        with self.lock:
            self.level = min(max(self.level + delta, 0), self.ground.num_levels - 1)
            self.changed = True

    def next_type(self):
        with self.lock:
            types = self.ground.types
            self.kind = types[(types.index(self.kind) + 1) % len(types)]
            self.changed = True

    def take(self):
        """The tile (kind, level) if a key changed it since the last call, else None."""
        with self.lock:
            out = (self.kind, self.level) if self.changed else None
            self.changed = False
        return out

    def describe(self, kind=None, level=None):
        kind, level = kind or self.kind, self.level if level is None else level
        col = self.ground.column_of_type(kind)
        return (f"tile {kind} level {level} of 0..{self.ground.num_levels - 1}: steepest slope "
                f"{self.ground.tile_slope[level, col]:.1f} deg, base tilt up to {self.ground.tile_tilt[level, col]:.1f}"
                f" deg")

    def keybinds(self):
        press = KeyAction.PRESS
        return [
            Keybind("teleop_level_down", Key.J, press, callback=self.step_level, args=(-1,)),
            Keybind("teleop_level_up", Key.K, press, callback=self.step_level, args=(1,)),
            Keybind("teleop_next_type", Key.U, press, callback=self.next_type),
        ]


def teleop_terrain(args, run_terrain, log_dir):
    """[P3.1] env_cfg["terrain"] for --terrain / --slope, the ground (for the start point and the tilt limit) and the
    start tile, or (None, None, None) on flat ground (the default, also with --dr). run_terrain: the run's own."""
    if args.slope is not None:
        cfg = {"mode": "slope", "slope_deg": args.slope, "random_yaw": False, "spawn_xy_noise": 0.0}
        return cfg, make_ground(cfg), None
    if not args.terrain:
        return None, None, None
    run = run_terrain
    cfg = dict(run) if run and run.get("mode", "grid") == "grid" else {"mode": "grid"}
    saved = os.path.join(log_dir, "terrain.npz")
    if run and os.path.isfile(saved):  # the exact heights the run trained on
        cfg["file"] = saved
    ground = make_ground(cfg)
    if args.tile is not None:
        kind, level = args.tile[0], int(args.tile[1])
        if kind not in ground.types:
            raise SystemExit(f"--tile {kind}: not in this grid ({ground.types})")
        if not 0 <= level < ground.num_levels:
            raise SystemExit(f"--tile: this grid has levels 0 .. {ground.num_levels - 1}")
    else:  # default: a hill at the middle level (a slope to walk down, then flat border or the next tile)
        kind = "hill" if "hill" in ground.types else ground.types[0]
        level = ground.num_levels // 2
    cfg.update(curriculum=False, spawn_xy_noise=0.0, random_yaw=False, spawn_tile=[kind, level])
    return cfg, ground, (kind, level)


class TeleopGo2Env(Go2Env):
    """Go2Env with the command pinned to the keyboard setpoint, and an optional recording camera."""

    def __init__(self, teleop, *args, record_res=None, start_pos=None, **kwargs):
        self.teleop = teleop  # before super().__init__(), which resets and resamples the command
        self.record_res = record_res
        self.start_pos = start_pos  # [P3.1] where the robot starts (camera placement), default base_init_pos
        self.cam = None
        super().__init__(*args, **kwargs)

    def set_tile(self, kind, level):
        """[P3.1] Respawn target for the next reset: tile (kind, level) of the grid."""
        col = self.ground.column_of_type(kind)
        self.terrain_cols[:] = col
        self.terrain_levels[:] = int(level)
        self.terrain_types[:] = int(self.col_type[col])

    def _add_extra_entities(self):
        if self.record_res is not None:
            base = self.start_pos or self.env_cfg["base_init_pos"]
            self.cam = self.scene.add_camera(
                res=tuple(self.record_res),
                pos=tuple(b + o for b, o in zip(base, CHASE_OFFSET)),
                lookat=tuple(base),
                fov=40,
                GUI=False,
            )

    def _resample_commands(self, envs_idx):
        self.commands[:] = self.teleop.policy_command().to(self.device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    p.add_argument("--ckpt", type=int, required=True, help="checkpoint (last of a 1000-iteration run: 999)")
    p.add_argument("--log_dir", type=str, default=None, help="default: logs/<exp_name>")
    p.add_argument("--cmd", type=float, nargs="+", metavar="V", default=None, help="initial VX VY WZ [DH]")
    p.add_argument("--dr", action="store_true", help="keep the training randomization, noise and pushes")
    p.add_argument("--latency_ms", type=float, default=20.0, help="action latency in the nominal world")
    p.add_argument("--record", type=str, default=None, help="video file (.mp4) from a camera following the robot")
    p.add_argument("--record_res", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"))
    p.add_argument("--headless", action="store_true", help="no viewer, no keys: the --cmd command for --duration s")
    p.add_argument("--duration", type=float, default=None, help="stop after this many seconds of simulated time")
    p.add_argument("--gpu", action="store_true", help="GPU backend (default: CPU, enough for one robot)")
    p.add_argument("--terrain", action="store_true",
                   help="[P3.1] drive on the run's terrain grid (runs trained on flat ground: the default grid)")
    p.add_argument("--tile", nargs=2, default=None, metavar=("TYPE", "LEVEL"),
                   help="[P3.1] --terrain: start tile, e.g. --tile pit 7 (default: hill at the middle level)")
    p.add_argument("--slope", type=float, default=None, metavar="DEG",
                   help="[P3.1] a plane tilted by DEG, rising along +x (forward = uphill; negative: downhill)")
    p.add_argument("--configured_ranges", action="store_true",
                   help="[P3.1] keys go up to the configured command ranges, not those the command curriculum reached")
    args = p.parse_args()
    if args.headless and args.duration is None:
        p.error("--headless needs --duration")
    if args.terrain and args.slope is not None:
        p.error("--terrain and --slope exclude each other")
    if args.tile is not None and not args.terrain:
        p.error("--tile needs --terrain")

    log_dir = args.log_dir or os.path.join("logs", args.exp_name)
    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    if args.cmd is not None and len(args.cmd) not in (3, 4):
        p.error("--cmd takes VX VY WZ, or VX VY WZ DH for runs trained with --height-cmd")
    if args.cmd is not None and len(args.cmd) == 4 and command_cfg["num_commands"] < 4:
        p.error("--cmd: this run takes VX VY WZ (no DH: it was trained without --height-cmd)")

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, logging_level="warning")
    ckpt_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    # [P3.1] the curriculum state saved with the checkpoint (runs from before P3.1: none)
    infos = torch.load(ckpt_path, map_location="cpu", weights_only=False).get("infos") or {}
    curriculum = infos.get("curriculum") or {}
    run_terrain = env_cfg.get("terrain")  # before nominal_world(), which sets flat ground
    reward_cfg["reward_scales"] = {}
    if not args.dr:
        nominal_world(env_cfg, obs_cfg, latency_s=args.latency_ms / 1000.0)  # also flat ground
    env_cfg["recovery"] = {}  # always start standing ([P1.5] fallen starts off, also with --dr)
    # [P3.1] ground: flat (default, also with --dr), the run's grid (--terrain) or a tilted plane (--slope)
    env_cfg["terrain"], ground, tile = teleop_terrain(args, run_terrain, log_dir)
    start = list(env_cfg["base_init_pos"])
    if ground is not None:
        tilt = abs(args.slope) if args.slope is not None else float(ground.tile_tilt.max())
        if not run_terrain:  # upright on a 25 deg face is a 25 deg pitch: not a fall
            for key in ("termination_if_roll_greater_than", "termination_if_pitch_greater_than"):
                env_cfg[key] = float(env_cfg[key]) + tilt
        if tile is not None:
            x, y, z = ground.tile_origins[tile[1], ground.column_of_type(tile[0])]
        else:
            x, y, z = 0.0, 0.0, float(ground.height_np(np.zeros(2)))
        start = [float(x), float(y), float(z) + float(env_cfg["base_init_pos"][2])]
    env_cfg.update(episode_length_s=1e6, viewer_camera_pos=tuple(s + o for s, o in zip(start, CHASE_OFFSET)),
                   viewer_camera_lookat=tuple(start))
    reached = None
    if curriculum.get("command_curriculum") and not args.configured_ranges:
        reached = curriculum.get("command_ranges")
        print(f"command curriculum: the keys stop at the ranges reached in this checkpoint {reached} "
              f"(--configured_ranges: up to the configured maximum)")
    teleop = KeyboardCommand(command_cfg, args.cmd, reached=reached)
    tiles = TileKeys(ground, *tile) if tile is not None else None
    env = TeleopGo2Env(teleop, num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg,
                       command_cfg=command_cfg, show_viewer=not args.headless,
                       record_res=args.record_res if args.record else None, start_pos=start)
    if ground is not None:  # [P3.1]
        print(f"ground: {env.ground.describe()}" + (f"; {tiles.describe()}" if tiles else "")
              + f"; tilt termination {env_cfg['termination_if_pitch_greater_than']:g} deg")

    runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=gs.device)
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=gs.device)

    viewer = None if args.headless else env.scene.viewer
    if viewer is not None:
        viewer.register_keybinds(*teleop.keybinds(), *(tiles.keybinds() if tiles else []))
        viewer.follow_entity(env.robot, smoothing=FOLLOW_SMOOTHING)
        print(key_help(teleop.n >= 4, tiles=tiles is not None))
    if env.cam is not None:
        env.cam.follow_entity(env.robot, smoothing=FOLLOW_SMOOTHING)
        env.cam.start_recording(save_to_filename=args.record, fps=round(1.0 / env.dt))  # one frame per policy step

    falls, steps = 0, 0
    max_steps = None if args.duration is None else round(args.duration / env.dt)
    obs = env.reset()
    try:
        with torch.no_grad():
            while max_steps is None or steps < max_steps:
                if viewer is not None and not viewer.is_alive():
                    break
                changed, reset, notes = teleop.take()
                for note in dict.fromkeys(notes):
                    print(note)
                new_tile = tiles.take() if tiles else None  # [P3.1] J / K / U
                if new_tile is not None:
                    env.set_tile(*new_tile)
                    print(tiles.describe(*new_tile))
                    reset = True
                if reset:
                    env.reset()  # pose, buffers; the command stays pinned
                    policy.reset()
                if changed or reset:
                    env.commands[:] = teleop.policy_command().to(env.device)
                    env._update_observation()  # the next action already uses the new command
                    obs = env.get_observations()
                    print(teleop.status() + ("  (robot reset)" if reset else ""))
                actions = policy(obs)
                obs, _, dones, extras = env.step(actions)
                policy.reset(dones)  # clear the memory of a reset env (recurrent policy; no-op for an MLP)
                falls += int((dones & (extras["time_outs"] == 0)).sum())
                steps += 1
    except KeyboardInterrupt:
        pass
    except Exception:
        # closing the viewer window stops the scene with an exception ("Viewer closed."): that is a normal way out
        if viewer is None or viewer.is_alive():
            raise
    finally:
        if env.cam is not None:
            env.cam.stop_recording()  # closes the video file
            print(f"video saved to {args.record}")
        print(f"{steps * env.dt:.1f} s simulated, {falls} fall(s)")


if __name__ == "__main__":
    main()
