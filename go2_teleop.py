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
Genesis' own viewer keys keep working (A C D F H I L O P V W Z F11; I shows them in the window). Close the window or
press Ctrl+C in the terminal to stop.

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

import torch

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind
from rsl_rl.runners import OnPolicyRunner

from go2_commands import apply_command_deadzone
from go2_env import Go2Env, nominal_world

LIN_STEP, ANG_STEP, HEIGHT_STEP = 0.1, 0.1, 0.01  # change per key press [m/s, rad/s, m]
CHASE_OFFSET = (-1.6, -1.2, 0.9)  # [m] camera position relative to the robot (viewer and recording)
FOLLOW_SMOOTHING = 0.9  # camera low-pass per physics step: closer to 1 = smoother, more lag
NAMES = ("vx", "vy", "wz", "dh")


def key_help(height_cmd):
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
        "Close the window or press Ctrl+C here to stop.",
    ))


class KeyboardCommand:
    """Command setpoint changed by key presses (from the viewer thread), read by the control loop."""

    def __init__(self, command_cfg, initial=None):
        self.n = command_cfg["num_commands"]
        ranges = [command_cfg["lin_vel_x_range"], command_cfg["lin_vel_y_range"], command_cfg["ang_vel_range"]]
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


class TeleopGo2Env(Go2Env):
    """Go2Env with the command pinned to the keyboard setpoint, and an optional recording camera."""

    def __init__(self, teleop, *args, record_res=None, **kwargs):
        self.teleop = teleop  # before super().__init__(), which resets and resamples the command
        self.record_res = record_res
        self.cam = None
        super().__init__(*args, **kwargs)

    def _add_extra_entities(self):
        if self.record_res is not None:
            base = self.env_cfg["base_init_pos"]
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
    args = p.parse_args()
    if args.headless and args.duration is None:
        p.error("--headless needs --duration")

    log_dir = args.log_dir or os.path.join("logs", args.exp_name)
    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    if args.cmd is not None and len(args.cmd) not in (3, 4):
        p.error("--cmd takes VX VY WZ, or VX VY WZ DH for runs trained with --height-cmd")
    if args.cmd is not None and len(args.cmd) == 4 and command_cfg["num_commands"] < 4:
        p.error("--cmd: this run takes VX VY WZ (no DH: it was trained without --height-cmd)")

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, logging_level="warning")
    reward_cfg["reward_scales"] = {}
    if not args.dr:
        nominal_world(env_cfg, obs_cfg, latency_s=args.latency_ms / 1000.0)
    env_cfg["recovery"] = {}  # always start standing ([P1.5] fallen starts off, also with --dr)
    env_cfg.update(episode_length_s=1e6, viewer_camera_pos=CHASE_OFFSET,
                   viewer_camera_lookat=tuple(env_cfg["base_init_pos"]))
    teleop = KeyboardCommand(command_cfg, args.cmd)
    env = TeleopGo2Env(teleop, num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg,
                       command_cfg=command_cfg, show_viewer=not args.headless,
                       record_res=args.record_res if args.record else None)

    runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    viewer = None if args.headless else env.scene.viewer
    if viewer is not None:
        viewer.register_keybinds(*teleop.keybinds())
        viewer.follow_entity(env.robot, smoothing=FOLLOW_SMOOTHING)
        print(key_help(teleop.n >= 4))
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
