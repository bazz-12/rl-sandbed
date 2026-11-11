"""
go2_log.py - [S2R] One log format for the three worlds, so go2_analyze.py can put them side by side:
    robot         go2_deploy_sdk2.py --net <if>
    unitree_mujoco  go2_deploy_sdk2.py --sim  (+ go2_mujoco_headless.py --truth for the simulator's ground truth)
    Genesis       go2_fsm_sim.py --log
numpy only.

A log is one .npz file:
    meta           JSON: source ("robot" | "mujoco" | "genesis"), deploy_cfg, state machine config, args, profile, joint
                   test plan, events [[t, message], ...], t0_mono (time.monotonic() at t = 0, to align two files of the
                   same run, e.g. the deploy log and the headless simulator's ground truth)
    ctrl.<field>   one row per control step: 500 Hz on the robot and in unitree_mujoco, 200 Hz in the Genesis bench
    pol.<field>    one row per policy step (50 Hz)
    truth.<field>  simulators only: ground truth at the physics rate
Everything is in POLICY joint order (deploy_cfg["joint_names"]); time t in seconds since the start of the run.
Streams are ring buffers: a run longer than max_s keeps its last max_s seconds.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import time

import numpy as np

FORMAT = "go2log/1"

# ctrl.flags bits
FLAG_CLAMPED = 1  # a position target was clipped to the joint range
FLAG_PROFILE = 2  # a command profile was running
FLAG_SCRIPTED = 4  # buttons came from a scenario or the keyboard
FLAG_STALE = 8  # the robot state was older than the timeout
FLAG_REPEAT = 16  # same robot state as the previous control step (tick unchanged)


def ctrl_fields(n=12):
    """One row per control step. tau_cmd = kp (q_des - q) + kd (dq_des - dq) + tau_ff at the logged state: the torque
    the command asks of the motors (before their own clipping); tau_est = what the robot reports (Genesis: applied)."""
    j = (n,)
    return {
        "t": ((), "f8"), "mode": ((), "i1"),
        "q": (j, "f4"), "dq": (j, "f4"), "tau_est": (j, "f4"),
        "quat": ((4,), "f4"), "gyro": ((3,), "f4"), "acc": ((3,), "f4"), "foot_force": ((4,), "f4"),
        "q_des": (j, "f4"), "dq_des": (j, "f4"), "kp": (j, "f4"), "kd": (j, "f4"), "tau_ff": (j, "f4"),
        "tau_cmd": (j, "f4"),
        "tick": ((), "u4"), "state_age": ((), "f4"), "loop_dt": ((), "f4"), "compute": ((), "f4"),
        "buttons": ((), "u2"), "sticks": ((4,), "f4"), "flags": ((), "u1"),
    }


def pol_fields(n_obs, n_cmd, n=12):
    """One row per policy step: the observation the network got, its output (clipped as in training), the command,
    the inference time and the profile segment (-1: none)."""
    return {"t": ((), "f8"), "obs": ((n_obs,), "f4"), "action": ((n,), "f4"), "cmd": ((n_cmd,), "f4"),
            "infer": ((), "f4"), "seg": ((), "i2")}


def truth_fields(n=12):
    """Simulator ground truth: base position (world), orientation (w, x, y, z), linear velocity (world), angular
    velocity (base frame), joints, applied torques, and per foot the contact force (world) and the position of the
    foot-sphere centre (world)."""
    j = (n,)
    return {"t": ((), "f8"), "pos": ((3,), "f4"), "quat": ((4,), "f4"), "lin_vel": ((3,), "f4"),
            "ang_vel": ((3,), "f4"), "q": (j, "f4"), "dq": (j, "f4"), "tau": (j, "f4"),
            "foot_force": ((4, 3), "f4"), "foot_pos": ((4, 3), "f4")}


class Stream:
    """Preallocated ring buffer of rows; append() is a handful of array writes (fine at 500 Hz)."""

    def __init__(self, fields, rows):
        self.fields = fields
        self.rows = int(rows)
        self.data = {k: np.zeros((self.rows, *shape), dtype) for k, (shape, dtype) in fields.items()}
        self.n = 0  # rows appended so far (may exceed self.rows)

    def append(self, **values):
        i = self.n % self.rows
        for k, arr in self.data.items():
            arr[i] = values.get(k, 0)
        self.n += 1

    def arrays(self):
        """The kept rows, oldest first. Safe while another thread appends: every field gets the same rows (n is read
        once; a row counts once all its fields are written), and once the buffer has wrapped, the oldest row, which
        the next append overwrites, is left out."""
        n = self.n
        if n <= self.rows:
            return {k: v[:n] for k, v in self.data.items()}
        i = n % self.rows
        return {k: np.concatenate((v[i + 1:], v[:i])) for k, v in self.data.items()}

    @property
    def dropped(self):
        return max(0, self.n - self.rows)


class Recorder:
    """Collects the streams of one run and writes the .npz file."""

    def __init__(self, path, source, meta=None, max_s=600.0, ctrl_hz=500.0, pol_hz=50.0, truth_hz=None, n_obs=None,
                 n_cmd=None, t0_mono=None):
        self.path = path
        self.t0_mono = time.monotonic() if t0_mono is None else t0_mono
        self.meta = {
            "format": FORMAT, "source": source, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "t0_mono": self.t0_mono, "t0_wall": time.time() - (time.monotonic() - self.t0_mono),
            "host": socket.gethostname(), "python": platform.python_version(), "max_s": max_s,
        }
        self.meta.update(meta or {})
        self.events = []
        self.closed = False
        self.streams = {"ctrl": Stream(ctrl_fields(), max_s * ctrl_hz * 1.05 + 10)}
        if n_obs is not None:
            self.streams["pol"] = Stream(pol_fields(n_obs, n_cmd), max_s * pol_hz * 1.05 + 10)
        if truth_hz is not None:
            self.streams["truth"] = Stream(truth_fields(), max_s * truth_hz * 1.05 + 10)

    def append(self, stream, **values):
        if not self.closed:
            self.streams[stream].append(**values)

    def close(self):
        """Stop recording: later appends are ignored. Call it before save() when other threads may still append
        (go2_deploy_sdk2.py keeps sending damping commands while it saves)."""
        self.closed = True
        time.sleep(0.01)  # an append already running in another thread finishes

    def event(self, t, msg):
        self.events.append([float(t), str(msg)])

    def save(self, path=None, extra_meta=None):
        path = path or self.path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        meta = dict(self.meta, events=self.events, **(extra_meta or {}))
        data = {name: s.arrays() for name, s in self.streams.items()}
        meta["rows"] = {name: int(len(d["t"])) for name, d in data.items()}
        meta["dropped_rows"] = {name: int(s.dropped) for name, s in self.streams.items()}
        arrays = {"meta": np.array(json.dumps(meta, default=_json_default))}
        for name, d in data.items():
            for k, v in d.items():
                arrays[f"{name}.{k}"] = v
        tmp = path + ".tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
        return path


def _json_default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return str(x)


class Log:
    """A loaded log: .meta (dict), .ctrl / .pol / .truth (dicts of arrays, empty when absent), .streams (every stream,
    e.g. a probe log's state / loop / sport), .events, .source."""

    def __init__(self, path):
        self.path = path
        self.streams = {}
        with np.load(path, allow_pickle=False) as z:
            self.meta = json.loads(str(z["meta"]))
            for key in z.files:
                if "." in key:
                    stream, field = key.split(".", 1)
                    self.streams.setdefault(stream, {})[field] = z[key]
        self.ctrl, self.pol, self.truth = (self.streams.get(k, {}) for k in ("ctrl", "pol", "truth"))
        self.events = self.meta.get("events", [])
        self.source = self.meta.get("source", "?")
        self.name = os.path.splitext(os.path.basename(path))[0]

    def __repr__(self):
        return f"Log({self.name}, {self.source}, {len(self.ctrl.get('t', []))} ctrl rows)"


def load(path):
    return Log(path)


def default_path(log_dir, source, tag=""):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(log_dir, f"{source}_{stamp}" + (f"_{tag}" if tag else "") + ".npz")
