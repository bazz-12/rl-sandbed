"""
go2_analyze.py - [S2R] Read go2_log.py logs of the three worlds (robot, unitree_mujoco, Genesis bench) and put them side
by side. numpy + matplotlib only (no Genesis, no SDK): runs anywhere.

    python go2_analyze.py RUN.npz                       # one run: phases, events, timing, safety, metrics per phase
    python go2_analyze.py A.npz B.npz C.npz             # side by side, A = reference; phases matched by name
    python go2_analyze.py mujoco_run.npz mj_truth.npz   # a go2_mujoco_headless.py --truth file joins its run
    python go2_analyze.py ... --plots out/              # figures: overview per run, overlay, joint test
    python go2_analyze.py ... --csv out/                # go2_metrics.py-style CSV per run: go2_compare.py reads them
    python go2_analyze.py logs/probe/probe_*.npz        # re-print a go2_probe.py report

Phases (matched across runs by name):
    stand            policy mode, zero command, outside any profile
    profile seg k    segment k of the command profile (same --profile in each run)
    joint test       the go2_inputs.JointTest plan (steps and chirp), robot hung
Each phase skips its first SETTLE_S = 1.5 s (the transient after a command change).

The same estimators run on every world, so their bias cancels in a comparison:
    velocity    leg odometry (go2_kinematics.py): stance feet assumed fixed; "vx_odo", "lin_err_odo"
    contact     vertical foot force from the joint torques, -J^-T tau, with hysteresis
    height      base above the stance feet
When a run has ground truth (Genesis bench, unitree_mujoco through rt/sportmodestate or the headless --truth file),
"vx_meas", "lin_err", "z" use it (as go2_metrics.py does); on the robot they fall back to the estimates.
"""

import argparse
import math
import os

import numpy as np

import go2_kinematics as K
import go2_log
from go2_inputs import SETTLE_S, joint_kind

DEFAULT_MODES = ["passive", "stand_up", "policy", "sit_down", "joint_test"]
GRID_HZ = 200.0  # common rate for torque spectra and contact estimates (Genesis physics rate, go2_metrics.py)
HF_HZ = 10.0  # go2_metrics.py HF_CUTOFF_HZ
MIN_RUN = int(round(0.04 * GRID_HZ))  # stance and swing shorter than 40 ms are contact chatter, not steps
G = 9.81
IMU_OFFSET = np.array([-0.02557, 0.0, 0.04232])  # IMU site in the base frame (go2.xml / go2.urdf)
# H3 gate thresholds (unitree_mujoco through the deploy script): edit to taste
GATE = {"lin_err": 0.15, "yaw_err": 0.2, "tau_peak": 95.0, "tilt_max": 25.0, "stand_drift": 0.10}


# ======================================================================================================================
# Runs
# ======================================================================================================================
class Window:
    def __init__(self, name, kind, t0, t1, cmd=None, seg=-1):
        self.name, self.kind, self.t0, self.t1, self.cmd, self.seg = name, kind, t0, t1, cmd, seg

    def label(self):
        if self.cmd is None:
            return self.name
        c = self.cmd
        s = f"vx {c[0]:+.2f} vy {c[1]:+.2f} wz {c[2]:+.2f}" + (f" dh {100 * c[3]:+.0f} cm" if len(c) > 3 else "")
        return f"{self.name}: {s} ({self.t1 - self.t0:.1f} s)"


class Run:
    """One deploy-format log, with the derived signals on a uniform GRID_HZ grid."""

    def __init__(self, log, letter):
        self.log, self.letter = log, letter
        self.name, self.source, self.meta = log.name, log.source, log.meta
        self.cfg = self.meta["deploy_cfg"]
        self.fsm_cfg = self.meta.get("fsm_cfg", {})
        self.modes = self.meta.get("modes", DEFAULT_MODES)
        self.c, self.p = log.ctrl, log.pol
        self.default_q = np.asarray(self.cfg["default_joint_angles"], dtype=float)
        self.torque_limits = np.asarray(self.cfg["torque_limits"], dtype=float)
        lim = np.asarray(self.cfg["dof_pos_limits"], dtype=float)
        self.q_lo, self.q_hi = lim[:, 0], lim[:, 1]
        self.kinds = [joint_kind(n) for n in self.cfg["joint_names"]]
        self.z_target = float(self.cfg.get("base_height_target", 0.30))
        self.truth = None
        if len(log.truth.get("t", [])):
            self.set_truth(log.truth, self.meta.get("truth_kind") or "full", 0.0)
        self._grid = None

    def set_truth(self, tr, kind, offset):
        """Ground truth on this run's clock (offset = truth t0 - run t0). kind "full": base origin pose and velocity
        (+ foot forces); "sportmodestate": IMU-site position and world velocity from unitree_mujoco."""
        t = tr["t"] + offset
        keep = np.r_[True, np.diff(t) > 0]
        self.truth = {"kind": kind, "t": t[keep], "pos": tr["pos"][keep].astype(float),
                      "lin_vel": tr["lin_vel"][keep].astype(float)}
        if kind == "full":
            self.truth["quat"] = tr["quat"][keep].astype(float)
            ff = tr.get("foot_force")
            if ff is not None and np.any(ff != 0):
                self.truth["foot_force"] = ff[keep].astype(float)
        self._grid = None

    def mode_index(self, name):
        return self.modes.index(name) if name in self.modes else -1

    # --- uniform grid -------------------------------------------------------------------------------------------
    def grid(self):
        """Signals at GRID_HZ from the distinct robot states (a state reused by several control steps counts once,
        at its arrival time), plus the estimates: stance, leg-odometry velocity, height, foot heights."""
        if self._grid is not None:
            return self._grid
        c = self.c
        fresh = (c["flags"] & go2_log.FLAG_REPEAT) == 0
        fresh[0] = True
        if fresh.sum() < 10:  # no usable repeat flags: take every row
            fresh[:] = True
        ts = (c["t"] - c["state_age"])[fresh]
        keep = np.r_[True, np.diff(ts) > 1e-6]
        idx = np.flatnonzero(fresh)[keep]
        ts = ts[keep]
        tg = np.arange(ts[0], ts[-1], 1.0 / GRID_HZ)
        g = {"t": tg}

        def interp(x):
            x = np.asarray(x, dtype=float)
            return np.stack([np.interp(tg, ts, x[:, k]) for k in range(x.shape[1])], 1)

        for k in ("q", "dq", "tau_est", "gyro", "quat", "acc"):
            g[k] = interp(c[k][idx])
        g["quat"] /= np.linalg.norm(g["quat"], axis=1, keepdims=True)
        near = np.clip(np.searchsorted(c["t"], tg), 0, len(c["t"]) - 1)
        g["mode"] = c["mode"][near]
        g["q_des"] = c["q_des"][near].astype(float)
        g["tau_cmd"] = c["tau_cmd"][near].astype(float)
        F = K.foot_forces(g["q"], g["tau_est"])
        g["fz"] = F[..., 2]
        g["stance"] = K.debounce(K.contacts(g["fz"]), MIN_RUN)
        v = K.leg_odometry(g["q"], g["dq"], g["gyro"], g["stance"])
        g["v_odo"] = moving_average(v, int(0.2 * GRID_HZ))
        g["z_est"] = moving_average(K.base_height(g["q"], g["quat"], g["stance"]), int(0.1 * GRID_HZ))
        g["foot_h"] = K.foot_heights(g["q"], g["quat"], g["stance"])
        g["gyro_f"] = moving_average(g["gyro"], int(0.02 * GRID_HZ))  # one policy step
        if self.truth is not None:
            tr = self.truth
            inside = (tg >= tr["t"][0]) & (tg <= tr["t"][-1])

            def tinterp(x):
                x = np.asarray(x, dtype=float)
                out = np.stack([np.interp(tg, tr["t"], x[:, k]) for k in range(x.shape[1])], 1)
                out[~inside] = np.nan
                return out

            pos, vel = tinterp(tr["pos"]), tinterp(tr["lin_vel"])
            if tr["kind"] == "sportmodestate":  # IMU site -> base origin, with the IMU's own quaternion and gyro
                pos = pos - K.rotate(g["quat"], np.broadcast_to(IMU_OFFSET, pos.shape))
                vel = vel - K.rotate(g["quat"], np.cross(g["gyro"], IMU_OFFSET))
                quat = g["quat"]
            else:
                quat = tinterp(tr["quat"])
                quat /= np.linalg.norm(np.nan_to_num(quat), axis=1, keepdims=True) + 1e-12
            g["z_true"] = pos[:, 2]
            g["pos_true"] = pos
            g["v_true"] = K.rotate_inv(np.nan_to_num(quat, nan=1.0), np.nan_to_num(vel))
            g["v_true"][~inside] = np.nan
            if "foot_force" in tr:
                ff = np.stack([tinterp(tr["foot_force"][:, k]) for k in range(4)], 1)
                g["stance_true"] = K.debounce(K.contacts(np.nan_to_num(ff[..., 2])), MIN_RUN)  # as the estimate
        self._grid = g
        return g

    # --- phases -------------------------------------------------------------------------------------------------
    def windows(self):
        out = []
        p = self.p
        if len(p.get("t", [])):
            pt, seg, cmd = p["t"], p["seg"], p["cmd"].astype(float)
            dt = float(self.cfg["policy_dt"])
            brk = np.flatnonzero((np.diff(seg) != 0) | np.any(np.abs(np.diff(cmd, axis=0)) > 1e-6, axis=1)
                                 | (np.diff(pt) > 0.2)) + 1  # 0.2 s: policy mode was left (a stall is shorter)
            names = {}
            for a, b in zip(np.r_[0, brk], np.r_[brk, len(pt)]):
                t0, t1 = pt[a], pt[b - 1] + dt
                if t1 - t0 < SETTLE_S + 1.0:
                    continue
                c0, s0 = cmd[a], int(seg[a])
                if s0 >= 0:
                    name, kind = f"profile seg {s0}", "segment"
                elif not np.any(c0[:3]):
                    name, kind = "stand", "stand"
                else:
                    continue  # stick driving: the command keeps changing, nothing to measure
                names[name] = names.get(name, 0) + 1
                if names[name] > 1:
                    name += f" #{names[name]}"
                out.append(Window(name, kind, t0 + SETTLE_S, t1, c0, s0))
        jt = self.mode_index("joint_test")
        if jt >= 0:
            starts, ends = runs_of(self.c["mode"] == jt)
            for k, (a, b) in enumerate(zip(starts, ends)):
                out.append(Window("joint test" + (f" #{k + 1}" if k else ""), "joint_test", self.c["t"][a],
                                  self.c["t"][b - 1]))
        return out

    def mode_spans(self):
        m, t = self.c["mode"], self.c["t"]
        cuts = np.r_[0, np.flatnonzero(np.diff(m) != 0) + 1, len(m)]
        return [(t[a], t[b - 1], self.modes[m[a]] if m[a] < len(self.modes) else str(m[a]))
                for a, b in zip(cuts[:-1], cuts[1:])]


def moving_average(x, n):
    """Centred moving average along axis 0, NaN-aware (NaN where the whole window is NaN)."""
    x = np.asarray(x, dtype=float)
    if n <= 1:
        return x
    shape = x.shape
    x2 = x.reshape(len(x), -1)
    valid = np.isfinite(x2)
    xs = np.where(valid, x2, 0.0)
    k = np.ones(n)
    num = np.stack([np.convolve(xs[:, j], k, mode="same") for j in range(x2.shape[1])], 1)
    den = np.stack([np.convolve(valid[:, j].astype(float), k, mode="same") for j in range(x2.shape[1])], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (num / den).reshape(shape)


def quat_rp(quat):
    """Roll, pitch [deg] (ZYX) from (w, x, y, z) rows."""
    w, x, y, z = quat.T
    roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
    return roll, pitch


def psd(x, fs):
    """One-sided PSD along axis 0 (Hann window, mean removed), scaled so that sum(psd) df = mean square
    (go2_metrics.py psd())."""
    T = x.shape[0]
    x = x - x.mean(0, keepdims=True)
    w = np.hanning(T).reshape(-1, *([1] * (x.ndim - 1)))
    p = np.abs(np.fft.rfft(x * w, axis=0)) ** 2 / (fs * np.sum(w ** 2))
    p[1:] *= 2.0
    if T % 2 == 0:
        p[-1] /= 2.0
    return np.fft.rfftfreq(T, d=1.0 / fs), p


def runs_of(mask):
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


# ======================================================================================================================
# Metrics of one policy phase (names as go2_metrics.py where the meaning is the same)
# ======================================================================================================================
def phase_metrics(run, w):
    g = run.grid()
    sel = (g["t"] >= w.t0) & (g["t"] < w.t1)
    psel = (run.p["t"] >= w.t0) & (run.p["t"] < w.t1)
    csel = (run.c["t"] >= w.t0) & (run.c["t"] < w.t1)
    m = {}
    if sel.sum() < 20:
        return m
    cmd = w.cmd
    dur = w.t1 - w.t0
    has_truth = "v_true" in g and np.isfinite(g["v_true"][sel]).any()
    v_odo = g["v_odo"][sel]
    v_best = g["v_true"][sel] if has_truth else v_odo
    wz = g["gyro_f"][sel, 2]
    m["vx_meas"] = np.nanmean(v_best[:, 0])
    m["vy_meas"] = np.nanmean(v_best[:, 1])
    m["wz_meas"] = float(np.mean(wz))
    m["lin_err[m/s]"] = np.nanmean(np.linalg.norm(v_best[:, :2] - cmd[:2], axis=1))
    m["yaw_err[rad/s]"] = float(np.mean(np.abs(wz - cmd[2])))
    m["vx_odo"] = np.nanmean(v_odo[:, 0])
    m["vy_odo"] = np.nanmean(v_odo[:, 1])
    m["lin_err_odo[m/s]"] = np.nanmean(np.linalg.norm(v_odo[:, :2] - cmd[:2], axis=1))
    z_t = run.z_target + (cmd[3] if len(cmd) > 3 else 0.0)
    z = g["z_true"][sel] if has_truth else g["z_est"][sel]
    m["z[m]"] = np.nanmean(z)
    m["z_err[cm]"] = 100.0 * np.nanmean(np.abs(z - z_t))
    m["z_est[m]"] = np.nanmean(g["z_est"][sel])
    roll, pitch = quat_rp(g["quat"][sel])
    m["roll[deg]"], m["pitch[deg]"] = float(np.mean(roll)), float(np.mean(pitch))
    tilt = np.degrees(np.arccos(np.clip(K.world_vertical(g["quat"][sel])[:, 2], -1, 1)))
    m["tilt_max[deg]"] = float(tilt.max())
    if w.kind == "stand":
        pos = g.get("pos_true")
        if has_truth and pos is not None and np.isfinite(pos[sel]).all():
            m["drift[m]"] = float(np.linalg.norm(pos[sel][-1, :2] - pos[sel][0, :2]))
        else:
            m["drift[m]"] = float(np.linalg.norm(np.nansum(K.rotate(g["quat"][sel], np.nan_to_num(v_odo))[:, :2], 0))
                                  / GRID_HZ)
        yaw = quat_yaw(g["quat"][sel][[0, -1]])
        m["yaw_drift[deg/s]"] = float(np.degrees(np.angle(np.exp(1j * (yaw[1] - yaw[0])))) / dur)
    # smoothness of the targets the policy sent (50 Hz)
    if psel.sum() > 8:
        dt = float(run.cfg["policy_dt"])
        qt = run.default_q + run.cfg["action_scale"] * run.p["action"][psel].astype(float)
        m["act_rate[rad/s]"] = float(np.mean(np.sqrt(np.mean((np.diff(qt, axis=0) / dt) ** 2, axis=1))))
        m["act_acc[rad/s2]"] = float(np.mean(np.sqrt(np.mean((np.diff(qt, 2, axis=0) / dt ** 2) ** 2, axis=1))))
        f, pq = psd(qt, 1.0 / dt)
        m[f"hf>{HF_HZ:.0f}Hz[%]"] = float(100.0 * np.mean(pq[f >= HF_HZ].sum(0) / np.maximum(pq[1:].sum(0), 1e-12)))
    # torques (what the motors report; Genesis: what was applied)
    tau = g["tau_est"][sel]
    m["tau_rms[Nm]"] = float(np.sqrt(np.mean(tau ** 2)))
    f, pt = psd(tau, GRID_HZ)
    m[f"tau>{HF_HZ:.0f}Hz[Nm]"] = float(np.sqrt(np.mean(pt[f >= HF_HZ].sum(0) * (f[1] - f[0]))))
    ratio = np.abs(tau) / run.torque_limits
    m["tau_peak[%lim]"] = float(100.0 * ratio.max())
    m["tau_sat[%time]"] = float(100.0 * np.mean((ratio >= 0.95).any(1)))
    m["tau_asked_max[%lim]"] = float(100.0 * (np.abs(run.c["tau_cmd"][csel]) / run.torque_limits).max()) \
        if csel.any() else float("nan")
    q = g["q"][sel]
    near = (q <= run.q_lo + 0.02) | (q >= run.q_hi - 0.02)
    m["jlim_hits[/min]"] = float((near[0].sum() + (near[1:] & ~near[:-1]).sum()) / (dur / 60.0))
    # energy
    power = np.abs(tau * g["dq"][sel]).sum(1)
    m["power[W]"] = float(np.mean(power))
    speed = np.nanmean(np.linalg.norm(v_best[:, :2], axis=1))
    m["speed[m/s]"] = speed
    m["CoT[-]"] = float(m["power[W]"] / (K.MASS * G * speed)) if np.linalg.norm(cmd[:2]) >= 0.1 else float("nan")
    # gait from the estimated contacts (and from the true ones when the simulator logged foot forces)
    for suffix, key in (("", "stance"), ("_true", "stance_true")):
        if key not in g:
            continue
        st = g[key][sel]
        m[f"duty{suffix}[-]"] = float(st.mean())
        td = (st[1:] & ~st[:-1]).sum()
        m[f"step_freq{suffix}[Hz]"] = float(td / 4.0 / dur)
    st, fh = g["stance"][sel], g["foot_h"][sel]
    heights = []
    for k in range(4):
        a_, b_ = runs_of(~st[:, k])
        for a, b in zip(a_, b_):
            if a > 0 and b < len(st) and b - a > 2 and np.isfinite(fh[a:b, k]).any():
                heights.append(np.nanmax(fh[a:b, k]))
    m["swing_h[cm]"] = 100.0 * float(np.mean(heights)) if heights else float("nan")
    # timing during the phase
    if csel.any():
        m["loop_p99[ms]"] = 1e3 * float(np.percentile(run.c["loop_dt"][csel], 99))
        m["state_age_p99[ms]"] = 1e3 * float(np.percentile(run.c["state_age"][csel], 99))
    if psel.any():
        m["infer_p99[ms]"] = 1e3 * float(np.percentile(run.p["infer"][psel], 99))
    return m


def quat_yaw(quat):
    """Yaw [rad] (ZYX) from (w, x, y, z) rows."""
    w, x, y, z = quat.T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# ======================================================================================================================
# Joint test (robot hung)
# ======================================================================================================================
def welch_tf(x, y, fs, seg_s=2.0):
    """Transfer function y/x by Welch averaging (Hann, 50 % overlap): f, H(f), coherence-free."""
    n = int(seg_s * fs)
    win = np.hanning(n)
    sxx, syx = 0.0, 0.0
    for i in range(0, len(x) - n + 1, n // 2):
        X = np.fft.rfft(win * (x[i:i + n] - x[i:i + n].mean()))
        Y = np.fft.rfft(win * (y[i:i + n] - y[i:i + n].mean()))
        sxx = sxx + np.abs(X) ** 2
        syx = syx + Y * np.conj(X)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    return f, syx / np.maximum(sxx, 1e-20)


def joint_test_results(run, w):
    """Per joint kind: step response metrics and the chirp frequency response (targets -> measured positions)."""
    plan = run.meta.get("joint_test")
    if not plan:
        return None
    g = run.grid()
    handover = float(run.fsm_cfg.get("handover_s", 0.5))
    t_plan = w.t0 + handover
    cfg = plan["cfg"]
    kinds = np.array(run.kinds)
    stand = run.default_q
    out = {"t_plan": t_plan, "kinds": {}}
    prev = {k: 0.0 for k in ("hip", "thigh", "calf")}
    steps = {k: [] for k in prev}
    for t_s, kind, amp in plan["steps"]:
        delta, prev[kind] = amp - prev[kind], amp
        if abs(delta) < 1e-6:
            continue
        ts = t_plan + t_s
        cols = kinds == kind
        pre = (g["t"] >= ts - 0.03) & (g["t"] < ts)
        win = (g["t"] >= ts) & (g["t"] < ts + cfg["step_s"])
        if pre.sum() < 2 or win.sum() < 10:
            continue
        y = (g["q"][win][:, cols] - g["q"][pre][:, cols].mean(0)).mean(1) / delta
        tt = g["t"][win] - ts
        steps[kind].append((tt, y, abs(delta)))
    for kind in ("hip", "thigh", "calf"):
        r = {}
        if steps[kind]:
            d10, d50, rise, over, sse = [], [], [], [], []
            for tt, y, dlt in steps[kind]:
                i10, i50, i90 = (np.argmax(y >= v) if np.any(y >= v) else -1 for v in (0.1, 0.5, 0.9))
                d10.append(tt[i10] if i10 >= 0 else np.nan)
                d50.append(tt[i50] if i50 >= 0 else np.nan)
                rise.append(tt[i90] - tt[i10] if i90 >= 0 and i10 >= 0 else np.nan)
                over.append(100.0 * max(y.max() - 1.0, 0.0))
                sse.append(1e3 * abs(1.0 - y[tt >= tt[-1] - 0.1].mean()) * dlt)
            r.update({"t10[ms]": 1e3 * np.nanmean(d10), "t50[ms]": 1e3 * np.nanmean(d50),
                      "rise10-90[ms]": 1e3 * np.nanmean(rise), "overshoot[%]": float(np.mean(over)),
                      "ss_err[mrad]": float(np.mean(sse))})
            r["_curve"] = (steps[kind][0][0], np.mean([np.interp(steps[kind][0][0], tt, y) for tt, y, _ in steps[kind]], 0))
        # chirp, taper excluded
        tc0 = t_plan + plan["chirp_t0"] + cfg["taper_s"]
        tc1 = t_plan + plan["chirp_t0"] + cfg["chirp_s"] - cfg["taper_s"]
        win = (g["t"] >= tc0) & (g["t"] < tc1)
        if win.sum() > 4 * GRID_HZ:
            cols = np.flatnonzero(kinds == kind)
            x = g["q_des"][win][:, cols] - stand[cols]
            y = g["q"][win][:, cols] - stand[cols]
            H = None
            for j in range(len(cols)):
                f, h = welch_tf(x[:, j], y[:, j], GRID_HZ)
                H = h if H is None else H + h
            H /= len(cols)
            r["_bode"] = (f, H)
            for fq in (1.0, 2.0, 4.0, 8.0):
                hq = np.interp(fq, f, H.real) + 1j * np.interp(fq, f, H.imag)
                r[f"gain@{fq:.0f}Hz"] = float(abs(hq))
                ph = float(np.angle(hq))
                r[f"phase@{fq:.0f}Hz[deg]"] = math.degrees(ph)
                r[f"lag@{fq:.0f}Hz[ms]"] = -1e3 * ph / (2 * math.pi * fq)
            r["chirp_rms_err[mrad]"] = 1e3 * float(np.sqrt(np.mean((y - x) ** 2)))
            r["_chirp"] = (g["t"][win] - t_plan, y.mean(1), x.mean(1))
        out["kinds"][kind] = r
    return out


def chirp_lag(ref, other, kind, max_ms=60):
    """Extra delay [ms] of `other` against `ref`: lag maximizing the cross-correlation of their chirp responses."""
    a, b = ref["kinds"].get(kind, {}).get("_chirp"), other["kinds"].get(kind, {}).get("_chirp")
    if a is None or b is None:
        return float("nan")
    t = np.arange(max(a[0][0], b[0][0]), min(a[0][-1], b[0][-1]), 0.001)
    ya, yb = np.interp(t, a[0], a[1]), np.interp(t, b[0], b[1])
    ya, yb = ya - ya.mean(), yb - yb.mean()
    lags = np.arange(-max_ms, max_ms + 1)
    corr = [np.dot(ya[max(0, -l):len(ya) - max(0, l)], yb[max(0, l):len(yb) - max(0, -l)]) for l in lags]
    return float(lags[int(np.argmax(corr))])


# ======================================================================================================================
# Run summary: timing, safety, events
# ======================================================================================================================
def timing(run):
    c, p = run.c, run.p
    out = {}
    nominal = float(run.meta.get("control_dt", 0.002))
    ldt = c["loop_dt"][c["loop_dt"] > 0]
    if len(ldt):
        out["loop period p50 / p99 / max [ms]"] = 1e3 * np.array([np.percentile(ldt, 50), np.percentile(ldt, 99), ldt.max()])
        out["loop steps > 2x period"] = int(np.sum(ldt > 2 * nominal))
    out["state age p50 / p99 / max [ms]"] = 1e3 * np.array([np.percentile(c["state_age"], 50),
                                                             np.percentile(c["state_age"], 99), c["state_age"].max()])
    out["control step p50 / p99 / max [ms]"] = 1e3 * np.array([np.percentile(c["compute"], 50),
                                                                np.percentile(c["compute"], 99), c["compute"].max()])
    out["stale-state steps (> 50 ms)"] = int(np.sum((c["flags"] & go2_log.FLAG_STALE) > 0))
    out["steps reusing the previous state [%]"] = 100.0 * float(np.mean((c["flags"] & go2_log.FLAG_REPEAT) > 0))
    if len(p.get("t", [])) > 2:
        pdt = np.diff(p["t"])
        pdt = pdt[pdt < 3 * float(run.cfg["policy_dt"])]
        if len(pdt):
            out["policy step interval p50 / p99 / max [ms]"] = 1e3 * np.array([np.percentile(pdt, 50),
                                                                               np.percentile(pdt, 99), pdt.max()])
        out["inference p50 / p99 / max [ms]"] = 1e3 * np.array([np.percentile(p["infer"], 50),
                                                                 np.percentile(p["infer"], 99), p["infer"].max()])
    out["targets clipped to the joint range [steps]"] = int(np.sum((c["flags"] & go2_log.FLAG_CLAMPED) > 0))
    return out


TRIP_WORDS = ("tilt", "torque", "outside", "error", "not finite", "no robot state")


def find_trips(run):
    """The safety trips of a run: state machine events that sent it to damping on their own."""
    return [e for e in run.log.events if any(k in e[1] for k in TRIP_WORDS)]


def evaluate(run):
    """Windows, metrics per phase, trips and the H3 verdict of one run, without printing (go2_sim_suite.py)."""
    windows = run.windows()
    metrics = {w.name: phase_metrics(run, w) for w in windows if w.kind != "joint_test"}
    trips = find_trips(run)
    return {"windows": windows, "metrics": metrics, "trips": trips, "fails": gate(run, metrics, trips, windows),
            "timing": timing(run)}


def print_run(run):
    m = run.meta
    src = m["deploy_cfg"].get("source", {})
    dur = run.c["t"][-1] - run.c["t"][0] if len(run.c["t"]) else 0.0
    print(f"\n=== {run.letter} = {run.name} ({run.source}) ===")
    print(f"policy {src.get('exp_name', '?')} ckpt {src.get('ckpt', '?')}"
          + (" ZERO ACTION" if m.get("zero_action") else "") + f" | {dur:.1f} s | profile {m.get('profile_name')} | "
          f"scenario {m.get('scenario')} | truth: {run.truth['kind'] if run.truth else 'none'}"
          + (f" | {sum(m.get('dropped_rows', {}).values())} rows dropped (ring buffer)" if any(m.get('dropped_rows', {}).values()) else ""))
    print("modes: " + " -> ".join(f"{name} {a:.1f}-{b:.1f} s" for a, b, name in run.mode_spans()))
    trips = find_trips(run)
    for t, msg in sorted(list(run.log.events) + list(getattr(run, "sim_events", [])), key=lambda e: e[0]):
        print(f"  {t:8.3f} s  {msg}")
    print("timing:")
    for k, v in timing(run).items():
        print(f"  {k:45s} " + (" / ".join(f"{x:.2f}" for x in v) if isinstance(v, np.ndarray) else f"{v:.1f}" if isinstance(v, float) else f"{v}"))
    return trips


# ======================================================================================================================
# Side-by-side tables
# ======================================================================================================================
SIGNED = ("vx", "vy", "wz", "z[m]", "z_est", "roll", "pitch", "yaw_drift", "phase", "lag", "t10", "t50", "rise")


def fmt_cell(name, v, ref):
    """'value (change vs the reference)': a difference d... for signed quantities, times and near-zero references,
    else a percentage."""
    if v is None or not np.isfinite(v):
        return "-"
    text = f"{v:.3g}"
    if ref is None or ref is v or not np.isfinite(ref):
        return text
    pct = 100.0 * (v - ref) / abs(ref) if abs(ref) > 1e-12 else float("inf")
    if name.startswith(SIGNED) or abs(pct) > 300:
        return f"{text} (d{v - ref:+.2g})"
    return f"{text} ({pct:+.0f}%)"


def print_table(title, rows, header):
    widths = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
    print(f"\n--- {title} ---")
    print("  ".join(str(h).rjust(wd) if i else str(h).ljust(wd) for i, (h, wd) in enumerate(zip(header, widths))))
    for r in rows:
        print("  ".join(str(x).rjust(wd) if i else str(x).ljust(wd) for i, (x, wd) in enumerate(zip(r, widths))))


def gate(run, metrics, trips, windows):
    """H3 gate verdict for one run (meant for unitree_mujoco through the deploy script)."""
    fails = [f"{msg} at {t:.1f} s" for t, msg in trips]
    if not any(metrics.values()) and not any(w.kind == "joint_test" for w in windows):
        fails.append("no policy phase long enough to measure")
    if not str(run.meta.get("scenario") or "").startswith("estop"):  # the e-stop scenarios end the profile on purpose
        fails += [f"{msg} at {t:.1f} s" for t, msg in run.log.events if " ended: " in msg and not msg.endswith(": done")]
    for name, m in metrics.items():
        if not m:
            continue
        if m.get("lin_err[m/s]", 0) > GATE["lin_err"]:
            fails.append(f"{name}: lin_err {m['lin_err[m/s]']:.2f} m/s > {GATE['lin_err']}")
        if m.get("yaw_err[rad/s]", 0) > GATE["yaw_err"]:
            fails.append(f"{name}: yaw_err {m['yaw_err[rad/s]']:.2f} rad/s > {GATE['yaw_err']}")
        if m.get("tau_peak[%lim]", 0) > GATE["tau_peak"]:
            fails.append(f"{name}: torque peak {m['tau_peak[%lim]']:.0f} % of the limit")
        if m.get("tilt_max[deg]", 0) > GATE["tilt_max"]:
            fails.append(f"{name}: tilt {m['tilt_max[deg]']:.0f} deg")
        if m.get("drift[m]", 0) > GATE["stand_drift"]:
            fails.append(f"{name}: drifted {m['drift[m]']:.2f} m while standing")
    return fails


# ======================================================================================================================
# Plots
# ======================================================================================================================
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


INK, MUTED, GRID_C = "#0b0b0b", "#52514e", "#e6e5e0"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a5cd1"]
MODE_BAND = {"stand_up": "#f0efec", "policy": "#e8f0fb", "sit_down": "#f0efec", "joint_test": "#fdf0e6"}


def _style(ax, ylabel):
    ax.set_ylabel(ylabel, fontsize=8, color=INK)
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.grid(True, axis="y", color=GRID_C, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_overview(run, path):
    plt = _plt()
    g = run.grid()
    t = g["t"]
    fig, axes = plt.subplots(5, 1, figsize=(10, 9), sharex=True)
    for ax in axes:
        for a, b, name in run.mode_spans():
            if name in MODE_BAND:
                ax.axvspan(a, b, color=MODE_BAND[name], linewidth=0)
    pt, cmd = run.p.get("t", np.zeros(0)), run.p.get("cmd", np.zeros((0, 3)))
    ax = axes[0]
    if len(pt):
        ax.step(pt, cmd[:, 0], where="post", color=INK, linewidth=1, label="vx command")
    ax.plot(t, g["v_odo"][:, 0], color=COLORS[0], linewidth=1, label="vx leg odometry")
    if "v_true" in g:
        ax.plot(t, g["v_true"][:, 0], color=COLORS[1], linewidth=1, label="vx truth")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    _style(ax, "vx [m/s]")
    ax = axes[1]
    if len(pt):
        ax.step(pt, cmd[:, 2], where="post", color=INK, linewidth=1, label="wz command")
    ax.plot(t, g["gyro_f"][:, 2], color=COLORS[0], linewidth=1, label="gyro z")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    _style(ax, "wz [rad/s]")
    ax = axes[2]
    ax.plot(t, g["z_est"], color=COLORS[0], linewidth=1, label="height (kinematics)")
    if "z_true" in g:
        ax.plot(t, g["z_true"], color=COLORS[1], linewidth=1, label="height truth")
    roll, pitch = quat_rp(g["quat"])
    ax2 = ax.twinx()
    ax2.plot(t, pitch, color=COLORS[2], linewidth=0.8, label="pitch [deg]")
    ax2.tick_params(labelsize=7, colors=MUTED)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax2.legend(frameon=False, fontsize=7, loc="upper right")
    _style(ax, "base height [m]")
    ax = axes[3]
    for j, name in ((1, "FR thigh"), (2, "FR calf")):
        ax.plot(t, g["q"][:, j], linewidth=1, color=COLORS[j - 1], label=f"{name} q")
        ax.plot(t, g["q_des"][:, j], linewidth=0.8, linestyle="--", color=COLORS[j - 1], label=f"{name} target")
    ax.legend(frameon=False, fontsize=7, loc="upper right", ncol=2)
    _style(ax, "joint [rad]")
    ax = axes[4]
    ax.plot(t, 100 * (np.abs(g["tau_est"]) / run.torque_limits).max(1), color=COLORS[0], linewidth=0.8,
            label="max |tau_est| / limit")
    ax.plot(t, 100 * (np.abs(g["tau_cmd"]) / run.torque_limits).max(1), color=COLORS[1], linewidth=0.8,
            label="max |torque asked| / limit")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    _style(ax, "torque [% limit]")
    axes[-1].set_xlabel("time [s]", fontsize=8)
    fig.suptitle(f"{run.letter} = {run.name} ({run.source}); shaded: policy (blue), stand/sit (grey), joint test (orange)",
                 fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_overlay(runs, path):
    """The runs aligned on their first profile segment (else policy start): velocity, yaw rate, height, a calf."""
    plt = _plt()
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for k, run in enumerate(runs):
        g = run.grid()
        seg = run.p.get("seg", np.zeros(0))
        if len(seg) and np.any(seg >= 0):
            t0 = run.p["t"][np.argmax(seg >= 0)]
            what = "profile start"
        elif len(run.p.get("t", [])):
            t0, what = run.p["t"][0], "policy start"
        else:
            t0, what = g["t"][0], "log start"
        t = g["t"] - t0
        col = COLORS[k % len(COLORS)]
        v = g["v_true"] if "v_true" in g else g["v_odo"]
        axes[0].plot(t, v[:, 0], color=col, linewidth=1, label=f"{run.letter} {run.source}")
        axes[1].plot(t, g["gyro_f"][:, 2], color=col, linewidth=1)
        axes[2].plot(t, g["z_true"] if "z_true" in g else g["z_est"], color=col, linewidth=1)
        axes[3].plot(t, g["q"][:, 2], color=col, linewidth=1)
        if k == 0 and len(run.p.get("t", [])):
            axes[0].step(run.p["t"] - t0, run.p["cmd"][:, 0], where="post", color=INK, linewidth=0.8, label="command")
            axes[1].step(run.p["t"] - t0, run.p["cmd"][:, 2], where="post", color=INK, linewidth=0.8)
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    for ax, lab in zip(axes, ("vx [m/s] (truth or odometry)", "wz gyro [rad/s]", "base height [m]", "FR calf [rad]")):
        _style(ax, lab)
    axes[-1].set_xlabel(f"time since {what} [s]", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_joint_test(runs, results, path):
    plt = _plt()
    fig, axes = plt.subplots(3, 3, figsize=(11, 8))
    for col, kind in enumerate(("hip", "thigh", "calf")):
        for k, (run, res) in enumerate(zip(runs, results)):
            if res is None:
                continue
            r = res["kinds"].get(kind, {})
            c = COLORS[k % len(COLORS)]
            if "_curve" in r:
                tt, y = r["_curve"]
                axes[0, col].plot(1e3 * tt, y, color=c, linewidth=1.2, label=f"{run.letter} {run.source}")
            if "_bode" in r:
                f, H = r["_bode"]
                ok = (f >= 0.7) & (f <= 10.0)
                axes[1, col].plot(f[ok], np.abs(H[ok]), color=c, linewidth=1.2)
                axes[2, col].plot(f[ok], np.degrees(np.angle(H[ok])), color=c, linewidth=1.2)
        axes[0, col].axhline(1.0, color=MUTED, linewidth=0.6, linestyle=":")
        axes[0, col].set_title(kind, fontsize=9, loc="left")
        _style(axes[0, col], "step response (normalized)")
        axes[0, col].set_xlabel("time since the target step [ms]", fontsize=7)
        _style(axes[1, col], "gain |q / q*|")
        _style(axes[2, col], "phase [deg]")
        axes[2, col].set_xlabel("frequency [Hz]", fontsize=7)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Joint test: target steps and chirp, policy gains, robot hung", fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ======================================================================================================================
# Main
# ======================================================================================================================
def attach_truth(runs, truths):
    """A headless-simulator truth file joins the run whose time span overlaps it (CLOCK_MONOTONIC is shared)."""
    for tr in truths:
        t0 = tr.meta["t0_mono"]
        span = (t0 + tr.truth["t"][0], t0 + tr.truth["t"][-1])
        for run in runs:
            r0 = run.meta["t0_mono"]
            rspan = (r0 + run.c["t"][0], r0 + run.c["t"][-1])
            if run.source in ("mujoco", "rehearsal") and span[0] < rspan[1] and rspan[0] < span[1]:
                run.set_truth(tr.truth, "full", t0 - r0)
                # what the simulator did on purpose (pushes, blackouts, fake services), on the run's clock
                run.sim_events = [(float(t) + t0 - r0, f"[sim] {msg}") for t, msg in tr.events]
                print(f"{tr.name}: ground truth attached to {run.letter} = {run.name}")
                break
        else:
            print(f"{tr.name}: no unitree_mujoco run overlaps this truth file in time")


def probe_odometry(lg):
    """A probe log recorded while the robot walks under Unitree's own controller (sport mode, remote): the leg odometry
    and contact estimate of this file against Unitree's state estimator (rt/sportmodestate). Validates the estimator
    before it judges your policy, on the real robot."""
    st, sp = lg.streams.get("state", {}), lg.streams.get("sport", {})
    if len(sp.get("t", [])) < 50 or np.abs(sp["velocity"]).max() < 0.05 or len(st.get("t", [])) < 100:
        return
    t = st["t"]
    rate = (len(t) - 1) / (t[-1] - t[0])
    q, dq, tau, gyro = (st[k].astype(float) for k in ("q", "dq", "tau_est", "gyro"))
    stance = K.debounce(K.contacts(K.foot_forces(q, tau)[..., 2]), int(round(0.04 * rate)))
    v = moving_average(K.leg_odometry(q, dq, gyro, stance), max(1, int(0.2 * rate)))
    speed_odo = np.interp(sp["t"], t, np.linalg.norm(np.nan_to_num(v[:, :2]), axis=1))
    speed_sport = np.linalg.norm(sp["velocity"][:, :2].astype(float), axis=1)
    moving = speed_sport > 0.1
    print("\n=== leg odometry vs Unitree's estimator (rt/sportmodestate), horizontal speed ===")
    if moving.sum() < 20:
        print("the robot hardly moved: walk it around with the remote during the probe")
        return
    err = speed_odo[moving] - speed_sport[moving]
    print(f"while moving ({moving.sum()} samples): Unitree {speed_sport[moving].mean():.3f} m/s, leg odometry "
          f"{speed_odo[moving].mean():.3f} m/s, bias {err.mean():+.3f} m/s, RMS {np.sqrt(np.mean(err ** 2)):.3f} m/s")
    yaw_sport = sp["yaw_speed"].astype(float)
    gz = np.interp(sp["t"], t, st["gyro"][:, 2].astype(float))
    print(f"yaw rate: gyro z vs sportmodestate yaw_speed RMS difference {np.sqrt(np.mean((gz - yaw_sport) ** 2)):.3f} rad/s")
    print(f"estimated duty factor while moving {np.interp(sp['t'], t, stance.mean(1))[moving].mean():.2f} "
          f"(trot about 0.5-0.6)")


def write_csv(run, metrics, windows, path):
    """go2_metrics.py-style rows (one per measured phase) so go2_compare.py can compare a deploy run with a Genesis
    evaluation: vx, vy, wz[, dh], ood, falls, <metric>_mean, <metric>_std (std empty: one run, not envs)."""
    import csv

    names = []
    for w in windows:
        for k in metrics.get(w.name, {}):
            if k not in names:
                names.append(k)
    n_cmd = len(run.cfg.get("commands_scale", [0, 0, 0]))
    cmd_cols = ["vx", "vy", "wz", "dh"][:n_cmd]
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(cmd_cols + ["ood", "falls", "phase"] + [f"{n}_{s}" for n in names for s in ("mean", "std")])
        written = set()
        for w in windows:
            key = tuple(np.round(w.cmd, 3)) if w.cmd is not None else None
            if key is None or w.name not in metrics or key in written:  # go2_compare.py keys rows by command
                continue
            written.add(key)
            m = metrics[w.name]
            fell = any(w.t0 <= t <= w.t1 + 0.5 and "-> passive" in msg and "Ctrl+C" not in msg
                       for t, msg in run.log.events)
            wr.writerow([f"{v:g}" for v in w.cmd[:n_cmd]] + [False, f"{int(fell)}/1", w.name]
                        + [x for n in names for x in (f"{m.get(n, float('nan')):.6g}", "")])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="+", help="go2_log.py files (deploy, Genesis bench, headless truth, probe)")
    p.add_argument("--plots", type=str, default=None, help="folder for the figures")
    p.add_argument("--csv", type=str, default=None, help="folder for go2_metrics-style CSV files")
    p.add_argument("--all-metrics", action="store_true", help="every metric in the tables (default: the main ones)")
    args = p.parse_args()

    logs = [go2_log.load(path) for path in args.logs]
    for lg in logs:
        if lg.source == "probe":
            import go2_probe

            print(f"\n##### probe {lg.name} #####")
            go2_probe.report(lg.streams.get("state", {}), lg.streams.get("loop", {}), lg.streams.get("sport", {}),
                             sim=lg.meta.get("sim", False))
            probe_odometry(lg)
    run_logs = [lg for lg in logs if "deploy_cfg" in lg.meta]
    if not run_logs:
        return
    runs = [Run(lg, chr(ord("A") + i)) for i, lg in enumerate(run_logs)]
    attach_truth(runs, [lg for lg in logs if lg.source == "mujoco_truth"])

    trips, all_windows, metrics = {}, {}, {}
    for run in runs:
        trips[run.letter] = print_run(run)
        all_windows[run.letter] = run.windows()
        metrics[run.letter] = {w.name: phase_metrics(run, w) for w in all_windows[run.letter] if w.kind != "joint_test"}

    main_metrics = ["vx_meas", "vy_meas", "wz_meas", "lin_err[m/s]", "yaw_err[rad/s]", "vx_odo", "lin_err_odo[m/s]",
                    "z[m]", "z_est[m]", "pitch[deg]", "tilt_max[deg]", "drift[m]", "yaw_drift[deg/s]",
                    "act_rate[rad/s]", "act_acc[rad/s2]", f"hf>{HF_HZ:.0f}Hz[%]", "tau_rms[Nm]",
                    f"tau>{HF_HZ:.0f}Hz[Nm]", "tau_peak[%lim]", "power[W]", "CoT[-]", "duty[-]", "duty_true[-]",
                    "step_freq[Hz]", "step_freq_true[Hz]", "swing_h[cm]", "loop_p99[ms]", "infer_p99[ms]"]
    order = []
    for run in runs:
        for w in all_windows[run.letter]:
            if w.kind != "joint_test" and w.name not in [o.name for o in order]:
                order.append(w)
    header = ["metric"] + [f"{r.letter} {r.source}" for r in runs]
    for w in order:
        names = [k for k in (main_metrics if not args.all_metrics else
                             sorted({k for r in runs for k in metrics[r.letter].get(w.name, {})}))
                 if any(k in metrics[r.letter].get(w.name, {}) for r in runs)]
        rows = []
        for name in names:
            ref = metrics[runs[0].letter].get(w.name, {}).get(name)
            rows.append([name] + [fmt_cell(name, metrics[r.letter].get(w.name, {}).get(name), ref) for r in runs])
        print_table(w.label(), rows, header)

    # joint test
    jt = [(run, next((w for w in all_windows[run.letter] if w.kind == "joint_test"), None)) for run in runs]
    results = [joint_test_results(run, w) if w is not None else None for run, w in jt]
    if any(r is not None for r in results):
        ref = next(r for r in results if r is not None)
        for kind in ("hip", "thigh", "calf"):
            names = sorted({k for r in results if r for k in r["kinds"].get(kind, {}) if not k.startswith("_")},
                           key=lambda s: (not s.startswith("t"), s))
            rows = []
            for name in names:
                rv = ref["kinds"].get(kind, {}).get(name)
                rows.append([name] + [fmt_cell(name, r["kinds"].get(kind, {}).get(name), rv) if r else "-" for r in results])
            rows.append(["extra delay vs A, chirp [ms]"] + [f"{chirp_lag(ref, r, kind):+.0f}" if r else "-" for r in results])
            print_table(f"joint test: {kind}s (targets -> measured)", rows, header)

    print("\n=== H3 gate (meant for unitree_mujoco runs through the deploy script) ===")
    for run in runs:
        fails = gate(run, metrics[run.letter], trips[run.letter], all_windows[run.letter])
        print(f"{run.letter} {run.source:8s} " + ("PASS" if not fails else "FAIL: " + "; ".join(fails)))

    if args.plots:
        os.makedirs(args.plots, exist_ok=True)
        for run in runs:
            path = os.path.join(args.plots, f"overview_{run.letter}_{run.name}.png")
            plot_overview(run, path)
            print(f"plot: {path}")
        if len(runs) > 1:
            path = os.path.join(args.plots, "overlay.png")
            plot_overlay(runs, path)
            print(f"plot: {path}")
        if any(r is not None for r in results):
            path = os.path.join(args.plots, "joint_test.png")
            plot_joint_test(runs, results, path)
            print(f"plot: {path}")
    if args.csv:
        os.makedirs(args.csv, exist_ok=True)
        for run in runs:
            path = os.path.join(args.csv, f"metrics_{run.name}.csv")
            write_csv(run, metrics[run.letter], all_windows[run.letter], path)
            print(f"csv: {path}  (python go2_compare.py <genesis metrics csv> {path})")


if __name__ == "__main__":
    main()
