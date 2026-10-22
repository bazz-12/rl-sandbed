"""
go2_sim_suite.py - [S2R] Runs whole batteries of cases in unitree_mujoco, each one through go2_deploy_sdk2.py as on the
robot, and checks every run against what it should do: walk and pass the H3 checks, stay upright under a perturbation,
or trip into damping for the right reason. Everything you can test before the robot exists, in one command.

    python go2_sim_suite.py --list --suite all                                                  # the cases
    python go2_sim_suite.py --policy logs/<exp>/exported --um ~/unitree_mujoco                  # suite "gate"
    python go2_sim_suite.py --policy logs/<exp>/exported --um ~/unitree_mujoco --suite stress --rt
    python go2_sim_suite.py --policy logs/<exp>/exported --um ~/unitree_mujoco --suite all --repeat 2
    python go2_sim_suite.py --policy logs/<exp>/exported --um ~/unitree_mujoco --only fwd_back_mu1,blackout

Suites (--list prints every case):
    gate        H3: the five robot profiles at foot friction 1.0 (must pass) and at unitree_mujoco's 0.4 (must stay
                safe: no fall, no trip)
    stress      one robot-like imperfection at a time (noise, delays, lost messages, payload, CoM, weak motors, pushes,
                per-step PD, encoder offset) on fwd_back at friction 1.0: inside the training ranges it must pass,
                beyond them it must stay safe
    faults      the safety layer: e-stops, re-stand, a state blackout, swapped motors, a reversed motor; each must end in
                damping for the right reason
    rehearsal   the robot's own code path (go2_deploy_sdk2.py --rehearsal) against fake robot services, the remote
                scripted as raw bytes
    actuators   the joint test with the base pinned
    all         everything above
Each case runs --repeat times (default 2: unitree_mujoco over DDS is not deterministic). Expectations:
    pass        the H3 verdict of go2_analyze.py is PASS
    safe        no safety trip, no fall, the profile ran to its end; only tracking errors (lin_err, yaw_err) allowed
    trip:TEXT   a safety trip whose message contains TEXT ended the run in damping
    event:TEXT  an event containing TEXT happened (e.g. "L1+Y")
Writes out/suite_<date>/summary.md and summary.csv; the logs stay in --log-dir (logs/deploy).
Run it alone on the machine: the simulator and the deploy loop must both keep real time.
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field

import numpy as np

import go2_analyze as A
import go2_log
from go2_fsm import MODES, FsmConfig, Go2StateMachine, Mode, Policy
from go2_inputs import CommandProfile, Scenario

HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Case:
    name: str
    suite: str
    scenario: str
    profile: str = None
    sim: list = field(default_factory=list)  # extra go2_mujoco_headless.py arguments
    deploy: list = field(default_factory=list)  # extra go2_deploy_sdk2.py arguments
    expect: str = "pass"
    note: str = ""
    rehearsal: bool = False  # robot path (--rehearsal) with the remote scripted by the simulator


MU1 = ["--foot-friction", "1.0"]
CASES = []
for prof in ("stand60", "slow", "fwd_back", "lateral", "yaw"):
    CASES.append(Case(f"{prof}_mu1", "gate", "walk", prof, MU1, expect="pass", note="H3, nominal friction"))
for prof in ("stand60", "slow", "fwd_back", "lateral", "yaw"):
    CASES.append(Case(f"{prof}_mu04", "gate", "walk", prof, [], expect="safe", note="unitree_mujoco's feet (0.4)"))
STRESS = [  # (name, simulator arguments, expectation, note)
    ("noise1", ["--noise", "1"], "pass", "training observation noise"),
    ("noise2", ["--noise", "2"], "safe", "twice the training noise"),
    ("cmd_delay10", ["--cmd-delay-ms", "10"], "pass", "inside the trained latency (0-20 ms)"),
    ("cmd_delay20", ["--cmd-delay-ms", "20"], "pass", "edge of the trained latency"),
    ("cmd_delay40", ["--cmd-delay-ms", "40"], "safe", "twice the trained latency"),
    ("state_delay10", ["--state-delay-ms", "10"], "safe", "observation latency (not trained)"),
    ("drop20", ["--drop", "0.2"], "pass", "20 % of rt/lowstate lost"),
    ("payload2", ["--payload", "2"], "pass", "edge of the trained added mass"),
    ("payload5", ["--payload", "5"], "safe", "beyond the trained added mass"),
    ("com3cm", ["--com-shift", "0.03,0,0"], "pass", "edge of the trained CoM shift"),
    ("weak90", ["--motor-strength", "0.9"], "pass", "edge of the trained motor strength"),
    ("weak75", ["--motor-strength", "0.75"], "safe", "beyond the trained motor strength"),
    ("push05", ["--push", "0.5"], "safe", "0.5 m/s kicks every 4 s"),
    ("pd_per_step", ["--pd-per-step"], "pass", "motor-driver PD instead of the bridge's held torque"),
    ("offset3", ["--offset", "FR_thigh:0.03,RL_calf:-0.03"], "safe", "encoder calibration errors (0.03 rad)"),
]
for name, extra, expect, note in STRESS:
    CASES.append(Case(name, "stress", "walk", "fwd_back", MU1 + extra, expect=expect, note=note))
CASES += [
    Case("estop", "faults", "estop", expect="event:L1+Y", note="L1+Y while the policy stands"),
    Case("estop_walk", "faults", "estop_walk", "fwd_back", MU1, expect="event:L1+Y", note="L1+Y while walking"),
    Case("restand", "faults", "restand", None, MU1, expect="pass", note="policy -> stand-up -> policy -> sit"),
    Case("blackout", "faults", "walk", "fwd_back", MU1 + ["--blackout", "12:0.3"], expect="trip:no robot state",
         note="rt/lowstate stops for 300 ms while walking"),
    Case("swap", "faults", "stand", None, MU1 + ["--swap", "FR_thigh,FR_calf"], expect="trip:torque",
         note="two motors wired to each other's joint"),
    Case("flip", "faults", "walk", "fwd_back", MU1 + ["--flip", "FR_thigh"], expect="event:L1+A refused",
         note="one motor turning the wrong way: the robot stands crooked, L1+A must be refused"),
    Case("rehearsal", "rehearsal", "walk", "fwd_back", MU1, expect="pass", rehearsal=True,
         note="robot path: sport-mode release, remote bytes"),
    Case("rehearsal_estop", "rehearsal", "estop_walk", "fwd_back", MU1, expect="event:L1+Y", rehearsal=True,
         note="robot path: L1+Y from the remote while walking"),
    Case("joint_test", "actuators", "joint_test", None, ["--hang", "0.6"], ["--allow-joint-test", "--zero-action"],
         expect="pass", note="base pinned, policy gains, steps and chirp"),
]
SUITES = ("gate", "stress", "faults", "rehearsal", "actuators")


def scenario_seconds(policy_dir, case):
    """How long the scripted part of a case lasts [s]."""
    fsm = Go2StateMachine(Policy(policy_dir, zero_action=True), FsmConfig(), verbose=False)
    profile = CommandProfile(case.profile, fsm.num_commands) if case.profile else None
    return Scenario(case.scenario, profile, fsm.joint_test_duration()).duration


def judge(expect, result):
    """(ok, why) of one run against its expectation."""
    kind, _, arg = expect.partition(":")
    fails, trips, events = result["fails"], result["trips"], result["events"]
    if kind == "pass":
        return (not fails), ("PASS" if not fails else "; ".join(fails))
    if kind == "safe":
        bad = [f for f in fails if "lin_err" not in f and "yaw_err" not in f]
        return (not bad), ("safe" + (f" (tracking: {'; '.join(fails)})" if fails else "") if not bad else "; ".join(bad))
    if kind == "trip":
        hit = [msg for t, msg in trips if arg in msg]
        if not hit:
            return False, "expected a trip" + (f" ({arg})" if arg else "") + ", none happened"
        return result["final_mode"] == "passive", f"tripped: {hit[0]}"
    if kind == "event":
        hit = [msg for t, msg in events if arg in msg]
        return bool(hit), (f"event: {hit[0]}" if hit else f"no event with {arg!r}")
    raise ValueError(f"unknown expectation {expect!r}")


def summarize(result):
    """A few numbers per run for the summary table."""
    out = {}
    for w in result["windows"]:
        m = result["metrics"].get(w.name, {})
        if w.cmd is not None and np.linalg.norm(w.cmd[:2]) >= 0.1 and m:
            key = "vx" if abs(w.cmd[0]) >= abs(w.cmd[1]) else "vy"
            out.setdefault("tracking", []).append(f"{key} {m.get(key + '_meas', float('nan')):+.2f}/{w.cmd[0 if key == 'vx' else 1]:+.2f}")
        elif w.cmd is not None and abs(w.cmd[2]) >= 0.1 and m:
            out.setdefault("tracking", []).append(f"wz {m.get('wz_meas', float('nan')):+.2f}/{w.cmd[2]:+.2f}")
    ms = [m for m in result["metrics"].values() if m]
    out["tilt_max"] = max((m.get("tilt_max[deg]", 0.0) for m in ms), default=float("nan"))
    out["tau_peak"] = max((m.get("tau_peak[%lim]", 0.0) for m in ms), default=float("nan"))
    out["loop_p99"], out["age_max"] = result.get("driven_timing", (np.nan, np.nan))
    return out


def driven_timing(c):
    """(loop period p99, longest state age) [ms] while the robot is driven: damping at the start and at the end (the
    deploy script saving its log, the simulator stopping) is left out."""
    active = c["mode"] != MODES.index(Mode.PASSIVE)
    ldt = c["loop_dt"][active & (c["loop_dt"] > 0)]
    if not len(ldt):
        return np.nan, np.nan
    return 1e3 * float(np.percentile(ldt, 99)), 1e3 * float(c["state_age"][active].max())


def run_case(case, rep, args, py, logf):
    tag = f"suite_{case.name}_{rep}"
    truth = os.path.join(args.log_dir, f"mjtruth_{tag}.npz")
    seconds = scenario_seconds(args.policy, case)
    sim_cmd = [py, os.path.join(HERE, "go2_mujoco_headless.py"), "--um", args.um, "--scene", args.scene, "--dt",
               str(args.dt), "--truth", truth, "--exit-idle", "3", "--duration", f"{seconds + 90:.0f}",
               "--seed", str(rep)] + case.sim
    dep_cmd = [py, os.path.join(HERE, "go2_deploy_sdk2.py"), "--policy", args.policy, "--tag", tag, "--log-dir",
               args.log_dir] + (["--profile", case.profile] if case.profile else []) + case.deploy
    dep_cmd += ["--rt"] if args.rt else []
    if case.rehearsal:
        sim_cmd += ["--robot-services", "--remote", case.scenario] + (["--profile", case.profile] if case.profile else [])
        dep_cmd += ["--rehearsal", "--duration", f"{seconds + 8:.0f}"]  # + the sport-mode release (about 4 s)
    else:
        dep_cmd += ["--sim", "--scenario", case.scenario, "--yes"]
    logf.write(f"\n##### {tag}\n$ {' '.join(sim_cmd)}\n$ {' '.join(dep_cmd)}\n")
    logf.flush()
    sim_out = open(os.path.join(args.out, f"{tag}_sim.txt"), "w")
    sim = subprocess.Popen(sim_cmd, stdout=sim_out, stderr=subprocess.STDOUT, cwd=HERE)
    try:
        t_end = time.monotonic() + 30.0  # wait until the simulator runs
        while time.monotonic() < t_end:
            time.sleep(0.5)
            with open(sim_out.name) as f:
                if "unitree_mujoco headless:" in f.read():
                    break
            if sim.poll() is not None:
                break
        if sim.poll() is not None:
            return {"error": f"simulator exited at start (see {sim_out.name})"}
        dep = subprocess.run(dep_cmd, input="\n", capture_output=True, text=True, cwd=HERE, timeout=seconds + 120)
        logf.write(dep.stdout[-4000:] + dep.stderr[-2000:])
        try:
            sim.wait(timeout=30)
        except subprocess.TimeoutExpired:
            sim.terminate()
            sim.wait(timeout=10)
    finally:
        if sim.poll() is None:
            sim.kill()
        sim_out.close()
    path = next((line.split("log saved to ", 1)[1].split()[0] for line in dep.stdout.splitlines()
                 if "log saved to " in line), None)
    if path is None or not os.path.exists(path):
        return {"error": f"no deploy log (exit code {dep.returncode}): {dep.stderr.strip()[-300:]}"}
    if not os.path.isabs(path):
        path = os.path.join(HERE, path)
    log = go2_log.load(path)
    run = A.Run(log, "A")
    if os.path.exists(truth):
        A.attach_truth([run], [go2_log.load(truth)])
    res = A.evaluate(run)
    res["driven_timing"] = driven_timing(run.c)
    res["events"] = list(log.events) + list(getattr(run, "sim_events", []))
    modes = run.mode_spans()
    res["final_mode"] = modes[-1][2] if modes else "?"
    res["path"] = path
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", default=None, help="exported policy folder (go2_export.py); not needed with --list")
    p.add_argument("--um", default=None, help="unitree_mujoco checkout; not needed with --list")
    p.add_argument("--scene", default=None, help="default: scene_flat.xml next to the Go2 scene.xml, else scene.xml")
    p.add_argument("--dt", type=float, default=0.002, help="physics step [s]")
    p.add_argument("--suite", default="gate", choices=SUITES + ("all",))
    p.add_argument("--only", default=None, help="comma-separated case names (any suite)")
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--rt", action="store_true", help="go2_deploy_sdk2.py --rt (real-time priority)")
    p.add_argument("--log-dir", default=os.path.join("logs", "deploy"))
    p.add_argument("--out", default=None, help="default: out/suite_<date>")
    p.add_argument("--list", action="store_true", help="print the cases and exit")
    args = p.parse_args()

    if args.only:
        names = [n.strip() for n in args.only.split(",")]
        unknown = [n for n in names if n not in {c.name for c in CASES}]
        if unknown:
            p.error(f"unknown case(s) {unknown}; --list shows them")
        cases = [c for c in CASES if c.name in names]
    else:
        cases = [c for c in CASES if args.suite == "all" or c.suite == args.suite]
    if args.list:
        print(f"{'case':16s} {'suite':10s} {'scenario':11s} {'profile':9s} {'expect':22s} simulator options / note")
        for c in cases:
            print(f"{c.name:16s} {c.suite:10s} {c.scenario:11s} {c.profile or '-':9s} {c.expect:22s} "
                  f"{' '.join(c.sim + c.deploy) or '-'}   ({c.note})")
        return
    if args.policy is None or args.um is None:
        p.error("--policy and --um are required to run cases")
    args.policy = os.path.abspath(args.policy)
    args.um = os.path.abspath(os.path.expanduser(args.um))
    if args.scene is None:
        flat = os.path.join(args.um, "unitree_robots", "go2", "scene_flat.xml")
        args.scene = flat if os.path.exists(flat) else os.path.join(args.um, "unitree_robots", "go2", "scene.xml")
    args.log_dir = os.path.abspath(args.log_dir)
    args.out = os.path.abspath(args.out or os.path.join("out", time.strftime("suite_%Y%m%d_%H%M%S")))
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    total = sum(scenario_seconds(args.policy, c) + 15 for c in cases) * args.repeat
    print(f"{len(cases)} cases x {args.repeat} runs, about {total / 60:.0f} min; scene {os.path.basename(args.scene)}, "
          f"physics {1000 * args.dt:g} ms; output in {args.out}")

    rows = []
    with open(os.path.join(args.out, "runs.log"), "w") as logf:
        for c in cases:
            for rep in range(args.repeat):
                t0 = time.monotonic()
                try:
                    res = run_case(c, rep, args, sys.executable, logf)
                except Exception as e:  # one broken case must not stop the battery
                    res = {"error": f"{type(e).__name__}: {e}"}
                if "error" in res:
                    ok, why, info = False, "ERROR " + res["error"], {}
                else:
                    ok, why = judge(c.expect, res)
                    info = summarize(res)
                rows.append({"case": c.name, "suite": c.suite, "run": rep + 1, "expect": c.expect,
                             "result": "ok" if ok else "NOT OK", "why": why,
                             "tracking": " ".join(info.get("tracking", [])),
                             "tilt_max_deg": f"{info.get('tilt_max', float('nan')):.0f}",
                             "tau_peak_pct": f"{info.get('tau_peak', float('nan')):.0f}",
                             "loop_p99_ms": f"{info.get('loop_p99', float('nan')):.1f}",
                             "state_age_max_ms": f"{info.get('age_max', float('nan')):.0f}",
                             "log": res.get("path", "")})
                print(f"{'ok    ' if ok else 'NOT OK'} {c.name:16s} run {rep + 1}  {why[:110]}"
                      f"  [{time.monotonic() - t0:.0f} s]", flush=True)

    keys = list(rows[0].keys())
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(f"# Simulation suite, {time.strftime('%Y-%m-%d %H:%M')}\n\npolicy `{args.policy}`, scene "
                f"`{os.path.basename(args.scene)}`, physics {1000 * args.dt:g} ms, {args.repeat} runs per case. "
                "Tracking = measured / commanded velocity per profile segment; loop and state age while the robot is "
                "driven (damping left out)\n\n")
        f.write("| case | run | expect | result | why | tracking (measured/command) | tilt max [deg] | "
                "torque peak [%] | loop p99 [ms] | state age max [ms] |\n|" + " --- |" * 10 + "\n")
        for r in rows:
            f.write(f"| {r['case']} | {r['run']} | {r['expect']} | {r['result']} | {r['why']} | {r['tracking']} | "
                    f"{r['tilt_max_deg']} | {r['tau_peak_pct']} | {r['loop_p99_ms']} | {r['state_age_max_ms']} |\n")
    bad = [r for r in rows if r["result"] != "ok"]
    print(f"\n{len(rows) - len(bad)}/{len(rows)} runs as expected. Summary: {os.path.join(args.out, 'summary.md')}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
