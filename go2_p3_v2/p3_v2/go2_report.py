"""
go2_report.py - [P3.4] The harness report: one command runs the standard evaluation set of go2_metrics.py on one
checkpoint and collects one summary (REPORT.md, summary.json and the plots) in logs/<exp>/report/. The same command then
measures every later candidate against the baseline (--ref).

    python go2_report.py -e go2-v1.0 --ckpt 1999                          # the standard set (GPU: est. 10-20 min)
    python go2_report.py -e go2-v1.0 --ckpt 1999 --gate ~/unitree_mujoco --deploy-python $DEPLOY_PY  # + the gate
    python go2_report.py -e go2-v1.0 --ckpt 1999 --collect --seeds go2-v1.0-s2 go2-v1.0-s3  # [P3v2] the seed band
    python go2_report.py -e go2-v2 --ckpt 2999 --ref go2-v1.0             # a candidate, side by side with v1.0
    python go2_report.py -e go2-v1.0 --ckpt 1999 --collect                # rebuild the report from existing CSVs

The standard set (WORLDS), one go2_metrics.py run each (a separate process, its log in report/logs/):
    flat         nominal world: no randomization, no noise, 20 ms latency           metrics_ckptN_seedS.csv
    flat_dr      the run's training randomization, noise and pushes (--keep_dr)       ..._dr.csv
    grid         the run's own tiles (the default grid for a flat-trained run)         ..._grid.csv
    grid_unseen  new random instances of the tiles (--terrain_set seed=1)              ..._grid_unseen.csv
    slope+10 ... slope-25   the command grid on a plane (forward = uphill for +)       ..._slope+10.csv ...
Headline numbers (summary.json "headline", the REPORT.md tables, the config card of go2_card.py):
    flat, flat_dr   per key command: falls, lin_err, yaw_err, vx_meas, CoT, slip_v; all bins: falls
    grid, unseen    per tile type: the frontier = the last level of the run of levels, from the lowest, whose
                    through[%] >= FRONTIER_THROUGH (80); through at the top level; falls. All tiles: mean through
    slopes          at (0.5, 0, 0): vx_meas, lin_err, through, falls, CoT, slip_v; standing (0, 0, 0): falls, creep
    gate            go2_sim_suite.py --suite gate in unitree_mujoco (needs go2_export.py first): runs as expected
--ref EXP adds a table "against EXP" (its report/summary.json) with the change of every headline number; --ref also
takes a frozen baseline folder (baselines/v1.0, from go2_card.py --freeze). No Genesis in this process: go2_metrics.py
runs as a subprocess, with its own Genesis.
[P3v2] From the P2 results (training noise far above evaluation noise; two conda environments):
    --seeds EXP ...   the other training seeds of this configuration (report them first): REPORT.md gets a seed band
                      (every headline number per run, mean, range), summary.json "band"; their configurations are
                      checked against this run's (cfgs.pkl and manifest.json arguments, except names and seed)
    --ref             with a band on either side, a verdict per number: better / worse only when the two sides' seed
                      ranges do not overlap, else "within noise"; one run on each side gets no verdict
    --deploy-python   the gate suite runs with the python of the unitree_mujoco + unitree_sdk2py environment (default:
                      $DEPLOY_PY), checked before the evaluation starts
"""

import argparse
import csv
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import time

from go2_provenance import now

HERE = os.path.dirname(os.path.abspath(__file__))
SLOPES = (10, -10, 20, -20, 25, -25)
WORLDS = {  # name: (go2_metrics.py arguments, file tag)
    "flat": ([], ""),
    "flat_dr": (["--keep_dr", "--tag", "dr"], "dr"),
    "grid": (["--terrain", "grid"], "grid"),
    "grid_unseen": (["--terrain", "grid", "--terrain_set", "seed=1", "--tag", "unseen"], "grid_unseen"),
    **{f"slope{d:+d}": (["--slope", str(d)], f"slope{d:+d}") for d in SLOPES},
}
FLAT_BINS = ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (-0.5, 0.0, 0.0), (0.0, 0.3, 0.0), (0.0, 0.0, 0.5))
FLAT_METRICS = ("lin_err[m/s]", "yaw_err[rad/s]", "vx_meas", "CoT[-]", "slip_v[cm/s]")
SLOPE_METRICS = ("vx_meas", "lin_err[m/s]", "through[%]", "CoT[-]", "slip_v[cm/s]")
FRONTIER_THROUGH = 80.0  # [%] a tile level counts as passed above this success rate
# headline numbers compared by --ref: (key, better) with better "lower", "higher" or "" (no direction)
HEADLINE_BETTER = {"falls": "lower", "lin_err": "lower", "yaw_err": "lower", "CoT": "lower", "slip_v": "lower",
                   "frontier": "higher", "through": "higher", "vx_meas": "", "as expected": "higher"}
# [P3v2] keys that may differ between the seeds of one configuration (go2_report.py --seeds checks the rest)
SEED_KEYS = ("train.run_name", "args.exp_name", "args.seed", "args.log_dir")


def number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def fmt(x, digits=3):
    if isinstance(x, str):
        return x
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}g}"


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def command(row):
    return tuple(round(number(row.get(k, 0.0)), 3) for k in ("vx", "vy", "wz"))


def at(rows, cmd):
    """The row of one command (dh = 0 for runs with the body-height command), or None."""
    for r in rows:
        dh = number(r["dh"]) if r.get("dh") not in (None, "") else 0.0
        if command(r) == tuple(round(c, 3) for c in cmd) and abs(dh) < 1e-9:
            return r
    return None


def falls(rows):
    a = b = 0
    for r in rows:
        x, _, y = (r.get("falls") or "0/0").partition("/")
        a, b = a + int(x or 0), b + int(y or 0)
    return [a, b]


def value(row, metric):
    return number(row.get(f"{metric}_mean")) if row else math.nan


# ---------------------------------------------------------------------------------------------------------------------
# One summary per world
# ---------------------------------------------------------------------------------------------------------------------
def summarize_flat(rows):
    bins = {}
    for cmd in FLAT_BINS:
        r = at(rows, cmd)
        if r is not None:
            bins[" ".join(f"{c:g}" for c in cmd)] = {"falls": r["falls"], **{m: value(r, m) for m in FLAT_METRICS}}
    return {"bins": bins, "falls": falls(rows)}


def summarize_grid(rows):
    types = {}
    for t in dict.fromkeys(r["terrain"] for r in rows):
        rs = sorted((r for r in rows if r["terrain"] == t), key=lambda r: int(r["level"]))
        levels = [int(r["level"]) for r in rs]
        through = [value(r, "through[%]") for r in rs]
        frontier = None
        for lvl, x in zip(levels, through):
            if not x >= FRONTIER_THROUGH:  # NaN counts as not passed
                break
            frontier = lvl
        types[t] = {"levels": levels, "through": through, "frontier": frontier, "top_level": levels[-1],
                    "top_through": through[-1], "falls": falls(rs)}
    finite = [x for t in types.values() for x in t["through"] if not math.isnan(x)]
    return {"types": types, "mean_through": sum(finite) / len(finite) if finite else math.nan, "falls": falls(rows)}


def summarize_slope(rows):
    walk, stand = at(rows, (0.5, 0.0, 0.0)), at(rows, (0.0, 0.0, 0.0))
    return {"walk_0.5": {"falls": walk["falls"] if walk else "n/a", **{m: value(walk, m) for m in SLOPE_METRICS}},
            "stand": {"falls": stand["falls"] if stand else "n/a", "vx_meas": value(stand, "vx_meas")},
            "falls": falls(rows)}


def summarize(world, rows):
    if world.startswith("grid"):
        return summarize_grid(rows)
    if world.startswith("slope"):
        return summarize_slope(rows)
    return summarize_flat(rows)


def gate_summary(path):
    """go2_sim_suite.py summary.csv -> {"as expected": "k/n", "not_ok": [case (why)], "file": path}."""
    rows = read_csv(path)
    bad = [f"{r['case']} run {r['run']}: {r['why']}" for r in rows if r.get("result") != "ok"]
    return {"as expected": f"{len(rows) - len(bad)}/{len(rows)}", "not_ok": bad, "file": path}


def headline(worlds, gate):
    """The numbers a baseline is judged and compared by: {key: value} (strings for counts "a/b")."""
    h = {}
    for name in ("flat", "flat_dr"):
        w = worlds.get(name)
        if w:
            h[f"{name}: falls (all bins)"] = "{}/{}".format(*w["falls"])
            b = w["bins"].get("0.5 0 0", {})
            h[f"{name}: lin_err @ 0.5 m/s [m/s]"] = b.get("lin_err[m/s]", math.nan)
            if name == "flat":
                h["flat: CoT @ 0.5 m/s"] = b.get("CoT[-]", math.nan)
                h["flat: slip_v @ 0.5 m/s [cm/s]"] = b.get("slip_v[cm/s]", math.nan)
                h["flat: yaw_err @ 0.5 rad/s [rad/s]"] = w["bins"].get("0 0 0.5", {}).get("yaw_err[rad/s]", math.nan)
    for name in ("grid", "grid_unseen"):
        w = worlds.get(name)
        if w:
            for t, d in w["types"].items():
                h[f"{name}: frontier {t}"] = d["frontier"] if d["frontier"] is not None else -1
            h[f"{name}: mean through [%]"] = w["mean_through"]
    for d in SLOPES:
        w = worlds.get(f"slope{d:+d}")
        if w:
            h[f"slope{d:+d}: vx_meas @ 0.5 m/s"] = w["walk_0.5"]["vx_meas"]
            h[f"slope{d:+d}: through @ 0.5 m/s [%]"] = w["walk_0.5"]["through[%]"]
            h[f"slope{d:+d}: falls (all bins)"] = "{}/{}".format(*w["falls"])
    if gate:
        h["gate: runs as expected"] = gate["as expected"]
    return h


def better(key):
    for word, direction in HEADLINE_BETTER.items():
        if word in key:
            return direction
    return ""


def ratio(text):
    a, _, b = str(text).partition("/")
    return number(a) / number(b) if number(b) else math.nan


def reps_text(reps):
    """20 -> "20"; [2, 20] (worlds measured with different --reps) -> "2 to 20 (worlds differ)"."""
    return f"{reps[0]} to {reps[-1]} (worlds differ)" if isinstance(reps, list) else str(reps)


def as_number(x):
    """Headline value -> float: a count "a/b" -> its share, None (n/a in summary.json) -> NaN."""
    if isinstance(x, str):
        return ratio(x)
    return math.nan if x is None else float(x)


def json_safe(x):
    """numpy -> Python, NaN -> None, recursively: summary.json stays strict JSON (readable by jq, JavaScript)."""
    if type(x).__module__ == "numpy":
        x = x.tolist()
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    return x


def find_summary(name):
    """[P3.4] A report's summary.json: a run name (logs/<name>), a run folder, a folder holding summary.json (a frozen
    baseline, e.g. baselines/v1.0) or the file itself; None when there is none."""
    return next((c for c in (name, os.path.join(name, "summary.json"), os.path.join(name, "report", "summary.json"),
                             os.path.join("logs", name, "report", "summary.json")) if os.path.isfile(c)), None)


def config_of(log_dir):
    """[P3v2] What a run trained with, flattened to {key: value}: its five configs (cfgs.pkl) and its arguments
    (manifest.json, P3.4 runs), without the keys that differ between the seeds of one configuration (SEED_KEYS)."""
    if not log_dir or not os.path.isfile(os.path.join(log_dir, "cfgs.pkl")):
        return None
    flat = {}

    def walk(prefix, x):
        if isinstance(x, dict) and x:
            for k, v in x.items():
                walk(f"{prefix}.{k}", v)
        else:
            flat[prefix] = x

    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
        for name, cfg in zip(("env", "obs", "reward", "command", "train"), pickle.load(f)):
            walk(name, cfg)
    if os.path.isfile(os.path.join(log_dir, "manifest.json")):
        with open(os.path.join(log_dir, "manifest.json")) as f:
            walk("args", json.load(f).get("args") or {})
    for k in SEED_KEYS:
        flat.pop(k, None)
    return flat


def run_folder(name):
    """The training folder of a --seeds entry (a run name or folder), for its configuration; None if not found."""
    for d in (name, os.path.join("logs", name)):
        if os.path.isfile(os.path.join(d, "cfgs.pkl")):
            return d
    return None


def band_of(summaries):
    """[P3v2] Seed band: {key: {"values", "mean", "min", "max", "share"}} over the runs, for every headline number they
    all have (a count "a/b" enters as its share)."""
    stats = {}
    for k, v0 in summaries[0]["headline"].items():
        if not all(k in s["headline"] for s in summaries):
            continue
        values = [as_number(s["headline"][k]) for s in summaries]
        ok = [v for v in values if not math.isnan(v)]
        stats[k] = {"values": values, "mean": sum(ok) / len(ok) if ok else math.nan,
                    "min": min(ok) if ok else math.nan, "max": max(ok) if ok else math.nan,
                    "share": isinstance(v0, str)}
    return stats


def side_of(summary, key):
    """{"mean", "min", "max", "n", "share"} of one headline number: the seed band when the report has one."""
    band = (summary.get("band") or {}).get("stats", {})
    if key in band:
        b = band[key]
        return {"mean": as_number(b["mean"]), "min": as_number(b["min"]), "max": as_number(b["max"]),
                "n": len(b["values"]), "share": b["share"]}
    v = summary["headline"].get(key)
    x = as_number(v)
    return {"mean": x, "min": x, "max": x, "n": 1, "share": isinstance(v, str), "raw": v}


def side_text(x):
    """A value, or a band "mean [min-max]"; shares in %."""
    if x["n"] == 1 and isinstance(x.get("raw"), str):
        return x["raw"]  # a count "a/b" of one run, as measured
    f = (lambda v: "n/a" if math.isnan(v) else f"{100 * v:.0f} %") if x["share"] else fmt
    return f(x["mean"]) if x["n"] == 1 else f"{f(x['mean'])} [{f(x['min'])}–{f(x['max'])}]"


def verdict(key, ref, cur):
    """[P3v2] better / worse only when the two sides' seed ranges do not overlap (every run of one side beyond every
    run of the other); "within noise" otherwise. One run on each side: no verdict (training noise unknown)."""
    if any(math.isnan(v) for v in (ref["min"], ref["max"], cur["min"], cur["max"])):
        return "n/a"
    if ref["n"] == 1 and cur["n"] == 1:
        return "–"
    side = "above" if cur["min"] > ref["max"] else "below" if cur["max"] < ref["min"] else None
    if side is None:
        return "within noise"
    direction = better(key)
    if not direction:
        return side
    return "better" if (side == "below") == (direction == "lower") else "worse"


# ---------------------------------------------------------------------------------------------------------------------
# REPORT.md
# ---------------------------------------------------------------------------------------------------------------------
def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def write_report(path, summary, ref=None):
    s, w = summary, summary["worlds"]
    lines = [f"# Harness report: {s['exp']}, checkpoint {s['ckpt']}", "",
             f"{s['created']} · evaluation seed {s['seed']} · {reps_text(s['reps'])} envs per bin · git commit "
             f"{s.get('git_commit') or 'n/a'} · `go2_report.py` (P3.4)", ""]
    for name, label in (("flat", "Flat ground, nominal world"), ("flat_dr", "Flat ground, training randomization")):
        if name in w:
            rows = [[k, b["falls"]] + [fmt(b[m]) for m in FLAT_METRICS] for k, b in w[name]["bins"].items()]
            lines += [f"## {label} (`{name}`)", "", table(["vx vy wz", "falls"] + list(FLAT_METRICS), rows),
                      "", f"All bins: {w[name]['falls'][0]}/{w[name]['falls'][1]} fallen.", ""]
    for name, label in (("grid", "Terrain grid, own tiles"), ("grid_unseen", "Terrain grid, unseen tiles")):
        if name in w:
            rows = [[t, d["frontier"] if d["frontier"] is not None else "below the lowest",
                     f"{fmt(d['top_through'])} (level {d['top_level']})", "{}/{}".format(*d["falls"]),
                     " ".join(fmt(x, 2) for x in d["through"])] for t, d in w[name]["types"].items()]
            lines += [f"## {label} (`{name}`)", "",
                      table(["type", f"frontier (through >= {FRONTIER_THROUGH:g} %)", "through at the top",
                             "falls", "through per level"], rows),
                      "", f"Mean through over all tiles: {fmt(w[name]['mean_through'])} %.", ""]
    slopes = [d for d in SLOPES if f"slope{d:+d}" in w]
    if slopes:
        rows = []
        for d in slopes:
            x = w[f"slope{d:+d}"]
            rows.append([f"{d:+d} deg", x["walk_0.5"]["falls"]] + [fmt(x["walk_0.5"][m]) for m in SLOPE_METRICS]
                        + [x["stand"]["falls"], fmt(x["stand"]["vx_meas"])])
        lines += ["## Slopes (`--slope`): walking at 0.5 m/s, and standing", "",
                  table(["slope", "falls"] + [f"{m} @0.5" for m in SLOPE_METRICS] + ["standing falls",
                                                                                    "standing creep vx [m/s]"], rows),
                  "", "+ = facing uphill, − = facing downhill.", ""]
    if s.get("gate"):
        g = s["gate"]
        lines += ["## unitree_mujoco gate suite (`go2_sim_suite.py --suite gate`)", "",
                  f"Runs as expected: **{g['as expected']}** (`{g['file']}`)."]
        lines += [f"- not as expected: {x}" for x in g["not_ok"]] + [""]
    if s.get("band"):  # [P3v2]
        b = s["band"]
        rows = [[k, *[side_text({"mean": v, "min": v, "max": v, "n": 1, "share": st["share"]}) for v in st["values"]],
                 side_text({"mean": st["mean"], "min": st["mean"], "max": st["mean"], "n": 1, "share": st["share"]}),
                 "n/a" if math.isnan(st["max"] - st["min"]) else
                 (f"{100 * (st['max'] - st['min']):.0f} % pts" if st["share"] else fmt(st["max"] - st["min"]))]
                for k, st in b["stats"].items()]
        lines += [f"## Seed band: {b['n']} runs of one configuration", "",
                  "Training seeds of the same configuration; their spread is the training noise every comparison is "
                  "judged against (P2: identical trainings differ far more than evaluation envs).", "",
                  table(["headline", *b["runs"], "mean", "range (max − min)"], rows), ""]
        if b.get("same_config") is False:
            lines += [f"**Not one configuration**: {b['config_differs']}", ""]
        if b.get("warnings"):
            lines += [f"- {w}" for w in b["warnings"]] + [""]
    if ref:
        rows = []
        for k in s["headline"]:
            if k not in ref["headline"]:
                continue
            a, c = side_of(ref, k), side_of(s, k)
            if math.isnan(a["mean"]) or math.isnan(c["mean"]):
                delta = "n/a"
            elif c["share"]:  # counts "a/b": the change of the share, in percentage points
                delta = f"{100 * (c['mean'] - a['mean']):+.0f} % pts"
            else:
                delta = f"{c['mean'] - a['mean']:+.3g}"
            rows.append([k, side_text(a), side_text(c), delta, verdict(k, a, c), better(k)])
        n_ref, n_cur = len((ref.get("band") or {}).get("runs", [1])), len((s.get("band") or {}).get("runs", [1]))
        lines += [f"## Against `{ref['exp']}` (checkpoint {ref['ckpt']})", "",
                  f"Seed bands: {n_ref} run(s) for `{ref['exp']}`, {n_cur} for `{s['exp']}` (values: mean "
                  f"[min–max]). A verdict needs the two ranges not to overlap.", "",
                  table(["headline", ref["exp"], s["exp"], "change of the means", "verdict", "better"], rows), ""]
        if n_ref == 1 and n_cur == 1:
            lines += ["One run on each side: the change includes the training noise, which P2 measured far above the "
                      "evaluation noise (e.g. lin_err at 0.5 m/s 0.060–0.085 m/s over 3 identical trainings). Give "
                      "both sides a seed band (`--seeds`) before reading a change as real.", ""]
    if s.get("plots"):
        lines += ["## Plots", ""] + [f"- `plots/{p}`" for p in s["plots"]] + [""]
    if s.get("missing"):
        lines += [f"Missing from the standard set: {', '.join(s['missing'])}.", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-e", "--exp_name", required=True)
    p.add_argument("--ckpt", type=int, required=True)
    p.add_argument("--log_dir", default=None, help="default: logs/<exp_name>")
    p.add_argument("--seed", type=int, default=0, help="go2_metrics.py --seed (evaluation seed)")
    p.add_argument("--reps", type=int, default=20, help="go2_metrics.py --reps (envs per bin)")
    p.add_argument("--worlds", nargs="+", default=list(WORLDS), choices=list(WORLDS), help="a subset of the set")
    p.add_argument("--cpu", action="store_true", help="go2_metrics.py --cpu (tests; the standard set wants a GPU)")
    p.add_argument("--grid-types", nargs="+", default=None, help="go2_metrics.py --types for the grid worlds (tests)")
    p.add_argument("--grid-levels", type=int, nargs="+", default=None, help="go2_metrics.py --levels (tests)")
    p.add_argument("--collect", action="store_true", help="no evaluation: build the report from the existing CSVs")
    p.add_argument("--skip-existing", action="store_true", help="evaluate only the worlds whose CSV is missing")
    p.add_argument("--gate", default=None, metavar="UNITREE_MUJOCO",
                   help="also run go2_sim_suite.py --suite gate with this unitree_mujoco checkout (export first)")
    p.add_argument("--gate-summary", default=None, help="use this go2_sim_suite.py summary.csv instead of running it")
    p.add_argument("--ref", default=None, metavar="EXP",
                   help="compare with this run's report (logs/EXP/report), or a frozen one (e.g. baselines/v1.0)")
    p.add_argument("--seeds", nargs="+", default=None, metavar="EXP",
                   help="[P3v2] other training seeds of this run's configuration (their reports first): the seed band")
    p.add_argument("--deploy-python", default=os.environ.get("DEPLOY_PY"), metavar="PYTHON",
                   help="[P3v2] the python of the unitree_mujoco + unitree_sdk2py environment, which runs the gate "
                        "suite (default: $DEPLOY_PY, else this python)")
    args = p.parse_args()

    # read and check first: a typo or a wrong environment must not wait for the whole evaluation
    ref = None
    if args.ref:
        ref_path = find_summary(args.ref)
        if ref_path is None:
            raise SystemExit(f"--ref: no summary.json for {args.ref} (run go2_report.py on that run first, or give "
                             f"the folder of a frozen baseline, e.g. baselines/v1.0)")
        with open(ref_path) as f:
            ref = json.load(f)
    seed_summaries = []
    for name in args.seeds or []:  # [P3v2]
        path = find_summary(name)
        if path is None:
            raise SystemExit(f"--seeds: no report for {name}: run go2_report.py -e {name} --ckpt ... first")
        with open(path) as f:
            seed_summaries.append((name, json.load(f)))
    log_dir = args.log_dir or os.path.join("logs", args.exp_name)
    deploy_python = args.deploy_python or sys.executable
    if args.gate:  # [P3v2] the gate suite runs in the unitree_mujoco environment (P2: two conda environments)
        if not os.path.isfile(os.path.join(log_dir, "exported", "policy.pt")):
            raise SystemExit(f"--gate: no export in {os.path.join(log_dir, 'exported')}: "
                             f"run go2_export.py -e {args.exp_name} --ckpt {args.ckpt} --onnx")
        if not os.path.isdir(os.path.expanduser(args.gate)):
            raise SystemExit(f"--gate: no unitree_mujoco checkout at {args.gate}")
        try:
            importable = subprocess.call([deploy_python, "-c", "import unitree_sdk2py"], stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL) == 0
        except OSError:
            importable = False
        if not importable:
            raise SystemExit(f"--gate: {deploy_python} cannot import unitree_sdk2py. The gate suite needs the python "
                             f"of your unitree_mujoco environment: --deploy-python /path/to/that/python, or "
                             f"DEPLOY_PY=... (as in p2_overnight.sh)")

    out = os.path.join(log_dir, "report")
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    os.makedirs(os.path.join(out, "plots"), exist_ok=True)
    stem = os.path.join(log_dir, f"metrics_ckpt{args.ckpt}_seed{args.seed}")
    csv_of = lambda world: stem + (f"_{WORLDS[world][1]}" if WORLDS[world][1] else "") + ".csv"

    t_start = time.monotonic()
    for world in args.worlds:
        if args.collect or (args.skip_existing and os.path.isfile(csv_of(world))):
            continue
        cmd = [sys.executable, os.path.join(HERE, "go2_metrics.py"), "-e", args.exp_name, "--ckpt", str(args.ckpt),
               "--seed", str(args.seed), "--reps", str(args.reps), "--log_dir", log_dir] + WORLDS[world][0]
        if args.cpu:
            cmd.append("--cpu")
        if world.startswith("grid"):
            cmd += (["--types", *args.grid_types] if args.grid_types else [])
            cmd += (["--levels", *map(str, args.grid_levels)] if args.grid_levels else [])
        t0 = time.monotonic()
        print(f"[{world}] {' '.join(cmd[1:])}", flush=True)
        with open(os.path.join(out, "logs", f"{world}.log"), "w") as logf:
            code = subprocess.call(cmd, stdout=logf, stderr=subprocess.STDOUT)
        print(f"[{world}] {'done' if code == 0 else f'FAILED (exit {code}), see report/logs/{world}.log'} in "
              f"{time.monotonic() - t0:.0f} s", flush=True)

    gate = None
    if args.gate:
        exported = os.path.join(log_dir, "exported")
        gate_dir = os.path.join(out, "gate")
        cmd = [deploy_python, os.path.join(HERE, "go2_sim_suite.py"), "--policy", exported, "--um", args.gate,
               "--suite", "gate", "--out", gate_dir]
        print(f"[gate] {' '.join(cmd)}", flush=True)
        code = subprocess.call(cmd)
        if code != 0:
            print(f"[gate] go2_sim_suite.py exited with {code}", flush=True)
        args.gate_summary = os.path.join(gate_dir, "summary.csv")
    if args.gate_summary:
        if not os.path.isfile(args.gate_summary):
            raise SystemExit(f"no gate summary at {args.gate_summary}")
        gate = gate_summary(args.gate_summary)

    worlds, missing, plots, reps = {}, [], [], set()
    for world in WORLDS:
        path = csv_of(world)
        if not os.path.isfile(path):
            missing.append(world)
            continue
        rows = read_csv(path)
        worlds[world] = summarize(world, rows)
        # envs per bin, as measured: the denominator of "falls" (a report --collect'ed from older CSVs keeps theirs)
        reps |= {int(number(str(r["falls"]).partition("/")[2])) for r in rows if "/" in str(r.get("falls", ""))}
        tag = WORLDS[world][1]
        name = f"ckpt{args.ckpt}_seed{args.seed}" + (f"_{tag}" if tag else "")
        for kind in ("terrain", "gait", "spectra"):
            src = os.path.join(log_dir, f"{kind}_{name}.png")
            if os.path.isfile(src):
                shutil.copy(src, os.path.join(out, "plots"))
                plots.append(os.path.basename(src))
    manifest = {}
    if os.path.isfile(os.path.join(log_dir, "manifest.json")):
        with open(os.path.join(log_dir, "manifest.json")) as f:
            manifest = json.load(f)
    reps = sorted(reps) if reps else [args.reps]
    summary = {"exp": args.exp_name, "ckpt": args.ckpt, "seed": args.seed,
               "reps": reps[0] if len(reps) == 1 else reps, "created": now(),
               "git_commit": manifest.get("git_hash"), "worlds": worlds, "gate": gate,
               "headline": headline(worlds, gate), "missing": missing, "plots": plots,
               "frontier_through": FRONTIER_THROUGH}
    if seed_summaries:  # [P3v2] the seed band: this run + the other seeds of its configuration
        mine, warnings, differs = config_of(log_dir), [], {}
        for name, other in seed_summaries:
            theirs = config_of(run_folder(name))
            if mine is None or theirs is None:
                warnings.append(f"{name}: configuration not checked (no cfgs.pkl found)")
            else:
                keys = sorted(k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k))
                if keys:
                    differs[name] = keys[:8] + ([f"... {len(keys) - 8} more"] if len(keys) > 8 else [])
            if (other["seed"], other["reps"]) != (summary["seed"], summary["reps"]):
                warnings.append(f"{name}: evaluated with seed {other['seed']} and {reps_text(other['reps'])} envs per "
                                f"bin, this run with {summary['seed']} and {reps_text(summary['reps'])}")
            if other["ckpt"] != summary["ckpt"]:
                warnings.append(f"{name}: checkpoint {other['ckpt']}, this run {summary['ckpt']}")
        summary["band"] = {"runs": [args.exp_name] + [n for n, _ in seed_summaries], "n": 1 + len(seed_summaries),
                           "same_config": not differs if not any("not checked" in w for w in warnings) else None,
                           "config_differs": differs, "warnings": warnings,
                           "stats": band_of([summary] + [o for _, o in seed_summaries])}
        for name, keys in differs.items():
            print(f"WARNING: {name} is not this run's configuration: {keys}")
        for w in warnings:
            print(f"WARNING: {w}")
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(json_safe(summary), f, indent=2, allow_nan=False)
    write_report(os.path.join(out, "REPORT.md"), summary, ref)
    print(f"\nreport: {os.path.join(out, 'REPORT.md')} and summary.json (worlds: {len(worlds)}"
          + (f"; missing: {', '.join(missing)}" if missing else "") + f"; {time.monotonic() - t_start:.0f} s)")
    for k, v in summary["headline"].items():
        print(f"  {k:42s} {fmt(v)}")


if __name__ == "__main__":
    main()
