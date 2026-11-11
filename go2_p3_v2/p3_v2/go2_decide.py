"""
go2_decide.py - [P3.2] [P3.3] A/B decisions from the harness, by rules fixed before the numbers are seen. No Genesis.

    python go2_decide.py slip   --off go2-p32 go2-p32-s2 --on go2-p32-slip go2-p32-slip-s2 --ckpt 1999
    python go2_decide.py critic --off go2-p32 go2-p32-s2 --on go2-p33 go2-p33-s2 --ckpt 1999

--off: the runs without the change (one per training seed), --on: the runs with it (same seeds). For every run the
go2_metrics.py CSVs of checkpoint --ckpt are read (evaluation seed --eval-seed): flat (metrics_ckptN_seedS.csv), the
terrain grid (_grid.csv) and every slope (_slope+15.csv ...), whichever exist; a world is used only if every run has it.
Each CSV is reduced to the mean of its MOVING bins (command speed >= 0.1 m/s or |wz| >= 0.1 rad/s), also per terrain
type on the grid. Run go2_metrics.py first (TEST_AND_TINKER_P3.md 5.4 and 6.4 have the commands).

The rules (constants below). Change them BEFORE looking at the numbers, and say so in --note:
slip    keep the penalty if, on every terrain world (the grid and each slope; flat if there is none):
          - slip_v drops by at least SLIP_DROP (30 %), and every --on seed is below every --off seed;
          - no foot's slip_v is worse (see "a cost" below; tolerance FOOT_WORSE, +10 %);
        and no harness cost: lin_err on flat (tolerance +5 % and +0.005 m/s), through[%] on any grid type or slope
        (5 points), CoT on flat (+5 %).
critic  keep the privileged critic unless it learns slower (the iteration at which Train/mean_reward first reaches
        90 % of the --off runs' final value, or the last iteration + 1 if never; tolerance +5 %), ends lower
        (Train/mean_reward over the last 5 % of the iterations, -2 %; Curriculum/terrain_level when logged, -0.2), or
        has a harness cost (as slip). Loss/value is shown, not judged (a critic with more inputs fits the returns
        better; that alone is not a better policy), and so is tracking at the commands P2 left weak (slow, lateral,
        turning in place), which a privileged critic may improve.
[P3v2] A cost is real only if it is beyond its tolerance AND every --on seed is worse than every --off seed (the two
arms' seed ranges do not overlap). P2 measured the training noise: between identical trainings lin_err at 0.5 m/s
varied 0.060-0.085 m/s and CoT at 1 m/s 0.595-0.706, far more than the 5 % tolerances. With no real effect, the arms
separate in a given direction by chance 1 / C(n_on + n_off, n_on): 1 in 6 with 2 + 2 seeds, 1 in 20 with 3 + 3.
Use 3 seeds per arm (TEST_AND_TINKER_P3.md 5.4); the record states the seed counts and this rate.
Prints the evidence and writes decisions/<rule>.md: the decision record (runs, rule, numbers, decision, the flags it
implies), which the P3.4 config card includes. Commit it: logs/ is git-ignored, decisions/ is not.
[P3.4] python go2_decide.py flags prints the flags the KEEP decisions imply (nothing for DROP), for the baseline run:
    V1FLAGS="--terrain --cmd-curriculum --set env.reset_forward=True $(python go2_decide.py flags)"
"""

import argparse
import csv
import datetime
import glob
import math
import os
import pickle
import re
import sys

# ---- pre-registered thresholds (TEST_AND_TINKER_P3.md 5.4 / 6.4)
SLIP_DROP = 0.30  # slip_v must drop by at least this share on every terrain world
FOOT_WORSE = 0.10  # ... and no foot may slip more than this share more
LIN_ERR_TOL = (0.05, 0.005)  # lin_err on flat: at most +5 % and +0.005 m/s
THROUGH_TOL = 5.0  # [points of %] through on any grid type or slope
COT_TOL = 0.05  # CoT on flat: at most +5 %
LEARN_REACH = 0.9  # critic: share of the --off final reward that defines "reached"
LEARN_TOL = 0.05  # critic: reached at most 5 % later
FINAL_TOL = 0.02  # critic: final reward at least --off - 2 %
LEVEL_TOL = 0.2  # critic: final terrain level at least --off - 0.2
MOVING = 0.1  # [m/s, rad/s] a bin is moving above this command
FEET = ("FL", "FR", "RL", "RR")


def number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def nanmean(values):
    values = [v for v in values if not math.isnan(v)]
    return sum(values) / len(values) if values else math.nan


def moving(row):
    vx, vy, wz = (number(row.get(k, 0.0)) for k in ("vx", "vy", "wz"))
    return math.hypot(vx, vy) >= MOVING or abs(wz) >= MOVING


def worlds_of(log_dir, ckpt, seed):
    """{world: rows} of one run: "flat", "grid" and "slope+15" ... (go2_metrics.py file names, no --tag)."""
    out = {}
    stem = os.path.join(log_dir, f"metrics_ckpt{ckpt}_seed{seed}")
    for path in [stem + ".csv", stem + "_grid.csv"] + sorted(glob.glob(stem + "_slope*.csv")):
        name = os.path.basename(path)[len(os.path.basename(stem)):-4].lstrip("_") or "flat"
        if name != "flat" and name != "grid" and not re.fullmatch(r"slope[+-]\d+(\.\d+)?", name):
            continue  # a --tag'ged evaluation (e.g. _slope+15_dr): not part of the standard set
        if os.path.isfile(path):
            with open(path, newline="") as f:
                out[name] = [r for r in csv.DictReader(f) if moving(r)]
    return out


def reduce(rows, metric, terrain=None):
    """Mean over the (moving) bins of one metric; terrain: only that grid type."""
    return nanmean([number(r.get(f"{metric}_mean")) for r in rows if terrain is None or r.get("terrain") == terrain])


def falls(rows):
    a = b = 0
    for r in rows:
        x, _, y = r.get("falls", "0/0").partition("/")
        a, b = a + int(x or 0), b + int(y or 0)
    return a, b


def arm(runs, world, metric, terrain=None):
    return [reduce(r[world], metric, terrain) for r in runs]


def rel(b, a):
    return (b - a) / abs(a) if a and not math.isnan(a) and not math.isnan(b) else math.nan


def all_worse(on, off, higher_is_worse=True):
    """[P3v2] Every --on seed worse than every --off seed: the two arms' seed ranges do not overlap. Without a real
    effect this happens by chance 1 / C(n_on + n_off, n_on) (chance())."""
    on, off = [v for v in on if not math.isnan(v)], [v for v in off if not math.isnan(v)]
    if not on or not off:
        return False
    return min(on) > max(off) if higher_is_worse else max(on) < min(off)


def chance(n_on, n_off):
    """[P3v2] Probability that the arms separate in one given direction by chance (all orders equally likely)."""
    return 1 / math.comb(n_on + n_off, n_on)


def pct(x):
    return "n/a" if math.isnan(x) else f"{100 * x:+.0f} %"


def fmt(values, digits=3):
    return " / ".join("n/a" if math.isnan(v) else f"{v:.{digits}g}" for v in values)


class Record:
    """Evidence lines and criteria, printed and written as markdown."""

    def __init__(self):
        self.rows, self.checks = [], []

    def evidence(self, world, metric, off, on, note=""):
        a, b = nanmean(off), nanmean(on)
        self.rows.append((world, metric, fmt(off), fmt(on), pct(rel(b, a)), note))

    def check(self, name, ok, detail):
        self.checks.append((name, bool(ok), detail))

    def passed(self):
        return all(ok for _, ok, _ in self.checks)


def harness_costs(rec, off, on, worlds):
    """The costs both rules share: tracking and energy on flat, getting through on the grid and the slopes."""
    if "flat" in worlds:
        a, b = arm(off, "flat", "lin_err[m/s]"), arm(on, "flat", "lin_err[m/s]")
        rec.evidence("flat", "lin_err[m/s]", a, b)
        limit = nanmean(a) * (1 + LIN_ERR_TOL[0]) + LIN_ERR_TOL[1]
        beyond, sep = nanmean(b) > limit, all_worse(b, a)
        rec.check("tracking", not (beyond and sep), f"lin_err on flat {nanmean(b):.4f} vs {nanmean(a):.4f} m/s; "
                  f"beyond {limit:.4f}: {beyond}, every --on seed worse: {sep}")
        a, b = arm(off, "flat", "CoT[-]"), arm(on, "flat", "CoT[-]")
        rec.evidence("flat", "CoT[-]", a, b)
        beyond, sep = nanmean(b) > nanmean(a) * (1 + COT_TOL), all_worse(b, a)
        rec.check("energy", math.isnan(nanmean(a)) or not (beyond and sep),
                  f"CoT on flat {pct(rel(nanmean(b), nanmean(a)))}; beyond +{100 * COT_TOL:.0f} %: {beyond}, "
                  f"every --on seed worse: {sep}")
    worst = []
    for world in worlds:
        if world == "flat":
            continue
        types = sorted({r.get("terrain") for r in off[0][world] if r.get("terrain")}) if world == "grid" else [None]
        for t in types:
            a, b = arm(off, world, "through[%]", t), arm(on, world, "through[%]", t)
            label = f"{world} {t}" if t else world
            rec.evidence(label, "through[%]", a, b, f"d {nanmean(b) - nanmean(a):+.1f} points")
            if not math.isnan(nanmean(a)):
                worst.append((nanmean(b) - nanmean(a), label, all_worse(b, a, higher_is_worse=False)))
        fa = [falls(r[world]) for r in off]
        fb = [falls(r[world]) for r in on]
        rec.rows.append((world, "falls", " / ".join(f"{x}/{y}" for x, y in fa), " / ".join(f"{x}/{y}" for x, y in fb),
                         "", ""))
    if worst:
        failing = [f"{label} {d:+.1f}" for d, label, sep in worst if d < -THROUGH_TOL and sep]
        d, label, sep = min(worst)
        rec.check("getting through", not failing,
                  f"worst through change {d:+.1f} points ({label}, every --on seed lower: {sep}), tolerance "
                  f"-{THROUGH_TOL:g}" + (f"; failing: {', '.join(failing)}" if failing else ""))


def rule_slip(off, on, worlds):
    rec = Record()
    terrain = [w for w in worlds if w != "flat"] or ["flat"]
    for world in terrain:
        a, b = arm(off, world, "slip_v[cm/s]"), arm(on, world, "slip_v[cm/s]")
        drop = -rel(nanmean(b), nanmean(a))
        rec.evidence(world, "slip_v[cm/s]", a, b)
        if math.isnan(drop) or any(map(math.isnan, a + b)):  # every robot fell, or no stance in the window
            rec.check(f"slip drops on {world}", False, "no slip data in some run (falls? see the falls rows)")
        else:
            consistent = max(b) < min(a)
            rec.check(f"slip drops on {world}", drop >= SLIP_DROP and consistent,
                      f"{100 * drop:.0f} % (needs {100 * SLIP_DROP:.0f} %), every --on seed below every --off seed: "
                      f"{consistent}")
        worse = []
        for foot in FEET:
            fa, fb = arm(off, world, f"slip_v_{foot}[cm/s]"), arm(on, world, f"slip_v_{foot}[cm/s]")
            rec.evidence(world, f"slip_v_{foot}[cm/s]", fa, fb)
            worse.append((rel(nanmean(fb), nanmean(fa)), foot, all_worse(fb, fa)))
        valid = [(c, f, sep) for c, f, sep in worse if not math.isnan(c)]
        if valid:
            change, foot, sep = max(valid)
            failing = [f for c, f, s in valid if c > FOOT_WORSE and s]
            rec.check(f"no foot worse on {world}", not failing,
                      f"largest per-foot change {pct(change)} ({foot}, every --on seed worse: {sep}), tolerance "
                      f"+{100 * FOOT_WORSE:.0f} %" + (f"; failing: {', '.join(failing)}" if failing else ""))
        else:
            rec.check(f"no foot worse on {world}", False, "no per-foot slip data (CSVs from before P3.2, or falls)")
        for metric in ("slip_share[%]", "pen_xy[cm/s]", "pen_t[cm/s]"):
            rec.evidence(world, metric, arm(off, world, metric), arm(on, world, metric))
        if world != "flat":  # [P3v2] P2.5: energy pulls swing height down; the slip penalty also prices foot motion
            for metric in ("stride[cm]", "swing_h[cm]"):
                rec.evidence(world, metric, arm(off, world, metric), arm(on, world, metric), "side effect, not judged")
    if "flat" in worlds:
        for metric in ("stride[cm]", "swing_h[cm]", "step_freq[Hz]", "slip_v[cm/s]"):
            rec.evidence("flat", metric, arm(off, "flat", metric), arm(on, "flat", metric), "side effect, not judged")
    harness_costs(rec, off, on, worlds)
    return rec


def scalars(log_dir, tag):
    """(steps, values) of one TensorBoard scalar of a run folder, or None."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return None
    events = acc.Scalars(tag)
    return [e.step for e in events], [e.value for e in events]


def tail_mean(values, share=0.05):
    n = max(10, int(len(values) * share))
    return sum(values[-n:]) / len(values[-n:])


def rule_critic(off_dirs, on_dirs, off, on, worlds):
    rec = Record()
    curves = {d: scalars(d, "Train/mean_reward") for d in off_dirs + on_dirs}
    missing = [d for d, c in curves.items() if c is None]
    if missing:
        raise SystemExit(f"no Train/mean_reward in the TensorBoard files of {missing}")
    final_a = [tail_mean(curves[d][1]) for d in off_dirs]
    final_b = [tail_mean(curves[d][1]) for d in on_dirs]
    target = LEARN_REACH * nanmean(final_a)

    def reach(d):  # a run that never reaches the target counts as its last iteration + 1 (slow, not ignored)
        steps, values = curves[d]
        return next((s for s, v in zip(steps, values) if v >= target), steps[-1] + 1)

    it_a, it_b = [reach(d) for d in off_dirs], [reach(d) for d in on_dirs]
    rec.evidence("training", f"iteration reaching {target:.3g}", it_a, it_b)
    beyond, sep = nanmean(it_b) > nanmean(it_a) * (1 + LEARN_TOL), all_worse(it_b, it_a)
    rec.check("learns at least as fast", not (beyond and sep),
              f"mean iteration {nanmean(it_b):.1f} vs {nanmean(it_a):.1f}; beyond +{100 * LEARN_TOL:.0f} %: {beyond}, "
              f"every --on seed later: {sep}")
    rec.evidence("training", "final Train/mean_reward", final_a, final_b)
    beyond, sep = nanmean(final_b) < nanmean(final_a) * (1 - FINAL_TOL), all_worse(final_b, final_a, False)
    rec.check("ends at least as high", not (beyond and sep),
              f"{nanmean(final_b):.3g} vs {nanmean(final_a):.3g}; beyond -{100 * FINAL_TOL:.0f} %: {beyond}, every "
              f"--on seed lower: {sep}")
    levels = {d: scalars(d, "Curriculum/terrain_level") for d in off_dirs + on_dirs}
    if all(levels.values()):
        la = [tail_mean(levels[d][1]) for d in off_dirs]
        lb = [tail_mean(levels[d][1]) for d in on_dirs]
        rec.evidence("training", "final Curriculum/terrain_level", la, lb)
        beyond, sep = nanmean(lb) < nanmean(la) - LEVEL_TOL, all_worse(lb, la, False)
        rec.check("terrain level at least as high", not (beyond and sep),
                  f"{nanmean(lb):.2f} vs {nanmean(la):.2f}; beyond -{LEVEL_TOL:g}: {beyond}, every --on seed lower: "
                  f"{sep}")
    values = {d: scalars(d, "Loss/value") for d in off_dirs + on_dirs}
    if all(values.values()):
        rec.evidence("training", "final Loss/value", [tail_mean(values[d][1]) for d in off_dirs],
                     [tail_mean(values[d][1]) for d in on_dirs], "shown, not judged")
    if "flat" in worlds:  # [P3v2] what P2 left weak (P2_SUMMARY.md): a privileged critic is one candidate fix
        slow = lambda r: math.hypot(number(r["vx"]), number(r["vy"])) <= 0.3 and abs(number(r["wz"])) < MOVING
        turn = lambda r: math.hypot(number(r["vx"]), number(r["vy"])) < MOVING
        for label, pred, metric in (("flat slow + lateral", slow, "lin_err[m/s]"),
                                    ("flat turning in place", turn, "yaw_err[rad/s]")):
            a = [nanmean([number(x.get(f"{metric}_mean")) for x in r["flat"] if pred(x)]) for r in off]
            b = [nanmean([number(x.get(f"{metric}_mean")) for x in r["flat"] if pred(x)]) for r in on]
            rec.evidence(label, metric, a, b, "weak after P2; shown, not judged")
    harness_costs(rec, off, on, worlds)
    return rec


def config_lines(rule, on_dirs):
    """What the --on runs changed, from their cfgs.pkl (the flags the decision implies)."""
    with open(os.path.join(on_dirs[0], "cfgs.pkl"), "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    if rule == "slip":
        w = reward_cfg["reward_scales"].get("foot_slip")
        mode = reward_cfg.get("foot_slip_mode", "xy")
        threshold = reward_cfg.get("contact_force_threshold", 1.0)
        return [f"foot_slip weight {w}, mode {mode}, contact threshold {threshold} N",
                f"flags: --foot-slip {abs(w) if w is not None else '?'}"
                + (f" --set reward.foot_slip_mode={mode}" if mode != "xy" else "")]
    groups = train_cfg.get("obs_groups", {})
    entries = obs_cfg.get("privileged") or []
    teacher = "privileged" in groups.get("actor", [])
    return [f"privileged entries {entries}", f"obs_groups {groups}",
            f"flags: --privileged {'teacher' if teacher else 'critic'}"
            + (f" --privileged-entries {' '.join(entries)}" if entries else "")]


def implied_flags(folder):
    """[P3.4] The go2_train.py flags of the KEEP decisions in folder/slip.md and folder/critic.md (nothing for DROP).
    Missing records are reported on stderr, so a baseline is never trained on an undecided question silently."""
    flags = []
    for rule in ("slip", "critic"):
        path = os.path.join(folder, f"{rule}.md")
        if not os.path.isfile(path):
            print(f"go2_decide.py flags: no {path} (decide it first: go2_decide.py {rule} ...)", file=sys.stderr)
            continue
        text = open(path).read()
        keep = re.search(r"^# Decision: .*-> KEEP", text, re.M)
        line = re.search(r"^- --on configuration: flags: (.*)$", text, re.M)
        print(f"go2_decide.py flags: {rule} {'KEEP' if keep else 'DROP'} ({path})", file=sys.stderr)
        if keep and line:
            flags.append(line.group(1).strip())
    return " ".join(flags)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("rule", choices=("slip", "critic", "flags"),
                   help="slip: P3.2 foot-slip penalty; critic: P3.3 privileged critic; [P3.4] flags: print the "
                        "go2_train.py flags the KEEP decisions of decisions/ imply")
    p.add_argument("--off", nargs="+", help="runs without the change (names under logs/, or folders)")
    p.add_argument("--on", nargs="+", help="runs with the change")
    p.add_argument("--ckpt", type=int)
    p.add_argument("--eval-seed", type=int, default=0, help="go2_metrics.py --seed of the CSVs")
    p.add_argument("--note", type=str, default="", help="free text for the record (e.g. a changed threshold)")
    p.add_argument("--out", type=str, default=None, help="default: decisions/<rule>.md")
    p.add_argument("--decisions", default="decisions", help="[P3.4] flags: the folder of the records")
    args = p.parse_args()
    if args.rule == "flags":  # [P3.4] V1FLAGS="... $(python go2_decide.py flags)"
        print(implied_flags(args.decisions))
        return
    if not (args.off and args.on and args.ckpt is not None):
        p.error(f"{args.rule}: --off, --on and --ckpt are required")

    def folder(run):  # a run name under logs/, or a run folder
        return run if os.path.isfile(os.path.join(run, "cfgs.pkl")) else os.path.join("logs", run)

    off_dirs, on_dirs = [folder(r) for r in args.off], [folder(r) for r in args.on]
    off = [worlds_of(d, args.ckpt, args.eval_seed) for d in off_dirs]
    on = [worlds_of(d, args.ckpt, args.eval_seed) for d in on_dirs]
    worlds = [w for w in ("flat", "grid") if all(w in r for r in off + on)]
    worlds += sorted({w for r in off + on for w in r if w.startswith("slope")} & set.intersection(
        *({w for w in r} for r in off + on)), key=lambda w: (abs(float(w[5:])), float(w[5:])))
    if not worlds:
        raise SystemExit("no go2_metrics.py CSV common to every run for this checkpoint: run go2_metrics.py first, "
                         f"e.g.\n  python go2_metrics.py -e {args.off[0]} --ckpt {args.ckpt}")
    rec = rule_slip(off, on, worlds) if args.rule == "slip" else rule_critic(off_dirs, on_dirs, off, on, worlds)
    decision = "KEEP" if rec.passed() else "DROP"
    title = {"slip": "P3.2 foot-slip penalty", "critic": "P3.3 privileged critic"}[args.rule]

    widths = [max(len(str(x)) for x in col) for col in zip(("world", "metric", "--off", "--on", "change", ""),
                                                          *rec.rows)]
    print(f"\n=== {title}: --off {' '.join(args.off)} | --on {' '.join(args.on)} | ckpt {args.ckpt} ===")
    print(f"worlds (moving bins): {', '.join(worlds)}")
    noise = (f"{len(args.on)} + {len(args.off)} seeds: with no real effect, the arms separate in a given direction by "
             f"chance 1 in {round(1 / chance(len(args.on), len(args.off)))} (a cost needs that AND its tolerance)")
    print(noise)
    if min(len(args.on), len(args.off)) < 3:
        print("note: fewer than 3 seeds per arm. P2 measured training noise far above the tolerances; a third seed "
              "makes each check far less likely to fire by chance (TEST_AND_TINKER_P3.md 5.4)")
    for row in [("world", "metric", "--off (per seed)", "--on (per seed)", "change", "")] + rec.rows:
        print("  ".join(str(x).ljust(w) for x, w in zip(row, widths)))
    print()
    for name, ok, detail in rec.checks:
        print(f"  [{'pass' if ok else 'FAIL'}] {name}: {detail}")
    implied = config_lines(args.rule, on_dirs)
    print(f"\ndecision: {decision} ({'all criteria pass' if decision == 'KEEP' else 'a criterion fails'})")
    for line in implied:
        print(f"  --on runs: {line}")

    out = args.out or os.path.join("decisions", f"{args.rule}.md")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write(f"# Decision: {title} -> {decision}\n\n")
        f.write(f"- date: {datetime.date.today().isoformat()}, checkpoint {args.ckpt}, evaluation seed "
                f"{args.eval_seed}\n- --off runs: {', '.join(args.off)}\n- --on runs: {', '.join(args.on)}\n")
        for line in implied:
            f.write(f"- --on configuration: {line}\n")
        f.write(f"- worlds (moving bins of the go2_metrics.py CSVs): {', '.join(worlds)}\n")
        f.write(f"- seeds: {noise}\n")
        f.write(f"- rule: go2_decide.py {args.rule} (thresholds in its header; SLIP_DROP {SLIP_DROP}, FOOT_WORSE "
                f"{FOOT_WORSE}, LIN_ERR_TOL {LIN_ERR_TOL}, THROUGH_TOL {THROUGH_TOL}, COT_TOL {COT_TOL}"
                + (f", LEARN_REACH {LEARN_REACH}, LEARN_TOL {LEARN_TOL}, FINAL_TOL {FINAL_TOL}, LEVEL_TOL {LEVEL_TOL}"
                   if args.rule == "critic" else "") + ")\n")
        if args.note:
            f.write(f"- note: {args.note}\n")
        f.write("\n| criterion | result | detail |\n|---|---|---|\n")
        for name, ok, detail in rec.checks:
            f.write(f"| {name} | {'pass' if ok else 'FAIL'} | {detail} |\n")
        f.write("\n| world | metric | --off (per seed) | --on (per seed) | change | |\n|---|---|---|---|---|---|\n")
        for row in rec.rows:
            f.write("| " + " | ".join(str(x) for x in row) + " |\n")
    print(f"decision record written to {out} (commit it; the P3.4 config card includes it)")


if __name__ == "__main__":
    main()
