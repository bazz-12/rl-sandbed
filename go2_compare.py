"""
go2_compare.py - [P1.6] Compare the go2_metrics.py tables of several runs, side by side. No Genesis needed.

    python go2_compare.py logs/go2-p15/metrics_ckpt999_seed0.csv logs/go2-p16/metrics_ckpt999_seed0.csv
    python go2_compare.py A.csv B.csv C.csv --cols lin_err act_acc "tau>10Hz" --rows "0 0 0" "0.5 0 0"
    python go2_compare.py A.csv B.csv --all                  # every metric
    python go2_compare.py A.csv B.csv --preset gait          # [P2] the gait-quality columns

The first file is the reference (A). For the others, each cell shows the value and its change from A: in % of A,
or as a difference "d..." when A is ~0 or the metric is compared to a command (measured velocities, height z).
"~" = within 2 standard errors of A (the difference may be noise; more --reps in go2_metrics.py narrows it).
Rows are matched on the command (vx, vy, wz[, dh]): a run without the body-height command matches the dh = 0 rows
of a run with it. --cols picks metrics by the start of their name (lin_err matches lin_err[m/s]).
Lower is better for most metrics, not all: vx_meas / vy_meas / wz_meas / z / speed are compared to the command,
and duty / step_freq / swing_h describe the gait (see TEST_AND_TINKER.md).
[P2] air_t / stride / swing_h are read against their targets (--air-time, --clearance), diag_sync is higher for a
trot, air_cv / air_asym / sym_err are lower for a regular, symmetric gait (see TEST_AND_TINKER_P2.md). Compare CSV
files written by the same go2_metrics.py version (P2 changed how power and CoT are measured).
[P3.1] go2_metrics.py --terrain grid CSVs have a terrain and a level column: rows are then matched on (terrain, level,
command), e.g. a flat-trained run and a terrain-trained run on the same tiles; through[%] and dist[m] are higher-is-
better. --preset terrain shows the terrain columns. --slope CSVs keep the command rows, so a flat CSV and slope CSVs
of one run compare row by row (A = flat, B = +10 deg, C = +20 deg ...).
    python go2_compare.py logs/go2-p20/metrics_ckpt1999_seed0_grid.csv logs/go2-p31/metrics_ckpt1999_seed0_grid.csv \\
        --preset terrain --rows "hill 9" "pit 9"
[P3.2] --preset slip: the foot-slip columns (slip_v of all feet, then of each foot: slip_v_FL ... slip_v_RR; slip_share,
slip_d, the penalty sums pen_xy / pen_t) next to the costs (lin_err, through, stride, swing_h, CoT). go2_decide.py
turns the same CSVs into the A/B decision.
"""

import argparse
import csv
import math

CMD_COLS = ("vx", "vy", "wz", "dh")
TILE_COLS = ("terrain", "level")  # [P3.1] go2_metrics.py --terrain grid
# [P3.2] "slip_v[" = all feet together (slip_v alone would also match the per-foot slip_v_FL ... columns)
DEFAULT_COLS = ("lin_err", "yaw_err", "z_err", "act_acc", "hf>10Hz", "tau>10Hz", "tau_peak", "CoT", "slip_v[")
# [P2] --preset gait: one column group per gait-quality part (P2.1 steps, P2.2 clearance, P2.3 symmetry, P2.4 pattern,
# P2.5 energy), plus tracking so that a gain is never read without its cost
PRESETS = {
    "default": DEFAULT_COLS,
    "gait": ("lin_err", "yaw_err", "air_t", "stride", "step_freq", "duty", "swing_h", "slip_v[", "sym_err",
             "air_asym", "diag_sync", "air_cv", "sched_match", "power", "CoT"),  # sched_match: go2_analyze.py CSVs
    # [P3.1] --terrain grid / --slope CSVs: did it get through, at what tracking, height, clearance and energy cost
    "terrain": ("through", "dist", "lin_err", "vx_meas", "z_err", "swing_h", "slip_v[", "nonfoot", "tau_peak", "CoT"),
    # [P3.2] the foot-slip decision: slip of all feet and of each foot (slip_v_FL ...), what the penalty sums, and the
    # costs it could have (tracking, getting through, gait, energy)
    "slip": ("slip_v", "slip_share[", "slip_d[", "pen_xy", "pen_t", "lin_err", "through", "stride", "swing_h", "CoT"),
}
SIGNED = ("vx_meas", "vy_meas", "wz_meas", "z[m]")  # compared to a command or target: differences, not %


def load(path):
    """-> {command key: row dict}, metric names (in file order), rows in file order."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path}: empty table")
    names = [c[: -len("_mean")] for c in rows[0] if c.endswith("_mean")]
    table = {}
    for r in rows:
        table[row_key(r)] = r
    return table, names


def command_key(values):
    values = [float(v) for v in values] + [0.0] * (4 - len(values))  # dh = 0 for 3-command runs
    return tuple(round(v, 3) + 0.0 for v in values[:4])


def row_key(r):
    """[P3.1] (terrain, level, vx, vy, wz, dh): terrain "" and level -1 for the CSVs without tiles (flat, slope)."""
    level = r.get("level")
    return (r.get("terrain") or "", int(level) if level not in (None, "") else -1) + \
        command_key([r.get(c) if r.get(c) not in (None, "") else 0.0 for c in CMD_COLS])


def parse_row(text):
    """--rows entry -> partial key: "VX VY WZ [DH]" (any tile), or "TERRAIN LEVEL [VX VY WZ [DH]]"."""
    parts = text.replace(",", " ").split()
    try:
        float(parts[0])
        return {"command": command_key(parts)}
    except ValueError:
        out = {"terrain": parts[0], "level": int(parts[1])}
        if len(parts) > 2:
            out["command"] = command_key(parts[2:])
        return out


def row_matches(key, want):
    return (want.get("terrain", key[0]) == key[0] and want.get("level", key[1]) == key[1]
            and want.get("command", key[2:]) == key[2:])


def number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def envs_in_bin(row):
    try:
        return int(row["falls"].split("/")[1])
    except (KeyError, IndexError, ValueError):
        return 0


def cell(name, ref, row):
    """'value (change)' of one metric of one run against the reference row."""
    if row is None:
        return "-"
    v = number(row.get(f"{name}_mean"))
    if math.isnan(v):
        return "n/a"
    text = f"{v:.3g}"
    if ref is None or ref is row:
        return text
    a = number(ref.get(f"{name}_mean"))
    if math.isnan(a):
        return text
    if abs(a) < 1e-9 or name.startswith(SIGNED):
        change = f"d{v - a:+.3g}"
    else:
        pct = 100.0 * (v - a) / abs(a)
        change = f"{pct:+.1f}%" if abs(pct) < 10 else f"{pct:+.0f}%"
    # 2 standard errors of the difference (std over the envs of the bin, n envs per bin)
    sa, sv = number(ref.get(f"{name}_std")), number(row.get(f"{name}_std"))
    na, nv = envs_in_bin(ref), envs_in_bin(row)
    noise = ""
    if na > 0 and nv > 0 and not (math.isnan(sa) or math.isnan(sv)):
        if abs(v - a) <= 2.0 * math.sqrt(sa * sa / na + sv * sv / nv):
            noise = "~"
    return f"{text} ({change}{noise})"


def print_table(title, header, lines):
    widths = [max(len(str(x)) for x in col) for col in zip(header, *lines)]
    print(f"\n--- {title} ---")
    print("  ".join(str(h).rjust(w) for h, w in zip(header, widths)))
    print("  ".join("-" * w for w in widths))
    for line in lines:
        print("  ".join(str(x).rjust(w) for x, w in zip(line, widths)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="+", help="metrics_ckpt<N>_seed<S>.csv files; the first one is the reference")
    p.add_argument("--cols", nargs="+", default=None, help="metrics to show, by the start of their name")
    p.add_argument("--all", action="store_true", help="show every metric")
    p.add_argument("--preset", choices=tuple(PRESETS), default="default",
                   help="[P2] column set when --cols is not given")
    p.add_argument("--rows", nargs="+", default=None, metavar='"VX VY WZ [DH]"', help="commands to show")
    args = p.parse_args()

    tables, names = [], []
    for path in args.csv:
        table, file_names = load(path)
        tables.append(table)
        names += [n for n in file_names if n not in names]
    letters = [chr(ord("A") + i) for i in range(len(tables))]
    for letter, path in zip(letters, args.csv):
        print(f"{letter} = {path}" + ("   (reference)" if letter == "A" else ""))

    if args.all:
        shown = names
    else:
        shown = []
        for pattern in args.cols or PRESETS[args.preset]:
            matches = [n for n in names if n.startswith(pattern)]
            if not matches and args.cols:
                raise SystemExit(f"--cols {pattern}: no such metric. Available: {' '.join(names)}")
            shown += [n for n in matches if n not in shown]

    # the reference's rows, then the rows only the other tables have
    keys = list(tables[0])
    for t in tables[1:]:
        keys += [k for k in t if k not in keys]
    if args.rows:  # [P3.1] "VX VY WZ [DH]" or "TERRAIN LEVEL [VX VY WZ [DH]]", in the order given
        wants = [parse_row(r) for r in args.rows]
        keys = [k for w in wants for k in keys if row_matches(k, w)] or [
            ("", -1) + w["command"] for w in wants if "command" in w]
    with_dh = any(k[5] != 0.0 for k in keys) or any("dh" in next(iter(t.values())) for t in tables)
    with_tiles = any(k[0] for k in keys)  # [P3.1]
    cmd_names = list(TILE_COLS if with_tiles else ()) + list(CMD_COLS if with_dh else CMD_COLS[:3])

    def command_cells(key):
        ood = any(t.get(key, {}).get("ood") == "True" for t in tables)
        tile = [key[0] or "-", str(key[1]) if key[1] >= 0 else "-"] if with_tiles else []
        return tile + [f"{v:.2f}" for v in key[2: 2 + len(cmd_names) - len(tile)]] + ["*" if ood else ""]

    header = cmd_names + ["ood"] + letters
    lines = [command_cells(k) + [t[k]["falls"] if k in t else "-" for t in tables] for k in keys]
    print_table("falls (fallen envs / envs)", header, lines)
    for name in shown:
        lines = []
        for k in keys:
            ref = tables[0].get(k)
            lines.append(command_cells(k) + [cell(name, ref, t.get(k)) for t in tables])
        print_table(name, header, lines)
    print("\n(change vs A in %, or d = difference; ~ = within 2 standard errors of A; - = no such row; "
          "* = command outside a training range)")


if __name__ == "__main__":
    main()
