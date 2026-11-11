"""
go2_compare.py - [P1.6] Compare the go2_metrics.py tables of several runs, side by side. No Genesis needed.

    python go2_compare.py logs/go2-p15/metrics_ckpt999_seed0.csv logs/go2-p16/metrics_ckpt999_seed0.csv
    python go2_compare.py A.csv B.csv C.csv --cols lin_err act_acc "tau>10Hz" --rows "0 0 0" "0.5 0 0"
    python go2_compare.py A.csv B.csv --all                  # every metric

The first file is the reference (A). For the others, each cell shows the value and its change from A: in % of A,
or as a difference "d..." when A is ~0 or the metric is compared to a command (measured velocities, height z).
"~" = within 2 standard errors of A (the difference may be noise; more --reps in go2_metrics.py narrows it).
Rows are matched on the command (vx, vy, wz[, dh]): a run without the body-height command matches the dh = 0 rows
of a run with it. --cols picks metrics by the start of their name (lin_err matches lin_err[m/s]).
Lower is better for most metrics, not all: vx_meas / vy_meas / wz_meas / z / speed are compared to the command,
and duty / step_freq / swing_h describe the gait (see TEST_AND_TINKER.md).
"""

import argparse
import csv
import math

CMD_COLS = ("vx", "vy", "wz", "dh")
DEFAULT_COLS = ("lin_err", "yaw_err", "z_err", "act_acc", "hf>10Hz", "tau>10Hz", "tau_peak", "CoT", "slip_v")
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
        table[command_key([r.get(c, 0.0) for c in CMD_COLS])] = r
    return table, names


def command_key(values):
    values = [float(v) for v in values] + [0.0] * (4 - len(values))  # dh = 0 for 3-command runs
    return tuple(round(v, 3) + 0.0 for v in values[:4])


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
        for pattern in args.cols or DEFAULT_COLS:
            matches = [n for n in names if n.startswith(pattern)]
            if not matches and args.cols:
                raise SystemExit(f"--cols {pattern}: no such metric. Available: {' '.join(names)}")
            shown += [n for n in matches if n not in shown]

    if args.rows:
        keys = [command_key(r.replace(",", " ").split()) for r in args.rows]
    else:  # the reference's rows, then the rows only the other tables have
        keys = list(tables[0])
        for t in tables[1:]:
            keys += [k for k in t if k not in keys]
    with_dh = any(k[3] != 0.0 for k in keys) or any("dh" in next(iter(t.values())) for t in tables)
    cmd_names = list(CMD_COLS if with_dh else CMD_COLS[:3])

    def command_cells(key):
        ood = any(t.get(key, {}).get("ood") == "True" for t in tables)
        return [f"{v:.2f}" for v in key[: len(cmd_names)]] + ["*" if ood else ""]

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
