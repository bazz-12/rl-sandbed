"""
go2_card.py - [P3.4] The one-page config card of a run, logs/<exp>/CONFIG_CARD.md: what the policy is, how it was
trained, how well it does, which files hold it, and why each P3 choice was made. It reads the run's own files (no
Genesis): cfgs.pkl, manifest.json, model_<ckpt>.pt, exported/ (deploy_cfg.json, export_check.json, the policy files),
report/summary.json (go2_report.py) and the decision records decisions/slip.md and decisions/critic.md (go2_decide.py).

    python go2_card.py -e go2-v1.0 --ckpt 1999 --name "Baseline v1.0"           # -> logs/go2-v1.0/CONFIG_CARD.md
    python go2_card.py -e go2-v1.0 --ckpt 1999 --name "Baseline v1.0" --freeze baselines/v1.0 --strict
                                                    # + the git-tracked copy; --strict: exit 1 unless every line is ok

The card opens with a freeze checklist; each line is "ok" or "NO" with the reason:
    code committed          the run's manifest has a git commit, and at training time the working tree was clean
                            and held no untracked .py file
    code unchanged          every .py file of this folder still has the fingerprint the run recorded
                            (go2_provenance.py)
    export checked          exported/ is from this checkpoint, passed every check of go2_export.py (TorchScript and
                            ONNX), and its files still have the recorded sha256
    report complete         report/summary.json is from this checkpoint, has every world of the standard set, every
                            tile type and level the run trained on, and at least REPORT_MIN_REPS (20) envs per bin
    seed band               [P3v2] the report holds a seed band (go2_report.py --seeds) of at least BAND_MIN_RUNS (3)
                            training seeds of one configuration
    decision slip           decisions/slip.md exists and the run follows it (foot-slip weight and mode)
    decision critic         decisions/critic.md exists and the run follows it (privileged entries)
    reset_forward           on (TEST_AND_TINKER_P3.md 4.1.8)
--freeze DIR copies the card and the small records (deploy_cfg.json, export_check.json, REPORT.md, summary.json, the
decisions, manifest.json, cfgs.pkl and cfgs.json) into DIR, a folder git tracks, and prints the git and gh commands of
the tag and the release (policy.pt, policy.onnx, deploy_cfg.json, the checkpoint and terrain.npz go to the release:
logs/ and *.pt are git-ignored). --strict makes any "NO" an error (exit 1), and then --freeze copies nothing.
"""

import argparse
import json
import math
import os
import pickle
import re
import shutil

from go2_provenance import compare_code, now, sha256_file

STANDARD_WORLDS = ("flat", "flat_dr", "grid", "grid_unseen", "slope+10", "slope-10", "slope+20", "slope-20",
                   "slope+25", "slope-25")  # go2_report.WORLDS
RELEASE_FILES = ("exported/policy.pt", "exported/policy.onnx", "exported/deploy_cfg.json", "terrain.npz")
REPORT_MIN_REPS = 20  # envs per bin a baseline report needs (go2_report.py --reps, default 20)
BAND_MIN_RUNS = 3  # [P3v2] training seeds in the report's seed band (go2_report.py --seeds): the baseline's own noise
BAND_KEYS = ("flat: lin_err @ 0.5 m/s [m/s]", "flat: CoT @ 0.5 m/s", "flat_dr: falls (all bins)",
             "grid: mean through [%]", "grid_unseen: mean through [%]")  # [P3v2] shown with their band on the card


def load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt(x, digits=3):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return "n/a" if math.isnan(x) else f"{x:.{digits}g}"
    return str(x)


def short(digest):
    return digest[:16] if digest else "n/a"


# ---------------------------------------------------------------------------------------------------------------------
# Decision records (go2_decide.py)
# ---------------------------------------------------------------------------------------------------------------------
def read_decision(path):
    """decisions/<rule>.md -> {"decision": KEEP/DROP, "flags": "...", "date", "runs", "criteria": [(name, ok)]}."""
    if not os.path.isfile(path):
        return None
    text = open(path).read()
    out = {"path": path, "criteria": []}
    m = re.search(r"^# Decision: .*-> (KEEP|DROP)", text, re.M)
    out["decision"] = m.group(1) if m else "?"
    m = re.search(r"^- --on configuration: flags: (.*)$", text, re.M)
    out["flags"] = m.group(1).strip() if m else ""
    m = re.search(r"^- date: ([0-9-]+), checkpoint (\d+)", text, re.M)
    out["date"], out["ckpt"] = (m.group(1), m.group(2)) if m else ("?", "?")
    for key in ("off", "on"):
        m = re.search(rf"^- --{key} runs: (.*)$", text, re.M)
        out[key] = m.group(1).strip() if m else "?"
    for name, result in re.findall(r"^\| ([^|]+?) \| (pass|FAIL) \|", text, re.M):
        out["criteria"].append((name, result == "pass"))
    return out


def decision_matches(rule, d, reward_cfg, obs_cfg, train_cfg):
    """Does the run's configuration follow the decision? -> (ok, reason)."""
    if d is None:
        return False, "no record"
    if rule == "slip":
        weight = reward_cfg["reward_scales"].get("foot_slip")
        mode = reward_cfg.get("foot_slip_mode", "xy")
        if d["decision"] == "DROP":
            return weight is None, "penalty off, as decided" if weight is None else f"the run has foot_slip {weight}"
        m = re.search(r"--foot-slip\s+([0-9.eE+-]+)", d["flags"])
        want_w = -abs(float(m.group(1))) if m else None
        m = re.search(r"foot_slip_mode=(\w+)", d["flags"])
        want_mode = m.group(1) if m else "xy"
        ok = weight is not None and want_w is not None and abs(weight - want_w) < 1e-12 and mode == want_mode
        if ok:
            return ok, f"the run follows it: foot_slip {weight}, mode {mode}"
        return ok, f"run: foot_slip {weight}, mode {mode}; decision: {want_w}, mode {want_mode}"
    groups = train_cfg.get("obs_groups", {})
    entries = list(obs_cfg.get("privileged") or [])
    critic = "privileged" in groups.get("critic", []) and "privileged" not in groups.get("actor", [])
    if d["decision"] == "DROP":
        return not entries, "no privileged group, as decided" if not entries else f"the run has {entries}"
    m = re.search(r"--privileged-entries\s+(.*)$", d["flags"])
    want = m.group(1).split() if m else []
    ok = critic and entries == want
    if ok:
        return ok, f"the run follows it: the critic reads {len(entries)} entries ({', '.join(entries)})"
    return ok, (f"run: critic reads {entries}" if critic else "run: the critic does not read the privileged "
                "group") + f"; decision: {want}"


# ---------------------------------------------------------------------------------------------------------------------
def switches(env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg):
    """The flags a configuration implies (for runs whose manifest has no command line)."""
    scales, out = reward_cfg["reward_scales"], []
    if env_cfg.get("terrain"):
        out.append("--terrain")
    if command_cfg.get("curriculum"):
        out.append("--cmd-curriculum")
    if "foot_slip" in scales:
        out.append(f"--foot-slip {abs(scales['foot_slip']):g}")
    groups = train_cfg.get("obs_groups", {})
    if obs_cfg.get("privileged"):
        out.append(f"--privileged {'teacher' if 'privileged' in groups.get('actor', []) else 'critic'}")
    sym = train_cfg["algorithm"].get("symmetry_cfg")
    if sym:
        out.append("--symmetry " + ("both" if sym["use_data_augmentation"] and sym["use_mirror_loss"] else
                                    "aug" if sym["use_data_augmentation"] else "loss" if sym["use_mirror_loss"] else
                                    "log"))
    if "gait" in scales:
        out.append("--gait spot")
    if env_cfg.get("gait_clock"):
        out.append("--gait wtw")
    if "energy" in scales:
        out.append(f"--energy {abs(scales['energy']):g}")
    if (env_cfg.get("recovery") or {}).get("init_prob", 0) > 0:
        out.append("--recovery")
    if command_cfg.get("num_commands", 3) == 4:
        out.append("--height-cmd")
    if train_cfg["actor"].get("class_name") == "RNNModel":
        out.append(f"--rnn {train_cfg['actor'].get('rnn_type')}")
    if "SmoothPPO" in train_cfg["algorithm"].get("class_name", ""):
        a = train_cfg["algorithm"]
        out.append(f"--lcp {a.get('grad_penalty_coef')} --caps-t {a.get('caps_temporal_coef')} "
                   f"--caps-s {a.get('caps_spatial_coef')}")
    if env_cfg.get("reset_forward") is not None:
        out.append(f"--set env.reset_forward={env_cfg['reset_forward']}")
    return out


def reps_text(reps):
    """Envs per bin of a report: 20, or [2, 20] when its worlds were measured with different --reps."""
    return f"{reps[0]} to {reps[-1]} (worlds differ)" if isinstance(reps, list) else str(reps)


def fmt_dict(d):
    """{'hip': 23.7, 'calf': 45.43} -> 'hip 23.7, calf 45.43' (other values unchanged)."""
    return ", ".join(f"{k} {v}" for k, v in d.items()) if isinstance(d, dict) else d


def param_count(state_dict):
    return sum(v.numel() for v in state_dict.values() if hasattr(v, "numel"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-e", "--exp_name", required=True)
    p.add_argument("--ckpt", type=int, required=True)
    p.add_argument("--log_dir", default=None, help="default: logs/<exp_name>")
    p.add_argument("--name", default=None, help='e.g. "Baseline v1.0" (title of the card, and of the tag)')
    p.add_argument("--decisions", default="decisions", help="folder of the go2_decide.py records")
    p.add_argument("--freeze", default=None, metavar="DIR", help="copy the card and records here (tracked by git)")
    p.add_argument("--tag", default=None, help="git tag for the printed commands (default: from --name)")
    p.add_argument("--strict", action="store_true",
                   help="exit with an error when a checklist line is NO (and then freeze nothing)")
    args = p.parse_args()

    log_dir = args.log_dir or os.path.join("logs", args.exp_name)
    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    manifest = load_json(os.path.join(log_dir, "manifest.json")) or {}
    import torch  # only to read the checkpoint

    ckpt_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    infos = ckpt.get("infos") or {}
    curriculum = infos.get("curriculum") or {}
    deploy = load_json(os.path.join(log_dir, "exported", "deploy_cfg.json"))
    check = load_json(os.path.join(log_dir, "exported", "export_check.json"))
    report = load_json(os.path.join(log_dir, "report", "summary.json"))
    decisions = {rule: read_decision(os.path.join(args.decisions, f"{rule}.md")) for rule in ("slip", "critic")}
    title = args.name or args.exp_name

    # --- freeze checklist
    items = []
    # untracked .py files fail too: a checkout of the commit would lack them (commit them, or list them in .gitignore)
    items.append(("code committed", bool(manifest.get("git_hash")) and manifest.get("git_dirty") is False
                  and not manifest.get("git_untracked_py"),
                  f"commit {manifest.get('git_hash') or 'unknown'}, dirty {manifest.get('git_dirty')}"
                  + (f", untracked .py {manifest['git_untracked_py']}" if manifest.get("git_untracked_py") else "")))
    diff = compare_code(manifest)
    if diff is None:
        items.append(("code unchanged", False, "no code fingerprint in manifest.json (trained before P3.4)"))
    else:
        changed, added, removed = diff
        items.append(("code unchanged", not (changed or removed),
                      "every .py file as at training" if not (changed or removed) else
                      f"changed since training: {changed + removed}"))
    if check is None:
        items.append(("export checked", False, "no exported/export_check.json: run go2_export.py --onnx"))
    else:
        exported = os.path.join(log_dir, "exported")
        intact = all(os.path.isfile(os.path.join(exported, n)) and sha256_file(os.path.join(exported, n)) == h
                     for n, h in check["sha256"].items())
        ok = check.get("ckpt") == args.ckpt and check.get("passed") and intact and "policy.onnx" in check["sha256"]
        worst = max(e for c in check["max_abs_error"].values() for e in c.values())
        items.append(("export checked", ok, (f"the export is of checkpoint {check.get('ckpt')}; "
                                             if check.get("ckpt") != args.ckpt else "")
                      + f"worst |export - policy| {worst:.1e} (tolerance {check['tolerance']:g}), passed "
                      f"{check.get('passed')}, files intact {intact}, ONNX {'policy.onnx' in check['sha256']}"))
    if report is None:
        items.append(("report complete", False, "no report/summary.json: run go2_report.py"))
    else:
        gaps = [] if report["ckpt"] == args.ckpt else [f"the report is of checkpoint {report['ckpt']}"]
        missing = [w for w in STANDARD_WORLDS if w not in report["worlds"]]
        if missing:
            gaps.append(f"missing {missing}")
        reps = report["reps"]
        if (min(reps) if isinstance(reps, list) else reps) < REPORT_MIN_REPS:
            gaps.append(f"below the {REPORT_MIN_REPS} envs per bin a baseline needs")
        t = curriculum.get("terrain")
        for name in ("grid", "grid_unseen"):  # every tile type and level the run trained on (no --grid-types subset)
            g = report["worlds"].get(name)
            if t and g:
                part = [ty for ty in t["types"] if len(g["types"].get(ty, {}).get("levels", [])) < t["num_levels"]]
                if part:
                    gaps.append(f"{name} lacks levels of {part}")
        items.append(("report complete", not gaps, f"{reps_text(reps)} envs per bin; "
                      + ("; ".join(gaps) if gaps else "every world, tile type and level of the standard set")))
    band = (report or {}).get("band")  # [P3v2] P2: one run against one run is not a result
    if not band:
        items.append(("seed band", False, f"none in the report: report {BAND_MIN_RUNS - 1} more training seeds of "
                                          f"this configuration, then go2_report.py ... --seeds EXP-s2 EXP-s3"))
    else:
        same = band.get("same_config")
        items.append(("seed band", band["n"] >= BAND_MIN_RUNS and same is True,
                      f"{band['n']} runs ({', '.join(band['runs'])}), "
                      + ("one configuration" if same else "configuration not checked" if same is None else
                         f"NOT one configuration: {band['config_differs']}")
                      + ("" if band["n"] >= BAND_MIN_RUNS else f"; fewer than {BAND_MIN_RUNS}")))
    for rule in ("slip", "critic"):
        ok, why = decision_matches(rule, decisions[rule], reward_cfg, obs_cfg, train_cfg)
        d = decisions[rule]
        if d is None:
            why = f"no {os.path.join(args.decisions, rule + '.md')}: decide it first (go2_decide.py {rule} ...)"
        items.append((f"decision {rule}", ok, (f"{d['decision']} ({d['path']}); " if d else "") + why))
    items.append(("reset_forward", env_cfg.get("reset_forward") is True,
                  f"env.reset_forward = {env_cfg.get('reset_forward')}"))

    # --- the card
    L = [f"# Config card: {title}", "",
         f"Run `{args.exp_name}`, checkpoint {args.ckpt} (iteration {ckpt.get('iter')}). Card written {now()} by "
         f"`go2_card.py` (P3.4).", "", "## Freeze checklist", ""]
    L += [f"- **{'ok' if ok else 'NO'}** {name}: {why}" for name, ok, why in items]
    args_m = manifest.get("args") or {}
    L += ["", "## Identity and training", "",
          f"- Code: commit `{manifest.get('git_hash') or 'unknown'}`"
          + (" (dirty: see git_diff.patch)" if manifest.get("git_dirty") else "")
          + f"; trained {manifest.get('created', 'n/a')} on {manifest.get('host', 'n/a')}"
          + (f", {manifest['gpu']}" if manifest.get("gpu") else ""),
          f"- Versions: python {manifest.get('python')}, torch {manifest.get('torch')} (CUDA {manifest.get('cuda')}), "
          f"Genesis {manifest.get('genesis')}, rsl-rl-lib {manifest.get('rsl_rl')}",
          "- Command: `" + (" ".join(["python"] + manifest["argv"]) if manifest.get("argv") else
                            "unknown (manifest before P3.4); switches: " + " ".join(
                                switches(env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg))) + "`",
          (f"- PPO: {args_m['num_envs']} envs × {train_cfg['num_steps_per_env']} steps, "
           f"{args_m['max_iterations']} iteration(s), seed {args_m['seed']}; lr " if args_m else
           f"- PPO: {train_cfg['num_steps_per_env']} steps per env and iteration (envs, iterations, seed: "
           f"not in a manifest before P3.4); lr ")
          + f"{train_cfg['algorithm']['learning_rate']} ({train_cfg['algorithm']['schedule']}, KL "
          f"{train_cfg['algorithm']['desired_kl']}), clip {train_cfg['algorithm']['clip_param']}, entropy "
          f"{train_cfg['algorithm']['entropy_coef']}, γ {train_cfg['algorithm']['gamma']}, λ "
          f"{train_cfg['algorithm']['lam']}, {train_cfg['algorithm']['num_learning_epochs']} epochs × "
          f"{train_cfg['algorithm']['num_mini_batches']} mini-batches"]
    actor, critic = train_cfg["actor"], train_cfg["critic"]
    groups = train_cfg.get("obs_groups", {})
    L += [f"- Networks: actor {actor['class_name']} {actor.get('hidden_dims')} {actor.get('activation')}"
          + (f" + {actor.get('rnn_type')} {actor.get('rnn_hidden_dim')}" if actor.get("rnn_type") else "")
          + f" reads {groups.get('actor')} ({param_count(ckpt['actor_state_dict']):,} parameters); critic "
          f"{critic.get('hidden_dims')} reads {groups.get('critic')} ({param_count(ckpt['critic_state_dict']):,})"
          + (f"; privileged entries {obs_cfg['privileged']}" if obs_cfg.get("privileged") else "")]
    dr = env_cfg.get("domain_rand", {})
    sim_dt, decimation = env_cfg.get("sim_dt", 0.005), env_cfg.get("decimation", 4)
    L += ["", "## World", "",
          f"- Physics: {1 / sim_dt:.0f} Hz, policy {1 / (sim_dt * decimation):.0f} Hz, armature "
          f"{env_cfg.get('armature')}, torque limits {fmt_dict(env_cfg.get('torque_limits'))} Nm, latency "
          f"{dr.get('latency_range_s', env_cfg.get('latency_s'))} s",
          f"- Contact model [S2R]: joints {env_cfg.get('joint_friction', 0)} Nm / {env_cfg.get('joint_damping', 0)} "
          f"Nm s/rad, feet torsion {env_cfg.get('foot_torsional_friction')} rolling "
          f"{env_cfg.get('foot_rolling_friction')}, cone {env_cfg.get('friction_cone')} impratio "
          f"{env_cfg.get('impratio')}, MJCF lower legs {bool(env_cfg.get('lower_leg_inertial'))}",
          "- Randomization: " + (", ".join(f"{k.replace('_range', '')} {v}" for k, v in dr.items()) or "none")]
    t = curriculum.get("terrain")
    if env_cfg.get("terrain"):
        tc = env_cfg["terrain"]
        line = f"- Terrain: {tc}"
        if t:
            per = ", ".join(f"{name} {sum(i * c for i, c in enumerate(counts)) / max(1, sum(counts)):.1f}"
                            for name, counts in t["level_counts"].items())
            line = (f"- Terrain: {', '.join(t['types'])}, {t['num_levels']} levels up to {t['max_slope_deg']:g}°; "
                    f"mean level reached (0..{t['num_levels'] - 1}): {per}")
        L.append(line)
    else:
        L.append("- Terrain: flat plane")
    ranges = curriculum.get("command_ranges") or {"lin_vel_x": command_cfg["lin_vel_x_range"],
                                                  "lin_vel_y": command_cfg["lin_vel_y_range"],
                                                  "ang_vel": command_cfg["ang_vel_range"]}
    L.append(f"- Commands: vx {ranges['lin_vel_x']}, vy {ranges['lin_vel_y']}, wz {ranges['ang_vel']}"
             + (" (reached by the command curriculum; deployed)" if curriculum.get("command_curriculum") else "")
             + (f", body height {command_cfg['body_height_range']}" if command_cfg.get("num_commands") == 4 else ""))
    scales = reward_cfg["reward_scales"]
    L += ["", "## Policy interface and rewards", "",
          "- Observation: " + (", ".join(deploy["obs_layout"]) if deploy else "see go2_export.py") + "; noise "
          + (fmt_dict(obs_cfg.get("noise_scales")) if obs_cfg.get("add_noise") else "none"),
          f"- Action: 12 joint-position offsets × {env_cfg['action_scale']} around the default pose, PD (Kp "
          f"{env_cfg['kp']}, Kd {env_cfg['kd']}) on the motors",
          "- Rewards (weight per second): " + ", ".join(f"{k} {v:g}" for k, v in scales.items()),
          f"- Reward parameters: tracking σ {reward_cfg['tracking_sigma']}, base height "
          f"{reward_cfg['base_height_target']} m,"
          f" air time {reward_cfg.get('feet_air_time_target')} s, clearance {reward_cfg.get('feet_height_target')} m"
          + (f", foot slip mode {reward_cfg.get('foot_slip_mode', 'xy')}" if "foot_slip" in scales else "")
          + f", only positive {reward_cfg.get('only_positive_rewards')}, penalties × "
          f"{reward_cfg.get('penalty_scale_init', 1)} → 1 over "
          f"{reward_cfg.get('penalty_ramp_steps', 0) / train_cfg['num_steps_per_env']:g} iterations"]
    L += ["", "## Results (go2_report.py)", ""]
    if report:
        h, w = report["headline"], report["worlds"]
        for name in ("flat", "flat_dr"):
            items_h = {k[len(name) + 2:]: v for k, v in h.items() if k.startswith(name + ":")}
            if items_h:
                L.append(f"- {name}: " + "; ".join(f"{k} {fmt(v)}" for k, v in items_h.items()))
        grids = []
        for name, label in (("grid", "own tiles"), ("grid_unseen", "unseen tiles")):
            if name in w:
                fr = ", ".join(f"{t} {d['frontier'] if d['frontier'] is not None else 'none'}"
                               for t, d in w[name]["types"].items())
                grids.append(f"{label}: {fr} (mean through {fmt(w[name]['mean_through'])} %)")
        if grids:
            L.append(f"- Terrain frontier (last level with through ≥ {report.get('frontier_through', 80):g} %): "
                     + "; ".join(grids))
        slope_rows = [(k, w[k]) for k in ("slope+10", "slope-10", "slope+20", "slope-20", "slope+25", "slope-25")
                      if k in w]
        if slope_rows:
            L += ["", "| slope | vx_meas @ 0.5 m/s | through @ 0.5 m/s [%] | falls (all bins) | standing falls |",
                  "|---|---|---|---|---|"]
            L += [f"| {k[5:]}° | {fmt(x['walk_0.5']['vx_meas'])} | {fmt(x['walk_0.5']['through[%]'])} | "
                  f"{x['falls'][0]}/{x['falls'][1]} | {x['stand']['falls']} |" for k, x in slope_rows]
            L.append("")
        if "gate: runs as expected" in h:
            L.append(f"- unitree_mujoco gate suite: {h['gate: runs as expected']} runs as expected")
        if band:  # [P3v2] mean [min-max] over the training seeds
            def show(st):
                f = (lambda v: "n/a" if v is None else f"{100 * v:.0f} %") if st["share"] else fmt
                return f"{f(st['mean'])} [{f(st['min'])}–{f(st['max'])}]"
            L.append(f"- Seed band ({band['n']} runs: {', '.join(band['runs'])}), mean [min–max]: "
                     + "; ".join(f"{k} {show(band['stats'][k])}" for k in BAND_KEYS if k in band["stats"]))
        L.append(f"- Full report: `report/REPORT.md` ({report['created']}, seed {report['seed']}, "
                 f"{reps_text(report['reps'])} envs per bin)")
    else:
        L.append("- no report: `python go2_report.py -e EXP --ckpt N`")
    L += ["", "## Export", ""]
    if check:
        for name, errs in check["max_abs_error"].items():
            L.append(f"- `{name}` sha256 `{short(check['sha256'].get(name))}`: max |export − training policy| "
                     + ", ".join(f"{k} {v:.1e}" for k, v in errs.items()))
        L.append(f"- `deploy_cfg.json` sha256 `{short(check['sha256'].get('deploy_cfg.json'))}`"
                 + (f"; ONNX I/O {check['onnx_io']}" if check.get("onnx_io") else ""))
    else:
        L.append("- not exported: `python go2_export.py -e EXP --ckpt N --onnx`")
    L += ["", "## Decisions", ""]
    for rule, label in (("slip", "P3.2 foot-slip penalty"), ("critic", "P3.3 privileged critic")):
        d = decisions[rule]
        if d is None:
            L.append(f"- {label}: no record in `{args.decisions}/{rule}.md` (go2_decide.py {rule})")
            continue
        crit = ", ".join(f"{n} {'✓' if ok else '✗'}" for n, ok in d["criteria"])
        L.append(f"- {label}: **{d['decision']}** ({d['date']}, checkpoint {d['ckpt']}; off: {d['off']}; on: "
                 f"{d['on']}; flags of the on runs: `{d['flags']}`). Criteria: {crit}.")
    L.append(f"- reset with forward kinematics (`env.reset_forward`): {env_cfg.get('reset_forward')} "
             "(TEST_AND_TINKER_P3.md 4.1.8)")
    if manifest.get("argv") and manifest.get("git_hash"):
        L += ["", "## Reproduce", "", "```bash", f"git checkout {manifest['git_hash']}",
              " ".join(["python"] + manifest["argv"]), "```"]
    card = "\n".join(L) + "\n"
    card_path = os.path.join(log_dir, "CONFIG_CARD.md")
    with open(card_path, "w") as f:
        f.write(card)
    bad = [name for name, ok, _ in items if not ok]
    print(f"config card: {card_path}")
    for name, ok, why in items:
        print(f"  {'ok' if ok else 'NO':2s}  {name}: {why}")

    if args.freeze and bad and args.strict:  # a baseline is frozen clean, or not at all
        print(f"\nNOT a clean freeze: {', '.join(bad)} (see the checklist). --strict: nothing copied to {args.freeze}")
        raise SystemExit(1)
    if args.freeze:
        os.makedirs(args.freeze, exist_ok=True)
        copies = [card_path, os.path.join(log_dir, "exported", "deploy_cfg.json"),
                  os.path.join(log_dir, "exported", "export_check.json"), os.path.join(log_dir, "report", "REPORT.md"),
                  os.path.join(log_dir, "report", "summary.json"), os.path.join(log_dir, "manifest.json"),
                  os.path.join(log_dir, "cfgs.pkl"), os.path.join(log_dir, "git_diff.patch")]
        copies += [d["path"] for d in decisions.values() if d]
        for src in copies:
            if os.path.isfile(src):
                shutil.copy(src, os.path.join(args.freeze, os.path.basename(src)))
        with open(os.path.join(args.freeze, "cfgs.json"), "w") as f:
            json.dump(dict(zip(("env", "obs", "reward", "command", "train"),
                               (env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg))), f, indent=2, default=str)
        tag = args.tag or re.sub(r"[^A-Za-z0-9._-]+", "-", (args.name or args.exp_name).strip().lower())
        assets = [os.path.join(log_dir, f) for f in RELEASE_FILES if os.path.isfile(os.path.join(log_dir, f))]
        assets.append(ckpt_path)
        print(f"\nfrozen records in {args.freeze}/. Then (git tracks {args.freeze}/ and {args.decisions}/; logs/ and "
              f"*.pt are ignored, so the files go to the release):")
        print(f"  git add {args.freeze} {args.decisions}")
        print(f'  git commit -m "{title}: {args.exp_name} checkpoint {args.ckpt} (config card, report, export hashes)"')
        print(f'  git tag -a {tag} -m "{title}: {args.exp_name} checkpoint {args.ckpt}, trained from '
              f'{manifest.get("git_hash") or "unknown commit"}"')
        print(f"  git push origin HEAD {tag}")
        # --verify-tag: attach the release to the tag pushed above (without it, gh would create a missing tag from the
        # default branch's latest state)
        print(f'  gh release create {tag} {" ".join(assets)} --verify-tag --title "{title}" --notes-file '
              f"{os.path.join(args.freeze, 'CONFIG_CARD.md')}")
    if bad:
        print(f"\nNOT a clean freeze yet: {', '.join(bad)} (see the checklist)")
        if args.strict:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
