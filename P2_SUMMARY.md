# P2 summary: gait quality

What P2 changed, what was tested, what was kept, and what it means for P3. The code details are in
`TEST_AND_TINKER_P2.md` (section 2 is the change map); this file is the short version, with the results.

Tools: Genesis 1.4.x, rsl_rl 5.5.1 (PPO), Unitree Go2; P1 with the `[S2R]` and `[fix]` changes (joint losses,
MuJoCo-like contacts, tracking 1.5 / 0.75, penalty curriculum, 2000 iterations).

## Overview

| part | goal | what was implemented | result | kept |
|---|---|---|---|---|
| P2.0 | sim-to-sim baseline | joint friction and damping (moved into P1 `[S2R]`) | Genesis bench ≈ unitree_mujoco | in P1 |
| P2.1 | longer steps | air-time target `--air-time`, cap `--air-time-max` | the term is ~0.5 % of the reward; the cap never binds | no change |
| P2.2 | foot clearance | Spot foot-clearance reward (moved into P1 `[S2R]`) | swing height 6–9 cm at walking speeds | in P1 (ablation not run) |
| P2.3 | symmetry | rsl_rl symmetry extension, `go2_symmetry.py` mirror | `sym_err` ÷10, left and right turns equal | **`--symmetry`** (augmentation) |
| P2.4 | gait pattern | Spot GaitReward, walk-these-ways clock | not needed: regular, mostly diagonal gait | skipped |
| P2.5 | energy | Σ\|τ·q̇\| penalty, CoT at 200 Hz | CoT −23…−31 %, also in unitree_mujoco | **`--energy 1.5e-3`** |

**Final configuration:** `--symmetry --energy 1.5e-3` on top of P1.
**Final policy:** `go2-p25-w1.5e-3` (seed 1, the one exported and tested in unitree_mujoco); second seed `go2-p25-s2`.
There is no separate `go2-p2` run: it would be the same configuration.

## Versions of the P2 code

| version | date | base | notes |
|---|---|---|---|
| P2 v1 | 2026-10-08 | old P1 (plain Genesis) | first implementation, guide v1 |
| **P2 v2 ("P2 vanilla")** | 2026-11-03 | P1 with `[S2R]` / `[fix]` | ported onto the new P1; P2.0 and P2.2 already in P1; weights ×1.5 for the new tracking; `go2_analyze.py` and `go2_sim_suite.py` gait columns. **P3 was built on this version.** |
| test & tinker | 2026-11-05 / 06 | P2 v2 | **no code change** to the `go2_*.py` files. Changed: the guide (section 2, 3.2, 5, `$KEEP` instead of placeholders) and `p2_overnight.sh` (new) |

## Method lessons (they apply to P3 too)

1. **Training noise is much larger than evaluation noise.** The three `--air-time-max` runs trained the same effective
   reward (the cap never bound), so their spread is run-to-run noise. Between evaluation envs the ± is about 0.001;
   between trainings:

   | at | spread over 3 trainings |
   |---|---|
   | `vx_meas`, cmd 0.25 | 0.140–0.196 m/s |
   | `lin_err`, cmd 0.5 | 0.060–0.085 m/s |
   | `lin_err`, cmd −0.5 | 0.043–0.082 m/s |
   | `lin_err`, cmd ±0.3 lateral | 0.075–0.115 m/s |
   | `air_t`, cmd 0.5 | 0.196–0.252 s |
   | `swing_h`, cmd 0.5 | 6.9–8.7 cm |
   | `CoT`, cmd 1.0 | 0.595–0.706 |
   | `air_asym`, cmd 0.5 | 2.4–11.0 % |

   `go2_compare.py`'s `~` only knows the evaluation spread, so it calls nearly every training difference significant.
   Judge changes against a band of several seeds, never against one reference run: `go2-p21-c0.45` happened to be the
   best of the three baseline runs, and compared alone it made the P2 runs look worse than they are.
2. **Check the size of a reward term, not only its shape.** The air-time term had the right shape (5.1 of the guide)
   but at weight 1 it is about 0.5 % of the reward, so its target and cap changed nothing.
3. **`air_asym` in lateral and turning bins is a gait choice**, not a symmetry defect: a sideways or turning step is
   naturally different for the left and right legs. Symmetry is read from `sym_err`, the mirror-pair check, and
   `air_asym` in the straight bins.
4. **Two conda environments.** Genesis (`sandbed`) and unitree_mujoco + `unitree_sdk2py` (`go2deploy`) live in
   separate envs; anything that starts unitree_mujoco must run with the `go2deploy` python
   (`p2_overnight.sh`: `DEPLOY_PY`).
5. **Run one MuJoCo batch at a time.** Two batches in parallel saturate the CPU, the simulator and the 2 ms loop fall
   behind real time, and the results are invalid (`flock -n` against double starts).

## P2.0 Sim-to-sim baseline

- Joint friction 0.2 Nm and damping 0.1 Nm·s/rad (unitree_mujoco `go2.xml`) are in P1 `[S2R]`, with the MuJoCo-like
  foot contacts. P2 added no code here.
- Baseline used for comparisons: the three `go2-p21-c*` runs (the P1 configuration; the cap never binds), see the noise
  band above. `go2-p20` was not used.

## P2.1 Longer steps

- **Sweep:** `--air-time-max` 0.35 / 0.45 / 0.6.
  - `air_t` stayed at 0.16–0.25 s in every bin, below the smallest cap, with `air_cv` 1–8 %: the cap never clips a
    swing.
  - At 0.5 m/s the term pays 0.08·(0.34 − 0.49) = −0.012 per second, about 0.5 % of the 2.25 per second of tracking.
- **Walking speeds were already fine:** 1.5–2 Hz, 0.20–0.25 s swings, 7–10 cm clearance, close to Unitree's own trot
  (0.225 s swings).
- **Side experiments:**
  - `tracking_sigma` 0.25 → 0.1 (slow commands undershoot): seed 1 moved the slow, lateral and yaw bins just past the
    band; **tested, not kept**.
  - Evaluation latency 20 → 10 ms: removes about a fifth of the linear shortfall (0.005–0.015 of 0.05–0.07 m/s), none
    of the yaw shortfall. The standard evaluation stays at 20 ms.
- **Decision:** no change. The air-time term stays at weight 1, target 0.3 s, no cap.

## P2.2 Foot clearance

- In P1 `[S2R]`: weight 1.0, std 0.02, target 7.5 cm (Spot `foot_clearance_reward`).
- Measured on the final policy: `swing_h` 8.2 cm at 0.5 m/s (seed 1), 6.2 cm (seed 2); 8.6 cm in unitree_mujoco.
- **Open:** the ablation `go2-p22-off` (guide 6.2) was not run; it would say what the term contributes.

## P2.3 Symmetry

- `go2_symmetry.py`: left-right mirror (FL ↔ FR, RL ↔ RR, hip sign, pseudovector angular velocity, commands, clock).
  Self-check on the `[S2R]` physics: largest block error 4.6e-4; a wrong sign gives about 0.8.
- `--symmetry` (data augmentation), two seeds (`go2-p23`, `go2-p23-s2`):
  - `sym_err` 0.016–0.066 rad → 0.002–0.005 rad in every bin;
  - turning in place became equal left and right (seed 1: swing height 6.9 / 6.8 cm, `air_cv` 2.8 % both ways); the
    right-turn shuffle of the baseline runs is gone;
  - tracking: forward, backward and lateral inside the baseline band; seed 1 ran about 0.02 m/s slow forward, seed 2
    did not;
  - CoT: no gain (seed 1 −12 %, seed 2 +4 % at 0.5–1 m/s). The P2.3 CoT gain seen on one seed was noise.
- **Decision:** keep `--symmetry`.

## P2.4 Gait pattern

- Implemented (Spot `gait` + `air_time_variance`, or the walk-these-ways clock with deploy-side clock and bench check),
  never trained.
- Not needed: `diag_sync` 80–95 %, `air_cv` 0.2–9 % on all P2 runs.

## P2.5 Energy

- `energy = −W · Σ|τ·q̇|` at 200 Hz (the same quantity as `power` and `CoT` in `go2_metrics.py`).
- **Weight sweep** on top of `--symmetry` (p23 uses ~30 W at 0.5 m/s, so the old calibration was too weak):

  | W | CoT 0.5 / 1.0 | what else |
  |---|---|---|
  | 0 (p23) | 0.498 / 0.584 | |
  | **1.5e-3** | 0.392 / 0.450 | tracking ±0.01, swing height unchanged at walking speeds |
  | 4e-3 | 0.349 / 0.390 | backward, lateral, 1 m/s tracking worse; swing height −20…−49 % |
  | 8e-3 | 0.296 / 0.365 | stands still at 0.25 m/s and ±0.3 lateral (energy beats tracking) |

- **Two seeds of W = 1.5e-3** (`go2-p25-w1.5e-3`, `go2-p25-s2`):
  - power −21…−32 % at 0.5 m/s, −30…−31 % at 1 m/s; CoT −23…−31 % at all forward speeds;
  - with the training randomization: power −20…−29 %, CoT −18…−29 %;
  - step frequency lower, swings longer (seed 1: 1.25 Hz and 0.32 s at 0.5 m/s): what P2.1 aimed at, obtained through
    energy (Fu et al. 2021);
  - cost: backward about 0.01–0.02 m/s slower (`lin_err` at −0.5: 0.083 / 0.098, band 0.043–0.082); lateral inside the
    band;
  - seed 2 is weaker overall: turns (`yaw_err` 0.19 against 0.13–0.16), lower swings (6.2 cm at 0.5 m/s).
- **Decision:** keep `--energy 1.5e-3`; deploy seed 1.

## Final policy against the baseline (Genesis, nominal world)

Baseline: `go2-p21-c0.45` (one of the three baseline runs; band in the method section). Final: seed 1 / seed 2.

| at | baseline | final (seed 1 / seed 2) |
|---|---|---|
| falls, all bins (nominal / randomized) | 0 / 1 of 20 at lateral | 0 / 0 |
| `lin_err`, cmd 0.5 / 1.0 | 0.085 / 0.080 | 0.094 / 0.101 · 0.079 / 0.089 |
| `lin_err`, cmd −0.5 | 0.043 (band to 0.082) | 0.083 / 0.098 |
| `sym_err`, cmd 0.5 | 0.041 rad | 0.005 / 0.003 rad |
| `power`, cmd 0.5 / 1.0 | 35.2 / 87.8 W | 24.0 / 61.1 · 27.7 / 60.7 W |
| `CoT`, cmd 0.5 / 1.0 | 0.566 / 0.632 | 0.392 / 0.450 · 0.437 / 0.439 |
| `step_freq`, cmd 0.5 | 1.65 Hz | 1.25 / 1.53 Hz |
| `swing_h`, cmd 0.5 | 6.9 cm | 8.2 / 6.2 cm |
| `diag_sync`, cmd 0.5 | 83 % | 89 / 82 % |

## Sim-to-sim (unitree_mujoco, through the deploy path)

- **Gate suite:** 20 / 20 (all profiles, foot friction 1.0 and 0.4); tilt ≤ 3°, torque peak ≤ 46 %.
- **Stress suite:** 26 / 26 (2× noise, 10–40 ms command delay, 10 ms state delay, packet drops, 2 and 5 kg payload,
  3 cm CoM shift, 90 % and 75 % motor strength, 0.5 m/s pushes, PD per policy step, 0.03 rad encoder offsets). The
  worst slowdown: 0.38 m/s for 0.5 at 40 ms command delay. The trot holds everywhere (`diag_sync` 85–96 % under stress, 77–92 % in the gate).
- **Energy transfers** (fwd_back profile, `go2-p23` → final, one run each):

  | | unitree_mujoco | Genesis |
  |---|---|---|
  | power, cmd +0.5 / −0.5 | −14 % / −25 % | −20 % / −28 % |
  | CoT, cmd +0.5 / −0.5 | −18 % / −23 % | −21 % / −27 % |

  The backward cost shows in MuJoCo too (`lin_err` at −0.5: 0.048 → 0.064).
- **Gait transfers:** `diag_sync` 87–93 %, `air_cv` about 2 %, left/right asymmetry about 1 %.
- **Contact estimate:** with a real policy, the torque-based estimate is close to the true contacts (swing 0.296 s
  estimated, 0.276 s true; duty 0.574 / 0.603). The large bias measured on the open-loop clock trot (0.13 / 0.21 s)
  does not appear. Good news for the estimated gait columns on the robot.
- **Open:** `loop p99` 4–6 ms against the 2 ms period (1 % of the cycles take 2–3 periods); the gate does not check it.
  Compare with an earlier suite run (p17h) or with "loop steps > 2x period" in `go2_analyze.py`'s timing table.

## Still weak after P2 (for later parts)

- Slow and lateral commands undershoot: 0.16 for 0.25 m/s, 0.22 for 0.3 m/s lateral (unitree_mujoco gate); the same in
  Genesis.
- Yaw: 0.54 for 0.6 rad/s (gate); in Genesis 0.34–0.37 for 0.5 rad/s turning in place.
- A constant shortfall of about −0.05 m/s at every forward speed, mostly in the policy (latency explains a fifth).
  Probably the price of a policy that does not observe its velocity; a privileged critic (P3.3) is one candidate.

## What this means for P3 (built on P2 vanilla)

The P2 code did not change after P2 vanilla (2026-11-03), so P3 already contains every P2 feature. What changes is the
configuration, the run names and some decision rules.

1. **Add the P2 decisions to P3's base options.** P3's commands use `$BASE` = "your P2 options". It should be
   ```bash
   BASE="--seed 1 --symmetry --energy 1.5e-3"     # + any P1 option you keep (e.g. --height-cmd)
   ```
   This applies to P3.1 to P3.3 runs and to the v1.0 command line (`go2_train.py -e go2-v1.0 $BASE $V1FLAGS`).
   No `--air-time-max`, no `--gait`, no `tracking_sigma` change.
2. **The P2 reference run is `go2-p25-w1.5e-3`, not `go2-p2`.** P3's guide uses `logs/go2-p2/...` (e.g.
   `--init-from logs/go2-p2/model_1999.pt`, `go2_compare.py logs/go2-p2/metrics_..._grid.csv ...`). Either replace the
   name, or make an alias once:
   ```bash
   ln -s go2-p25-w1.5e-3 logs/go2-p2
   ```
   `go2-p2` is then a name for the final P2 policy; do not train a run with that name afterwards (training deletes and
   recreates its folder).
3. **The decision rules of `go2_decide.py` are tighter than the measured training noise.** P3.2 / P3.3 keep a feature
   only if the costs stay small (`lin_err` ≤ +5 %, CoT ≤ +5 %, through within 5 points). P2 measured `lin_err` at
   0.5 m/s varying 0.060–0.085 (±17 %) and CoT at 1 m/s 0.595–0.706 (±9 %) between identical trainings. With 2 seeds,
   a +5 % rule will often DROP a feature for noise. Options: compare seed means against the spread of the A seeds, use
   3 seeds, or widen the cost thresholds to the measured band. Check the thresholds in `go2_decide.py`'s header before
   running the decisions.
4. **Watch energy against the terrain curriculum.** Climbing a 25° face costs much more power than flat walking, and
   the energy term grows with power (it is ramped by the penalty curriculum like every penalty). I expect it may slow
   the terrain curriculum on hill and pit tiles; not tested. If `Curriculum/terrain_level` stalls on those types, train
   one terrain run without `--energy` to compare, or lower W on terrain.
5. **Energy and foot slip interact.** P2.5 showed energy pulling swing height down at W ≥ 4e-3; P3.2's foot-slip
   penalty also prices foot motion. Read `swing_h` and `stride` in the P3.2 A/B runs with energy on in both.
6. **`go2_report.py --gate` and the two environments.** The report runs the gate suite. If it starts
   `go2_sim_suite.py` with the `sandbed` python, it will fail with `No module named 'unitree_sdk2py'`, as the first
   P2 batch did. Run the gate with the `go2deploy` python and pass its summary with `--gate-summary`, or give the
   report a deploy-python option (the `DEPLOY_PY` of `p2_overnight.sh`).
7. **Seed statistics.** P3.4 lists "seed statistics in the report" as not done. P2's noise band makes it worth doing
   before v1.0: one candidate against one baseline run is not a result.
8. **Minor:** the help text of `--air-time-max` in `go2_train.py` says the uncapped term "always prefers fewer, longer
   steps". True of its shape; at weight 1 it has no practical effect. No behaviour change.

## Key files and runs

| what | where |
|---|---|
| final policy (seed 1), export | `logs/go2-p25-w1.5e-3/`, `logs/go2-p25-w1.5e-3/exported/` |
| second seeds | `logs/go2-p25-s2/`, `logs/go2-p23-s2/` |
| symmetry only | `logs/go2-p23/`, `logs/go2-p23-s2/` |
| baseline band | `logs/go2-p21-c0.35/`, `logs/go2-p21-c0.45/`, `logs/go2-p21-c0.6/` |
| energy sweep | `logs/go2-p25-w4e-3/`, `logs/go2-p25-w8e-3/` |
| comparisons, sim-to-sim, suites | `out/p2_final/` (`compare_nominal.txt`, `compare_dr.txt`, `energy_in_*`, `suite_gate/`, `suite_stress/`) |
| batch script | `p2_overnight.sh` (`STAGES`, `DEPLOY_PY`) |
| guide | `TEST_AND_TINKER_P2.md` |

## Sources

- Mittal et al., "Symmetry Considerations for Learning Task Symmetric Robot Policies", ICRA 2024 (arXiv:2403.04359)
- Fu, Kumar, Malik, Pathak, "Minimizing Energy Consumption Leads to the Emergence of Gaits in Legged Robots", CoRL 2021
  (arXiv:2111.01674)
- Isaac Lab Spot `rewards.py` (foot clearance, gait, air-time variance); legged_gym `_reward_feet_air_time`;
  walk-these-ways (Margolis & Agrawal 2022); unitree_rl_lab Go2 velocity config (energy, air-time variance);
  unitree_mujoco `go2.xml`; unitree_guide (0.45 s trot period)
