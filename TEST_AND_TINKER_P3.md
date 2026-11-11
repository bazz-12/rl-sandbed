# P3 · Terrain, curricula and the v1.0 baseline: test & tinker guide

Companion to the P3 code, in the format of `TEST_AND_TINKER_P2.md`. **P3.1 (curriculum training), P3.2 (foot slip and
the foot-slip penalty decision), P3.3 (privileged observations) and P3.4 (the Baseline v1.0 freeze: provenance, checked
ONNX export, harness report, config card, git tag) are implemented**: they were checked on CPU here and are described
in full (sections 4 to 7).

**P3 v2 (2026-11-06): the P2 results applied** (`P2_SUMMARY.md`). `$BASE` holds the P2 choices (`--symmetry --energy
1.5e-3`) and `$P2` names the final P2 policy (`go2-p25-w1.5e-3`, second seed `go2-p25-s2`). The A/B decisions judge
costs against the training noise P2 measured, with 3 seeds per arm. The report builds seed bands, the gate suite runs
in the unitree_mujoco environment, and energy is watched on terrain. Markers `[P3v2]`; the list is in section 2.

Each part section covers:
- what the code does and why;
- the commands;
- what to read in `go2_metrics.py` and TensorBoard, and how to interpret it;
- the knobs;
- symptoms and fixes;
- a "done when" check.

Reference tables are at the end.

Contents
1. Files and install
2. Change map: what each part modifies, file by file (P3.1 to P3.4)
3. Workflow and conventions (base command, run names, knobs, evaluation worlds, comparing, TensorBoard, visual checks)
4. P3.1 Curriculum training: terrain tiles, terrain curriculum, command curriculum, slopes up to 25°
5. P3.2 Foot slip from ground truth, and the foot-slip penalty decision
6. P3.3 Privileged observations for the critic and a teacher
7. P3.4 Baseline v1.0: final run, config card, checked export, harness report, git tag
8. Reference: configuration keys and constants
9. Reference: metrics glossary (P3)
10. Troubleshooting
11. Sources

---

## 1. Files and install

| File | Status vs P2 | Part |
|---|---|---|
| `go2_terrain.py` | **new** | P3.1: tile grid, tilted plane, ground-height lookup, terrain cache, preview, self-checks (numpy + torch, no Genesis). P3.2: ground normals, `contact_slip_speed()`, the contact-normal check |
| `go2_decide.py` | **new** | P3.2 / P3.3: A/B decisions from the harness by fixed rules; writes `decisions/<rule>.md` (no Genesis). P3.4: `flags` prints the flags of the KEEP records. P3 v2: a cost must be beyond its tolerance and outside the seed range; P2's weak commands shown |
| `go2_provenance.py` | **new** | P3.4: git state, code fingerprint, file hashes, versions (standard library only) |
| `go2_report.py` | **new** | P3.4: the harness report: the standard evaluation set in one command, `report/REPORT.md` + `summary.json`, `--ref`. P3 v2: `--seeds` (seed band, verdicts), `--deploy-python` |
| `go2_card.py` | **new** | P3.4: the one-page config card with its freeze checklist; `--freeze` into `baselines/`, the git tag and release commands. P3 v2: `seed band` line |
| `go2_env.py` | modified | P3.1: ground entity, spawning, terrain and command curricula, ground-relative rewards, tilt limits per env, terrain build fix, resets with forward kinematics on terrain. P3.2: per-foot slip, `foot_slip` reward. P3.3: `privileged` observation group |
| `go2_train.py` | modified | P3.1: `--terrain`, `--cmd-curriculum`, `--init-from`, `CMD_CURRICULUM`, `TERRAIN_TILT_MARGIN_DEG`, P3 summary block, `terrain.npz`, curriculum state in checkpoints. P3.2: `--foot-slip`. P3.3: `--privileged`, `--privileged-entries`. P3.4: full `manifest.json` (code fingerprint, argv, untracked files), `git_diff.patch`, repeatable `--set`. P3 v2: `--energy 0` = off |
| `go2_metrics.py` | modified | P3.1: `--terrain grid`, `--slope`, Progress table, success heat map, ground-relative heights. P3.2: slip along the ground, `slip_share`, `pen_xy` / `pen_t`, "Foot slip per foot" table. P3.3: `sym_err` over every group |
| `go2_compare.py` | modified | P3.1: tile rows (`--rows "hill 9"`), `--preset terrain`. P3.2: `--preset slip` |
| `go2_feet_plot.py` | modified | P3.1: clearance above the ground, `--tile TYPE LEVEL`. P3.2: speed along the ground |
| `go2_export.py` | modified | P3.1: deploys the command ranges the curriculum reached; `training_terrain` record. P3.3: refuses teacher runs; records the critic's entries. P3.4: `--onnx` (one self-contained file), both exports checked on random and rollout observations, `export_check.json`, hashes |
| `go2_symmetry.py` | modified | P3.3: mirror of every privileged entry; `--privileged`, `--slope` checks |
| `go2_teleop.py` | modified | P3.1: `--terrain`, `--tile`, `--slope`, J / K / U keys, reached command ranges |
| `go2_fsm_sim.py` | modified | P3.1: `--slope`; flat ground otherwise (also with `--dr`) |
| `go2_sim_suite.py` | modified | P3 v2: one battery at a time (a lock in the temp folder) |
| `go2_fsm.py`, `go2_deploy_sdk2.py`, `go2_inputs.py`, `go2_commands.py`, `go2_analyze.py`, `go2_eval.py`, `go2_smooth_ppo.py`, `go2_kinematics.py`, `go2_log.py`, `go2_mujoco_headless.py`, `go2_probe.py`, `test_go2_deploy.py` | unchanged | |

- Put `go2_terrain.py` next to the other scripts. `go2_env.py` imports it, so every script that builds an env needs
  it, even on flat ground.
- No new package for training. `go2_terrain.py` uses numpy and torch; `matplotlib` only for the preview. [P3.4]
  `go2_export.py --onnx` needs `pip install onnx onnxscript onnxruntime`.
- **Terrain cache**: generated grids are stored in `~/.cache/go2_terrain/` (set `GO2_TERRAIN_CACHE` to move it), one
  `.npz` per configuration. Generating the default grid takes well under a second, so the cache only saves time. Delete the
  folder whenever you like.
- **Every run on terrain saves the exact grid it trained on** in `logs/<exp>/terrain.npz`. `go2_metrics.py`,
  `go2_teleop.py` and `python go2_terrain.py --run logs/<exp>` read that file, so a run is always evaluated on its own
  tiles, even if you change a generator later.
- Robot computer: nothing new. Re-export P3.1 runs with `go2_export.py` (the deployed command ranges changed, see
  4.4.6).
- Old runs (P1, P2) still load and evaluate. Their checkpoints have no curriculum state; the scripts say so and
  carry on.
- Without the P3 flags, a run is exactly your P2 configuration: same weights bit for bit (checked, see `CHANGES.md`).
- Markers are `[P3.1]` to `[P3.4]`, and `[P3v2]` for the P2 results applied: `grep -n "\[P3" *.py` lists every
  change.
- `go2_decide.py` writes its decision records to `decisions/` at the top of the repository. Commit them: `logs/` is
  git-ignored, `decisions/` is not. Likewise `go2_card.py --freeze` writes `baselines/<version>/` (P3.4).

Self-checks you can run any time (CPU):

```bash
python go2_terrain.py --check             # numpy: lookup, gradients, slopes, tile edges, cache (seconds)
python go2_terrain.py --check --genesis   # + the lookup against Genesis' own terrain, and resting spheres (1-2 min, ~3 GB)
python go2_symmetry.py                    # the policy mirror (P2.3), worth running after any edit
python go2_symmetry.py --privileged --slope 10   # [P3.3] every privileged entry, on a slope
python test_go2_deploy.py                 # deploy side; add --policy logs/<exp>/exported for an export
python go2_provenance.py                  # [P3.4] git state and package versions; + logs/<exp>: code changed since?
```

---

## 2. Change map: what each part modifies, file by file

Each part builds on the previous one. "Why" is the reason the file has to change; the details are in that part's
section.

### P3.1 Curriculum training (implemented; section 4)

| File | What changes | Why |
|---|---|---|
| `go2_terrain.py` (new) | `TERRAIN_DEFAULTS`; tile generators (flat, rough, hill, pit, wave, obstacles); `TerrainGrid` / `PlaneGround` / `FlatGround`; heights and gradients on Genesis' triangles; cache; preview PNG; self-checks | Genesis 1.4.2's subterrains take one parameter set per type, so they cannot grade difficulty per row. A custom heightfield can, and the same lookup serves the env (heights under the feet) and the offline tools. |
| `go2_env.py` | Ground entity (plane / tilted plane / heightfield); spawning on tiles; `_update_terrain_curriculum()`; command curriculum; `Curriculum/*` logs; heights from the ground in `base_height`, `foot_clearance` and the recovery terms; per-env tilt limits; off-grid termination; `curriculum_state()`; terrain build fix; `reset_forward` | The curricula live where episodes end (`_reset_idx`). Every reward that assumed z = 0 had to measure from the ground instead. |
| `go2_train.py` | `--terrain`, `--cmd-curriculum`, `--init-from`; tilt margin; summary block; `terrain.npz`; curriculum state saved in every checkpoint | Switches, and reproducibility: the exact grid and the curricula's state travel with the run. |
| `go2_metrics.py` | `--terrain grid` (one bin per tile), `--slope`; Progress table (`dist`, `through`); heat map; heights from the ground; reset fix on terrain | The question changes from "how well does it track on flat ground" to "how far does it get, on what". |
| `go2_compare.py` | Tile rows, `--preset terrain` | Compare runs tile by tile (P2 vs P3.1 on the same tiles). |
| `go2_feet_plot.py` | Clearance above the ground, `--tile` | Swing height is only meaningful above the ground under the foot. |
| `go2_export.py` | Reached command ranges into `deploy_cfg.json`; terrain record | The policy never trained beyond the ranges its curriculum reached; the joystick must not ask for more. |
| `go2_teleop.py` | `--terrain`, `--tile`, `--slope`, J / K / U | See it walk on the tiles; drive P2 and P3 policies over the same ground. |
| `go2_fsm_sim.py` | `--slope` | Stand-up, hand-over and sit-down on a slope, through the deploy code. |

### P3.2 Foot slip from ground truth; foot-slip penalty decision (implemented; section 5)

| File | What changes | Why |
|---|---|---|
| `go2_terrain.py` | `normal_at()` / `normal_np()` on every ground; `contact_slip_speed()`; checks (rolling / sliding / sinking on a 20° face; the solver's contact normal vs the lookup's) | One definition of slip, along the face the foot stands on, shared by the env, the harness and the plot. |
| `go2_metrics.py` | Slip along the ground (P1 formula kept for flat recordings); `slip_share`, `pen_xy`, `pen_t`; "Foot slip per foot" table; `foot_normal` in terrain `.npz` | The horizontal P1 slip misses cos θ on slopes and counts sinking; the mean over feet hides which leg slips. Flat values unchanged. |
| `go2_env.py` | `feet_ang`, `feet_normal`, `feet_force`, `foot_slip` [N, 4]; `_reward_foot_slip()` (modes `xy` = Isaac Lab Spot, `tangent`) | The penalty needs the signal; P3.3's privileged entries reuse it. |
| `go2_train.py` | `--foot-slip [W]`, `FOOT_SLIP_WEIGHT` (0.1), a P3.2 line in the start-up block | Opt-in: the A/B differs by one flag. |
| `go2_compare.py` | `--preset slip` | |
| `go2_feet_plot.py` | Speed along the ground on terrain recordings; mean stance slip per foot in the title | Same quantity as the harness. |
| `go2_decide.py` (new) | Rule `slip`; `decisions/slip.md` | The decision is taken by a rule fixed before the numbers, and recorded for the config card (P3.4). |

### P3.3 Privileged observations (implemented; section 6)

| File | What changes | Why |
|---|---|---|
| `go2_env.py` | `PRIVILEGED_SIZES` (11 entries), `privileged_layout()`, a `privileged` group next to `policy` (none by default); `_privileged_obs()`, `_height_scan()` | The critic (and a teacher) may see what the robot cannot measure. The env already draws and stores the randomized values (P1.2). |
| `go2_train.py` | `--privileged critic` / `teacher`, `--privileged-entries`, `PRIVILEGED_DEFAULT` (36 values); checks; CAPS noise 0 on the privileged part | rsl_rl 5.5.1 selects observation groups per network with `obs_groups`. |
| `go2_symmetry.py` | A mirror rule per entry; every group mirrored; `--privileged` and `--slope` checks | `--symmetry` augments the critic's inputs too; a missing rule is an error, not a silent bug. |
| `go2_export.py` | Refuses a teacher (its actor needs privileged inputs); records the critic's entries | Deployment stays blind. |
| `go2_metrics.py` | `sym_err` mirrors every group the actor reads | Teachers in the harness. |
| `go2_decide.py` | Rule `critic` (TensorBoard curves + harness); `decisions/critic.md` | Same traceability as P3.2. |

### P3.4 Baseline v1.0 (implemented; section 7)

| File | What changes | Why |
|---|---|---|
| `go2_provenance.py` (new) | Git commit, dirty flag, untracked `.py` files; sha256 of every `.py` file (code fingerprint); package versions | Git says which commit a run started from, not whether a file changed afterwards; the fingerprint does. |
| `go2_train.py` | Complete `manifest.json` (argv, arguments, host, GPU, fingerprint), also without git; `git_diff.patch` and warnings; `--set` repeatable | The card must say exactly what trained the policy; the v1.0 command line combines several `--set`. |
| `go2_export.py` | `--onnx` (rsl_rl `export_policy_to_onnx`, weights folded into one file); both exports checked against the training policy on random and rollout observations; `export_check.json`; hashes in `deploy_cfg["source"]` | "Checked against the training model", for TorchScript and ONNX, on the states the policy meets. |
| `go2_report.py` (new) | The standard set (flat, flat DR, own tiles, unseen tiles, slopes ±10/20/25, optional gate) in one command; `REPORT.md`, `summary.json`, plots; `--ref` | The harness report: the same numbers for every later comparison. |
| `go2_card.py` (new) | `CONFIG_CARD.md` with a freeze checklist; `--freeze` into `baselines/`; prints the git tag and `gh release` commands | One page that says exactly what v1.0 is, and refuses a freeze that is not clean. |
| `go2_decide.py` | `flags`: the flags of the KEEP records | The v1.0 command line is built from the decisions, and the card checks it. |
| `baselines/v1.0/`, git, `runs.md` | Frozen records (tracked), annotated tag `baseline-v1.0`, a GitHub release with the binaries, the v1.0 row | `logs/` and `*.pt` are git-ignored, so the tag alone does not hold the policy. |

### P3 v2: the P2 results applied (implemented; `P2_SUMMARY.md`)

| What P2 found | What changes | Where |
|---|---|---|
| Final P2 configuration: `--symmetry --energy 1.5e-3` | `$BASE` holds it, for every P3 run and the v1.0 command line | 3.1 |
| The final P2 policy is `go2-p25-w1.5e-3` (second seed `go2-p25-s2`); there is no `go2-p2` run | `$P2`, `$P2S2`; P3.1 is compared with both P2 seeds | 3.1, 3.4, 4.4 |
| Training noise ≫ evaluation noise (lin_err at 0.5 m/s 0.060–0.085 over 3 identical trainings; CoT at 1 m/s 0.595–0.706) | `go2_decide.py`: a cost counts only if it is beyond its tolerance **and** every --on seed is worse than every --off seed; 3 seeds per arm | 5.4, 6.4 |
| One candidate against one baseline run is not a result | `go2_report.py --seeds`: a seed band, and verdicts that need non-overlapping seed ranges; card line `seed band` (3 seeds) | 7 |
| Two conda environments (Genesis; unitree_mujoco + unitree_sdk2py) | `go2_report.py --deploy-python` / `$DEPLOY_PY` for the gate, checked before the evaluation | 3.1, 7 |
| One MuJoCo batch at a time | `go2_sim_suite.py` refuses a second battery (lock) | 7 |
| Energy may slow the terrain curriculum (expected, not tested) | what to watch; `--energy 0` removes the term from `$BASE` for a comparison run | 4.3, 4.5, 4.9 |
| Energy and the foot-slip penalty both price foot motion | both slip arms with `$BASE`; `stride` and `swing_h` on the terrain worlds of the record | 5.4 |
| Slow, lateral and turning commands still undershoot | the critic record shows them (not judged): a privileged critic is one candidate | 6.4 |
| `--air-time-max` help overstated the uncapped term | help text corrected (no behaviour change) | `go2_train.py` |

---

## 3. Workflow and conventions

### 3.1 Base command and run names
Keep the P2 options you settled on in one variable, so every P3 run differs only by its P3 flags:

```bash
BASE="--seed 1 --symmetry --energy 1.5e-3"   # [P3v2] the P2 choices; + any P1 option you keep (e.g. --height-cmd)
CKPT=1999                  # 2000 iterations: the last checkpoint is model_1999.pt
P2=go2-p25-w1.5e-3         # [P3v2] the final P2 policy (seed 1, exported, tested in unitree_mujoco): P3.1's reference
P2S2=go2-p25-s2            # [P3v2] its second seed: the P2 band
DEPLOY_PY=$(conda run -n go2deploy which python)   # [P3v2] the unitree_mujoco + unitree_sdk2py python (gate suite)
```
- [P3v2] There is no `go2-p2` run: it would be the same configuration as `go2-p25-w1.5e-3`. Use the real name rather
  than a `logs/go2-p2` symlink, which would get in the way of any later run of that name (training deletes and
  recreates its folder).
- [P3v2] Two conda environments, as in P2: everything here runs in the Genesis one (`sandbed`), except what starts
  unitree_mujoco. `go2_report.py --gate` starts the gate suite with `$DEPLOY_PY` (7.1.3); run `go2_sim_suite.py` by
  hand with that python too, one battery at a time.

Run names below: `go2-p31c` (command curriculum alone, flat), `go2-p31t` (terrain alone), `go2-p31` (both),
`go2-p31i` (both, initialized from P2). P3.2: `go2-p32` (the P3.1 flags: the "without" arm), `go2-p32-slip` (+ the
foot-slip penalty). P3.3: `go2-p33` (privileged critic), `go2-p33t` (teacher). Further seeds get `-s2`, `-s3` and
`--seed 2`, `--seed 3`.

### 3.2 Where the knobs are
1. **Flags** of `go2_train.py`:
   - `--terrain`: the tile grid with the terrain curriculum;
   - `--cmd-curriculum`: growing command ranges;
   - `--init-from logs/<exp>/model_N.pt`: start from those weights.
2. **`--set env.terrain.KEY=VALUE`** changes any key of `go2_terrain.TERRAIN_DEFAULTS` (table in 4.7). Notes:
   - It works with or without `--terrain`: a non-empty `env.terrain` switches the grid on.
   - The config is checked before Genesis starts, so a typo fails at once (`unknown terrain keys`).
   - Values are Python literals:
     ```bash
     --set env.terrain.max_slope_deg=20 env.terrain.num_levels=6
     --set "env.terrain.types={'flat':0.1,'hill':0.45,'pit':0.45}"     # slopes only
     --set env.terrain.up_distance=3.0 env.terrain.max_init_level=0
     ```
3. **`--set command.curriculum.KEY=VALUE`**, only together with `--cmd-curriculum`. Without the flag, `--set` creates a
   curriculum without start ranges, i.e. one that never narrows anything.
   ```bash
   --cmd-curriculum --set command.curriculum.threshold=0.7 command.curriculum.interval_s=10.0
   --cmd-curriculum --set "command.curriculum.start={'lin_vel_x':[-0.7,0.7],'lin_vel_y':[-0.3,0.3],'ang_vel':[-0.7,0.7]}"
   ```
4. **Terminations**:
   - With `--terrain`, the tilt limit becomes `max_slope_deg` + `TERRAIN_TILT_MARGIN_DEG` (55°).
   - `--set env.termination_if_pitch_greater_than=40 env.termination_if_roll_greater_than=40` takes over, and then the
     margin is not applied.
5. **Foot slip** (P3.2): `--foot-slip [W]`; `--set reward.foot_slip_mode=tangent` (5.6).
6. **Privileged observations** (P3.3): `--privileged critic|teacher`, `--privileged-entries ENTRY ...` (6.6).
7. **Baseline tools** (P3.4): `go2_export.py --onnx --check-envs --check-steps`, `go2_report.py --reps --worlds --ref`,
   [P3v2] `--seeds --deploy-python`, `go2_card.py --name --freeze --strict` (7.7). `--set` may be given several
   times: the lists add up. [P3v2] `--energy 0` removes the energy term that `$BASE` adds.
8. **Constants**: `TERRAIN_DEFAULTS`, `GENERATOR_VERSION`, `FOOTPRINT_HALF`, `BODY_LENGTH` (`go2_terrain.py`);
   `CMD_CURRICULUM`, `TERRAIN_TILT_MARGIN_DEG`, `FOOT_SLIP_WEIGHT`, `PRIVILEGED_DEFAULT` (`go2_train.py`);
   `TERRAIN_FAST_BUILD`, `PRIVILEGED_SIZES`, `HEIGHT_SCAN_*` (`go2_env.py`); `THROUGH_FRAC`, `SLIP_SHARE_V`
   (`go2_metrics.py`); the decision thresholds (`go2_decide.py`); `EXPORT_TOL` (`go2_export.py`); `WORLDS`,
   `FRONTIER_THROUGH`, `FLAT_BINS` (`go2_report.py`); `STANDARD_WORLDS`, `REPORT_MIN_REPS`, `RELEASE_FILES`,
   [P3v2] `BAND_MIN_RUNS` (`go2_card.py`).

Every value ends up in `cfgs.pkl`. The start of the training log prints a `--- terrain, curricula, foot slip,
privileged observations (P3) ---` block under the P2 block. Read it before leaving a run overnight (examples in 4.2,
5.2, 6.2).

### 3.3 Evaluation worlds
`go2_metrics.py` measures on flat ground unless told otherwise, also for runs trained on terrain, so the flat tables
of P1/P2 stay comparable. P3.1 adds three kinds of ground:

| World | Command | Files (tag) | Answers |
|---|---|---|---|
| flat, nominal | `go2_metrics.py -e R --ckpt $CKPT` | `metrics_ckpt1999_seed0.*` | did terrain training cost anything on flat ground? |
| flat, training DR | `... --keep_dr --tag dr` | `..._dr.*` | robustness, as in P2 |
| the run's own tiles | `... --terrain grid` | `..._grid.*` + `terrain_..._grid.png` | how far up each terrain type it gets: the curriculum's own test |
| unseen tiles | `... --terrain grid --terrain_set seed=1 --tag unseen` | `..._grid_unseen.*` | the same, on new random instances (rough and obstacle tiles change; hill, pit and wave are deterministic) |
| beyond training | `... --terrain grid --terrain_set max_slope_deg=30 --types hill pit --tag 30deg` | `..._grid_30deg.*` | where it breaks |
| a slope | `... --slope 15` (`-15`: facing downhill) | `..._slope+15.*` | the full command grid (all speeds, sideways, turning) on one long slope |

- `--terrain grid` evaluates the run's own `terrain.npz` when it has one and no generation key is changed (4.7).
  Otherwise it uses the default grid (P1/P2 runs), so a flat-trained run is measured on the same tiles as a P3.1 run.
- Size: 60 tiles × `--reps` (20) = 1200 envs. Fine on your GPU; on CPU use `--reps 2`, or `--types` / `--levels`.
- `--slope` and `--terrain grid` exclude each other. `--slope` uses the usual 11 command bins.

### 3.4 Comparing
```bash
python go2_compare.py logs/$P2/metrics_ckpt1999_seed0_grid.csv logs/go2-p31/metrics_ckpt1999_seed0_grid.csv \
    --preset terrain                                         # every tile, the terrain columns
python go2_compare.py A_grid.csv B_grid.csv --preset terrain --rows "hill 9" "pit 9" "obstacles 6"
python go2_compare.py logs/go2-p31/metrics_ckpt1999_seed0.csv logs/go2-p31/metrics_ckpt1999_seed0_slope+10.csv \
    logs/go2-p31/metrics_ckpt1999_seed0_slope+20.csv --cols lin_err vx_meas z_err tau_peak CoT through
```
- Grid CSVs match on (terrain, level, command). `--rows "hill 9"` picks a tile; `--rows "hill 9 0.5 0 0"` a tile and a
  command.
- Slope CSVs keep the command rows, so a flat CSV and slope CSVs of one run compare row by row.
- `--preset terrain` shows `through, dist, lin_err, vx_meas, z_err, swing_h, slip_v, nonfoot, tau_peak, CoT`.
- [P3.2] `--preset slip` shows the slip of all feet and of each foot, `slip_share`, `slip_d`, `pen_xy`, `pen_t`, and
  the costs (`lin_err`, `through`, `stride`, `swing_h`, `CoT`). `go2_decide.py` turns the CSVs of two arms × 3 seeds
  into a decision (5.4, 6.4).
- As before, the first file is the reference and `~` means within 2 standard errors over the evaluation envs.
- [P3v2] `~` knows nothing of the **training noise**, which P2 measured far larger. Over 3 identical trainings:

  | at | spread |
  |---|---|
  | `vx_meas`, cmd 0.25 | 0.140–0.196 m/s |
  | `lin_err`, cmd 0.5 / −0.5 | 0.060–0.085 / 0.043–0.082 m/s |
  | `CoT`, cmd 1.0 | 0.595–0.706 |
  | `swing_h`, cmd 0.5 | 6.9–8.7 cm |

  So `go2_compare.py` calls nearly every training difference significant. Judge a change against a band of seeds,
  never against one run: P3.1 against both `$P2` and `$P2S2`, decisions with 3 seeds per arm, candidates with
  `go2_report.py --seeds` (7.6).
- [P3.4] `go2_report.py -e B --ckpt $CKPT --ref A` compares the headline numbers of two runs over the whole standard
  set, with the direction of each (7.6); `go2_compare.py` stays the tool for row-by-row detail.

### 3.5 TensorBoard
```bash
tensorboard --logdir logs
```
P3.1 adds a `Curriculum/` section. rsl_rl logs every `extras["episode"]` key that contains a `/` under its own name,
and the rest under `Episode/`.

| Tag | Meaning | Start → healthy end (defaults) |
|---|---|---|
| `Curriculum/terrain_level` | mean level over all envs | 2.0 (levels drawn in 0–4) → toward **6.0**, the ceiling when every robot passes every level (4.1.4) |
| `Curriculum/top_level_share` | share of envs on the last level | ~0 → toward **18 %** (same ceiling) |
| `Curriculum/level_<type>` | mean level of the envs on each type | `flat` reaches ~6 first; the hard types lag |
| `Curriculum/slope_hill_deg`, `slope_pit_deg` | mean level × 25°/9 = the mean face slope trained on | → 16.7° at level 6.0 |
| `Curriculum/cmd_vx_max`, `cmd_vy_max`, `cmd_wz_max` | current command ranges (half-widths) | 0.5 / 0.25 / 0.5 → 1.0 / 0.5 / 1.0 |
| `Curriculum/cmd_score_lin`, `cmd_score_yaw` | last score: mean per-step tracking reward ÷ its weight (0–1) | ranges grow while > 0.8 |
| `Episode/rew_*` | as in P2; on terrain the tracking terms read lower (hard tiles), `base_height` and `foot_clearance` now measure from the ground | |
| `Train/mean_episode_length` | lower on terrain early (falls), rising as the levels rise | |
| `Perf/total_fps` | compare with your P2 run: the heightfield contacts and the per-step height lookups cost a little | |
| `Episode/rew_foot_slip` | [P3.2] the foot-slip penalty, with `--foot-slip` | negative, shrinking toward 0 as the feet plant |
| `Loss/value` | the critic's loss; [P3.3] lower with a privileged critic (it sees more), which alone does not mean a better policy | |

### 3.6 Visual checks
```bash
python go2_terrain.py                                     # tile table + terrain_grid.png (default grid)
python go2_terrain.py --run logs/go2-p31                  # the grid this run trained on
python go2_teleop.py -e go2-p31 --ckpt $CKPT --terrain    # drive it; J / K level, U type
python go2_teleop.py -e $P2 --ckpt $CKPT --terrain --tile pit 6   # a flat-trained policy on the same tile
python go2_teleop.py -e go2-p31 --ckpt $CKPT --slope 20
PYOPENGL_PLATFORM=egl python go2_teleop.py -e go2-p31 --ckpt $CKPT --terrain --tile hill 9 --headless \
    --cmd 0.5 0 0 --duration 10 --record hill9.mp4        # video, no display
python go2_feet_plot.py logs/go2-p31/metrics_ckpt1999_seed0_grid.npz --tile obstacles 9 --target 0.075
python go2_fsm_sim.py -e go2-p31 --slope 15               # the deploy state machine on a slope
```

---

## 4. P3.1 Curriculum training

### 4.1 What the code does, and why

#### 4.1.1 The grid
```
 y (columns: one per type, `copies` columns per type)
 ^
 |  border 5 m (flat)
 |  +--------+--------+-- ... --+--------+
 |  | obst 0 | obst 1 |         | obst 9 |      each tile: 8 m x 8 m, heights every 0.1 m
 |  +--------+--------+-- ... --+--------+      tile (l, c) spans x in [8l, 8l+8], y in [8c, 8c+8]
 |  |  ...   |        |         |        |      world origin = corner of tile (0, 0)
 |  +--------+--------+-- ... --+--------+      robots spawn at a tile centre, +-0.5 m
 |  | flat 0 | flat 1 |         | flat 9 |
 |  +--------+--------+-- ... --+--------+
 +---------------------------------------------> x (rows: level 0 ... 9 = difficulty 0 ... 1)
```
- Default: 10 levels × 6 columns (flat, rough, hill, pit, wave, obstacles) = 80 m × 48 m + a 5 m border, 901 × 581
  vertices.
- Env columns follow the type shares (10 / 20 / 17.5 / 17.5 / 15 / 20 %) deterministically: env i takes the column
  at quantile (i + 0.5) / N, so 4096 envs always split the same way.
- Every tile is exactly 0 on its outer ring of vertices, so neighbours join without a step whatever their types.
  Walking +x from a tile centre leads onto the next level of the same type; walking ±y leads onto another type.
- Because every tile is 0 at its edge:
  - a **hill** spawns its robots on top (1.40 m up at level 9), so walking away is **downhill**;
  - a **pit** spawns them at the bottom, so walking away is **uphill**.

#### 4.1.2 Tile types per level
Measured by `python go2_terrain.py` on the default grid. Each cell is "steepest slope by construction / measured
(95th percentile of the triangle slopes) [deg], height range [m]":

| level | flat | rough | hill | pit | wave | obstacles |
|---|---|---|---|---|---|---|
| 0 | 0 / 0 | 0 / 2.4, ±0.00 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 11.3, ±0.02 |
| 1 | 0 / 0 | 0 / 3.8, ±0.01 | 2.8 / 2.8, 0..+0.15 | 2.8 / 2.8, −0.15..0 | 1.5 / 1.4, ±0.02 | 0 / 16.1, ±0.03 |
| 3 | 0 / 0 | 0 / 6.3, ±0.01 | 8.3 / 8.3, 0..+0.44 | 8.3 / 8.3, −0.44..0 | 4.5 / 4.3, ±0.05 | 0 / 25.0, ±0.05 |
| 5 | 0 / 0 | 0 / 9.0, ±0.02 | 13.9 / 13.9, 0..+0.74 | 13.9 / 13.9, −0.74..0 | 7.5 / 7.2, ±0.08 | 0 / 32.8, ±0.06 |
| 7 | 0 / 0 | 0 / 11.4, ±0.02 | 19.4 / 19.4, 0..+1.06 | 19.4 / 19.4, −1.06..0 | 10.4 / 10.0, ±0.12 | 0 / 39.4, ±0.08 |
| 9 | 0 / 0 | 0 / 14.1, ±0.03 | 25.0 / 25.0, 0..+1.40 | 25.0 / 25.0, −1.40..0 | 13.3 / 12.7, ±0.15 | 0 / 45.0, ±0.10 |

- **rough**: random heights on a 0.2 m grid, linear in between, ±0.5 cm (level 0) to ±3 cm (level 9). The measured
  slope is local roughness (14° at level 9 over 0.2 m), not a slope the base follows.
- **hill / pit**: 4 planar faces at level × 2.78° and a 2 m flat platform. Robots spawn on the platform and reach the
  face after 1 m.
  - Faces are planar (height ∝ distance to the nearest edge), so the measured slope equals the nominal one.
  - legged_gym and Isaac Lab use bilinear pyramids, whose slope varies over a face.
  - Single inclines (Genesis `sloped_terrain`) are not used: a 25° incline across 8 m rises 3.7 m and leaves a cliff
    to the next tile.
- **wave**: A·sin·sin with whole half-waves per tile (4 m wavelength), up to ±15 cm and 13.3°: bumps and dips under
  every foot.
- **obstacles**: 50 rectangles of 0.3–1 m per tile, raised or sunk by h or h/2, with h from 2 cm (level 0) to 10 cm
  (level 9), and a flat 2 m centre.
  - The "measured slope" is the 45° ramp a 10 cm step makes over one 0.1 m cell, i.e. the edges.
  - The P2.2 swing target is 7.5 cm, so levels 7–9 (h = 8–10 cm, with h/2 steps in between) ask a blind policy to
    clear, or stumble over, edges near or above its normal swing height.

#### 4.1.3 Slopes up to 25° and friction
`max_slope_deg` = 25 is the top level of hill and pit. tan 25° = 0.47 is above the lowest friction of the P1
randomization (μ = 0.4, `friction_range` [0.4, 1.25]).
- **An env drawn with μ < 0.47 cannot stand still on a top-level face**, whatever the policy. About 8 % of the envs
  are drawn below 0.47.
- The terrain curriculum is per env, so these envs simply stay lower; nothing breaks.
- But when you read the top levels **with `--keep_dr`**, part of the failures is physics, not policy. The nominal
  evaluation (μ = 1) has no such limit.

#### 4.1.4 The terrain curriculum
Rudin et al. 2022's game-inspired curriculum, as in legged_gym `_update_terrain_curriculum` (Isaac Lab
`terrain_levels_vel` is the same rule). It is judged per env, when its episode ends (time-out or fall):

| Outcome of the episode | Next level |
|---|---|
| walked more than `up_distance` (half a tile, 4 m) straight-line from its spawn point, and did not fall | +1 |
| fell (`fall_moves_down`), or covered less than `down_ratio` (0.5) of the distance its commands asked for (when they asked for ≥ 1 m) | −1 (not below 0) |
| otherwise | same |
| +1 from the top level | a random level (keeps every level in the mix) |

Start levels are uniform in 0–`max_init_level` (4).

**One change from legged_gym:** legged_gym moves a robot down when it walked less than half of *last command ×
episode length*. With commands resampled every few seconds, that judges a robot on the command it had at the end.
Here the commanded distance and the progress along the command are integrated over the whole episode (`cmd_dist`,
`progress` in `step()`), so a robot is judged on what it was asked during the episode.

**The ceiling of `Curriculum/terrain_level` is 6.0, not 9.** A robot that passes every level climbs one level per
episode until the top, then restarts at a random level. Level k is visited in a share (k + 1)/55 of the episodes, so:
- the mean level is 6.0;
- the top-level share is 10/55 = 18 %.

A type whose `Curriculum/level_<type>` sits at ~6 is **solved**. Below that, the type's difficulty is holding robots
back.

**The terrain curriculum depends on the command ranges.** "Up" needs a 4 m straight-line displacement in one episode,
and commands change direction every 4 s. I simulated a perfect tracker with uniform commands (10 % standing, the
0.2 deadzone; this ignores the small-command and single-axis samplers) over a 20 s episode:

| Command ranges | Resampling | Mean displacement | P(> 4 m) = "up" | P(> 3 m) |
|---|---|---|---|---|
| ±0.5 / ±0.25 / ±0.5 (command curriculum start) | 4 s | 2.3 m | 8 % | 25 % |
| ±0.5 / ±0.25 / ±0.5 | 10 s | 2.7 m | 23 % | 41 % |
| ±1 / ±0.5 / ±1 (full ranges) | 4 s | 4.0 m | 46 % | 64 % |
| ±1 / ±0.5 / ±1 | 10 s | 3.6 m | 36 % | 48 % |

So, at the command curriculum's start ranges, a robot that handles its tile is promoted in only ~1 episode in 12.
Terrain progress is therefore slow until the command ranges have grown.

That is why the runs in 4.3 separate the two curricula first. The fix, if `go2-p31` climbs too slowly, is
`--set env.terrain.up_distance=3.0`. legged_gym avoids the issue with heading commands (straighter paths) and 10 s
resampling.

#### 4.1.5 The command curriculum
legged_gym `update_command_curriculum` grows only vx, by ±0.5 m/s, when the episodes' mean linear tracking reward
exceeds 0.8 of its maximum. P3.1 applies the same idea to all three axes:

- The ranges start at `CMD_CURRICULUM["start"]`: vx ±0.5, vy ±0.25, wz ±0.5.
- Every `interval_s` (20 s of simulated time, one episode, ~42 iterations), it scores the episodes that ended since
  the last check and lasted at least 1 s:
  - linear score = mean per-step `tracking_lin_vel` reward ÷ its weight;
  - yaw score = the same for `tracking_ang_vel`.
- If the linear score is > `threshold` (0.8), vx and vy grow by `step` (0.1 and 0.05 per side). If the yaw score is
  > 0.8, wz grows by 0.1.
- Never beyond the configured ranges (±1, ±0.5, ±1).

What 0.8 means:
- The tracking reward is exp(−e²/σ) with σ = 0.25, so a mean per-step reward of 0.8 corresponds to an error of about
  **0.24 m/s** (0.24 rad/s for yaw) on average. Lower the threshold to accept worse tracking before growing.
- From ±0.5 to ±1 m/s takes 5 successful checks: at least 100 s of simulated time, about 210 iterations.

Saved and used afterwards:
- Every checkpoint stores the ranges reached (`infos["curriculum"]`).
- `go2_metrics.py` prints them, and marks commands outside them `*` in the ood column.
- `go2_export.py` deploys them (4.4.6).
- `go2_teleop.py` stops its keys there.

On terrain, the score includes every env, also those on hard tiles. A curriculum that reached hard levels therefore
scores lower, which slows the command growth. That is the second coupling of the two curricula (see the
symptoms in 4.9).

#### 4.1.6 Terminations on terrain
- **Tilt.** A base held parallel to a 25° face already pitches 25°, so the P1.2 limit (30°) would end good climbs.
  - With `--terrain` the limit is `max_slope_deg` + 30° = 55°.
  - The real fall criterion stays base contact. legged_gym's and Isaac Lab's rough-terrain tasks have no tilt
    termination at all.
- **Off the grid.** The base drops 1 m below the lowest ground (a robot that walked off the border).
- `go2_metrics.py` keeps its own fall criterion for every run: `--fall_deg` (10°) **plus the tilt the ground itself
  causes** (`tile_tilt`):
  - hill, pit, wave: the tile's slope;
  - rough: atan(2·amplitude / 0.4 m);
  - obstacles: atan(h / 0.4 m);
  - `--slope`: |DEG|.

  0.4 m is about the Go2's front-to-rear foot spacing (`BODY_LENGTH`).

#### 4.1.7 What is measured from the ground
On terrain, `go2_env.py` looks up the ground height under each foot every step (`feet_ground` [N, 4]) and defines
`base_height` = base z − the mean of those four heights. legged_gym uses the mean of the measured heights under the
body.
- `base_height` reward: `base_height` vs 0.30 m (+ dh).
- `foot_clearance` reward (P2.2): each foot above the ground under it.
- Recovery terms (P1.5) and fallen starts: from the ground.
- `go2_metrics.py`: `z`, `z_err`, `swing_h`, and the feet plot.

On flat ground all of these are exactly what they were (the lookup returns 0).

#### 4.1.8 Spawning and resets
- A robot spawns at its tile centre ± `spawn_xy_noise` (0.5 m, inside the 2 m platforms), with a random heading
  (`random_yaw`).
- Its base is 0.42 m above the highest ground under a 0.8 m × 0.8 m footprint (`FOOTPRINT_HALF`), so it never spawns
  inside a bump.
- On a `--slope` plane, robots spawn standing on it: base parallel to the plane, 0.42 m above it along its normal.
- **Resets on terrain run forward kinematics** (`reset_forward`). Genesis' `set_qpos(skip_forward=True)`, used by the
  example's reset, leaves the joint readings at their pre-reset values until the next physics step. So the explicit
  PD of P1.2 kicked every freshly reset robot from its old joint angles for one 5 ms substep, up to the torque
  limit.
  - Flat runs keep that behaviour, so they stay identical to P2.
  - `--set env.reset_forward=True` removes it there too; I suggest it for the v1.0 baseline (P3.4).
  - Cost: about 3 % per step on CPU here (200 steps × 64 envs: 8.7 s → 9.0 s), since the reset runs every step with
    its mask.

#### 4.1.9 What does not change
The observation (45 / 46 / 54 entries), the actions, the export format and the robot side. The policy is **blind**: it
has to infer the ground from its IMU and joint states. Slopes, waves and rough ground are feasible blind; steps near
the swing height (obstacles 8–10 cm) are where a blind policy is expected to stumble. Seeing the terrain (height scan)
is a privileged input for P3.3's teacher, not for the deployed policy.

### 4.2 First session: smoke tests (CPU, ~10 min)
```bash
python go2_terrain.py                       # check terrain_grid.png: pits blue, hills red, one colour scale
python go2_terrain.py --check               # "ok" x 6 (the last one is P3.2's slip check)
python go2_train.py -e smoke-p31 --cpu -b 32 --max-iterations 3 --terrain --cmd-curriculum \
    --set env.terrain.num_levels=3 env.terrain.horizontal_scale=0.2     # small, coarse grid: builds in seconds
python go2_terrain.py --run logs/smoke-p31  # the grid it saved (logs/smoke-p31/terrain.npz)
python go2_metrics.py -e smoke-p31 --ckpt 2 --cpu --reps 2 --terrain grid
python go2_metrics.py -e smoke-p31 --ckpt 2 --cpu --reps 2 --slope 15
python go2_export.py -e smoke-p31 --ckpt 2  # prints the reached command ranges (still the start ranges)
python go2_teleop.py -e smoke-p31 --ckpt 2 --terrain          # J / K / U in the viewer
python go2_fsm_sim.py --slope 15            # zero-action policy: stand-up and hand-over on a slope
```
With the default terrain (no `--set`), the training log prints this block under the P2 block:

```
--- terrain, curricula, foot slip, privileged observations (P3) ---
P3.1 terrain grid 10 levels x flat 0.1, rough 0.2, hill 0.175, pit 0.175, wave 0.15, obstacles 0.2; tiles 8 m, cells 0.1 m, border 5 m, seed 0; top level: hill/pit 25 deg, rough +-3.0 cm, waves 15 cm, obstacles 10 cm
     terrain curriculum on: start levels 0..4, up after 4 m, down below 0.5 x the commanded distance or after a fall; spawn +-0.5 m, random heading; tilt termination 55 deg
P3.1 command curriculum from {'lin_vel_x': [-0.5, 0.5], 'lin_vel_y': [-0.25, 0.25], 'ang_vel': [-0.5, 0.5]}, +{'lin_vel_x': 0.1, 'lin_vel_y': 0.05, 'ang_vel': 0.1} per 20 s while tracking > 0.8 x max, up to vx [-1.0, 1.0] vy [-0.5, 0.5] wz [-1.0, 1.0]
P3.2 foot-slip penalty off
P3.3 privileged observations off (actor and critic read the policy observation)
```
(The block was titled `--- terrain and curricula (P3) ---` before P3.2.) A smoke policy falls everywhere and walks nowhere. The point is that every script runs, and that the files exist:
`terrain.npz`, `metrics_ckpt2_seed0_grid.csv`, `terrain_ckpt2_seed0_grid.png`, `exported/deploy_cfg.json` with
`command_ranges_configured`.

**First GPU session** (before the long runs):
```bash
python go2_train.py -e go2-p31-test $BASE --terrain --cmd-curriculum --max-iterations 50
```
Note:
- the build time: 35 s here for the default grid (flat: 15 s), 4.3 GB of RAM at the peak; it does not depend on the
  number of envs;
- `Perf/total_fps` against your P2 run;
- that the `Curriculum/*` tags appear in TensorBoard.

### 4.3 Training runs
```bash
# 1) The two curricula one at a time (1 seed each)
python go2_train.py -e go2-p31c $BASE --cmd-curriculum                  # flat ground, growing commands
python go2_train.py -e go2-p31t $BASE --terrain                          # terrain, full command ranges
# 2) P3.1: both (3 seeds: they are also the "without" arm of the P3.2 decision, 5.3)
python go2_train.py -e go2-p31 $BASE --terrain --cmd-curriculum
python go2_train.py -e go2-p31-s2 $BASE --seed 2 --terrain --cmd-curriculum
python go2_train.py -e go2-p31-s3 $BASE --seed 3 --terrain --cmd-curriculum
# 3) Optional: from your P2 policy (fresh optimizer, iteration 0; P2 checkpoints carry no curriculum state)
python go2_train.py -e go2-p31i $BASE --terrain --cmd-curriculum --init-from logs/$P2/model_$CKPT.pt
# 4) Continue a run whose levels were still rising at the end (weights and curriculum state)
python go2_train.py -e go2-p31-more $BASE --terrain --cmd-curriculum --init-from logs/go2-p31/model_$CKPT.pt
# 5) [P3v2] Only if hill / pit levels stall (4.5): the same without the energy term, to see whether energy is the cause
python go2_train.py -e go2-p31-noE $BASE --energy 0 --terrain --cmd-curriculum
```
- `$BASE` holds the flags of `$P2` (`--symmetry --energy 1.5e-3`): P3.1 then differs from P2 by its P3 flags only.
- With `--init-from`, the network shape must match: same observation (e.g. both with or without `--height-cmd`),
  same `--rnn`.
- `--init-from` restarts at iteration 0 with a fresh optimizer. The action noise std is part of the actor's weights,
  so it continues where it was.
- Checkpoints of the new run start again at `model_0.pt`.

Why this order:
- `go2-p31c` tells you what the command curriculum alone does on flat ground: faster early learning, or nothing.
- `go2-p31t` gives the terrain curriculum full command ranges, so its "up" rule fires at its full rate (4.1.4).
- `go2-p31` combines them. If it climbs clearly slower than `go2-p31t`, the coupling of 4.1.4 is the reason: re-run
  with `--set env.terrain.up_distance=3.0`.

### 4.4 Evaluation and how to read it

#### 4.4.1 Commands
```bash
for R in $P2 $P2S2 go2-p31c go2-p31t go2-p31 go2-p31-s2 go2-p31-s3; do
    python go2_metrics.py -e $R --ckpt $CKPT                                     # flat
    python go2_metrics.py -e $R --ckpt $CKPT --terrain grid                      # own tiles (P2: default grid)
done
python go2_metrics.py -e go2-p31 --ckpt $CKPT --terrain grid --terrain_set seed=1 --tag unseen
python go2_metrics.py -e go2-p31 --ckpt $CKPT --keep_dr --tag dr
for S in 10 -10 20 -20 25 -25; do python go2_metrics.py -e go2-p31 --ckpt $CKPT --slope $S; done
python go2_compare.py logs/$P2/metrics_ckpt1999_seed0_grid.csv logs/go2-p31{t,}/metrics_ckpt1999_seed0_grid.csv \
    --preset terrain --rows "hill 3" "hill 6" "hill 9" "pit 3" "pit 6" "pit 9" "obstacles 6" "obstacles 9"
python go2_compare.py logs/$P2/metrics_ckpt1999_seed0.csv logs/go2-p31/metrics_ckpt1999_seed0.csv --preset gait
python go2_compare.py logs/$P2/metrics_ckpt1999_seed0.csv logs/$P2S2/metrics_ckpt1999_seed0.csv --preset gait  # P2 band
```
- [P3v2] Read a P3.1 difference on flat ground against the P2 band (`$P2` vs `$P2S2`, last line) and P3.1's own
  three seeds, not against one P2 run (3.4).

#### 4.4.2 The grid tables (`--terrain grid`)
- **Rows.** One row per tile (type, level). Every robot of the row:
  - starts at the tile centre, facing +x, with `--grid_cmd` (0.5 0 0);
  - runs for as long as a perfect tracker needs to walk half a tile, i.e. to the tile edge: 8 s at 0.5 m/s, of which
    1 s of warm-up and 7 s measured.
- **Direction.** +x leads toward the next level of the same type, so walking out of the tile is:
  - on hill: down the face;
  - on pit: up the face;
  - on the others: across the relief.
- **`falls`**: base contact, or a tilt beyond `--fall_deg` + the tile's tilt (4.1.6).
- **Progress table** (only in the terrain modes):
  - `dist[m]`: straight-line horizontal distance from the start point, at the end or at the fall. It is the
    curriculum's own measure. A perfect tracker reaches 4 m on flat tiles, slightly less on faces (it tracks 0.5 m/s
    along the slope, so it covers 0.5·cos θ m/s horizontally).
  - `through[%]`: share of robots with `dist` ≥ `--through` (0.75) × the planned distance and no fall: the success
    rate of the tile. A perfect tracker passes even on a 25° face (cos 25° = 0.91 > 0.75).
- **Tracking.** `lin_err`, `vx_meas` are in the body frame, so on a face they are along the slope.
  - Uphill (pit), `vx_meas` below the command with `tau_peak` near 100 % = the legs are at their torque limit.
  - Downhill (hill), overspeed or falls = braking problems.
- **`z_err`, `z`**: base above the mean ground under the feet, vs 0.30 m. On obstacles and rough tiles, `z_err` rises
  with the relief under the feet even for a perfect gait, so compare runs on the same tile, not across tiles.
- **`swing_h`**: peak clearance above the ground under the foot. On obstacles the ground under a foot changes during
  the swing, so the number is noisier.
- **`CoT`**: lifting the robot uphill costs m·g·v·sin θ, so CoT rises by about sin θ: +0.17 at 10°, +0.34 at 20°,
  +0.42 at 25°.
  - Downhill it rises too: the motors brake, and `|τ·q̇|` counts negative work as cost (no regeneration).
  - A CoT uphill far above flat + sin θ = an inefficient climbing gait.
- **Foot slip** (`slip_v`, `slip_d`): since P3.2, measured along the ground under each foot, per foot too (5.1.2).
- **ood**: commands outside the ranges the checkpoint's command curriculum reached are marked `*`.

#### 4.4.3 The heat map (`terrain_<name>.png`)
One cell per (type, level):
- colour = `through[%]` (light = few got through, dark blue = all did);
- the number = `through`;
- below it, the share of robots that fell.

Read it in this order:
1. **The frontier per type**: the first level where `through` drops below ~80 %. That is the type's difficulty limit
   for this policy.
2. **Hill vs pit at the same level**: downhill vs uphill on the same slope. The two directions can differ a lot; the
   `--slope ±DEG` tables then tell which part of the gait fails.
3. **Obstacles**: where `through` falls while the falls stay low, robots get stuck behind edges rather than tumbling.
   `dist` tells how far they got.
4. **P2 vs P3.1** (two PNGs, or `go2_compare.py --preset terrain`): the frontier should move right on every type,
   and the low levels stay at ~100 %.

#### 4.4.4 Unseen tiles and beyond training
- `--terrain_set seed=1 --tag unseen`: new random rough and obstacle tiles. Hill, pit and wave tiles are deterministic
  and come out identical unless `slope_noise` > 0. `through` within a few % of the own tiles = no overfitting to the
  exact tile instances.
- `--terrain_set max_slope_deg=30 --types hill pit --tag 30deg`: levels at 0–30°. Read where it breaks beyond 25°.
- Any generation key in `--terrain_set` (4.7) means a newly generated grid, not the run's `terrain.npz`. The title
  line prints which grid was used.

#### 4.4.5 Slope tables (`--slope DEG`)
- The usual 11 command bins on a plane through the origin, rising along +x: the robot faces +x, so
  - forward = uphill for +DEG and downhill for −DEG;
  - vy bins walk across the slope;
  - wz bins turn on it.
- Robots start standing on the plane (base parallel to it).
- `falls` uses `--fall_deg` + |DEG|. Heights are measured from the plane.
- The Progress table adds `dist` and `through` per command. They are n/a for turning bins, because a curved path ends
  closer than planned even for a perfect tracker.

What to read:
- **Up vs down**: `vx_meas` at (0.5, 0, 0) on +DEG vs −DEG.
- **Drift**: `vy_meas` and `wz_meas` on the sideways bins. Expect a robot crossing a slope to drift or turn downhill.
- **Standing**: the standing bin (0, 0, 0) on a slope is a real test. Watch `vx_meas` (creep) and falls.
- **Effort**: `tau_peak`, `tau_sat` uphill.

Compare `flat → +10 → +20 → +25` for one run (3.4), then the same for `$P2`.

#### 4.4.6 Export and the reached command ranges
```bash
python go2_export.py -e go2-p31 --ckpt $CKPT
```
For a `--cmd-curriculum` run the export prints the ranges reached, and writes them as `command_ranges` in
`deploy_cfg.json`. The configured ones move to `command_ranges_configured`.
- `go2_fsm.py` scales the gamepad sticks to `command_ranges`, so the robot never asks for more than the policy trained
  on. If the curriculum stopped early (say vx ±0.8), full stick = 0.8 m/s.
- Terrain runs also get `training_terrain`: types, mean level reached per type, top-level share. It is a record only;
  nothing on the robot reads it.

Then the usual P2 sim-to-sim loop (TEST_AND_TINKER_P2.md 3.8) works unchanged. The deploy path is flat ground.

#### 4.4.7 What I expect (hypotheses, not measurements)
| | `$P2` (flat-trained) on the grid | `go2-p31` |
|---|---|---|
| flat, rough 0–4, wave 0–3 | `through` ≈ 100 % | same |
| hill / pit | falls off around the middle levels (8–14°) | high to levels 7–9 |
| obstacles 6–9 (7–10 cm) | low | better, not 100 % (blind) |
| flat tables (no `--terrain`) | reference | `~` on tracking; `swing_h` and CoT may rise a little (a more careful gait) |
| slopes ±20° | `vx_meas` off, falls | tracks, CoT ≈ flat + sin θ |

If `go2-p31`'s flat tables are clearly worse than `$P2`'s, terrain training cost flat performance: check the flat
share (10 %) and `Curriculum/level_flat`.

### 4.5 Reading TensorBoard during training
- **First ~100–300 iterations.** `Curriculum/terrain_level` may fall below its start (2.0): robots that cannot walk
  yet fall, and falls move them down. It turns up once they walk.
- **Then.** It rises toward 6.0. Read it per type (`Curriculum/level_<type>`):
  - `flat` should reach ~6 quickly (nothing to learn);
  - `rough` and `wave` follow;
  - `hill` / `pit` / `obstacles` are the ones to watch. `Curriculum/slope_*_deg` translates their levels into the
    mean face slope trained on.
- **Command ranges.** `Curriculum/cmd_*_max` should reach their maximum. If they stall at the start values, read
  `Curriculum/cmd_score_lin` / `_yaw`: a score stuck below 0.8 means tracking is not good enough to grow (on terrain,
  partly because of the hard tiles, 4.1.5).
- **Rewards.**
  - `Episode/rew_tracking_*` reads lower than in P2 on terrain: the hard tiles count.
  - `Train/mean_reward` falling while the levels rise is normal: the robots are on harder ground.
- **When to stop.** Levels still rising at the end of the run = more training is useful: continue with `--init-from`
  (4.3, run 4).
- [P3v2] **Energy on the slopes.** `$BASE` holds `--energy 1.5e-3`, and climbing costs much more power than flat
  walking. Lifting about 15 kg up a 25° face at 0.5 m/s takes 15 × 9.81 × 0.5 × sin 25° ≈ 31 W, more than the final
  P2 policy's whole flat-walking power at that speed (24–28 W), and Σ|τ·q̇| counts more than the net work. So the term
  weighs more on hill and pit than its flat calibration suggests; P2.5 saw swing height fall 20–49 % at W = 4e-3 on
  flat. I expect it may slow `Curriculum/level_hill` and `level_pit` (not tested). Compare
  `Episode/rew_energy` with `Episode/rew_tracking_lin_vel`; if those levels stall while `level_flat` and `level_rough`
  climb, train `go2-p31-noE` (4.3, run 5) or a lower W, and compare the levels.

### 4.6 Visual checks
- **`go2_terrain.py`**: `terrain_grid.png` shows the heights on a diverging scale (pits blue, hills red, ground grey)
  with one colour scale for the whole grid, so small relief (rough, obstacles) looks faint next to the 1.4 m pyramids.
  The printed table (4.1.2) has the numbers.
- **`go2_teleop.py --terrain`**: starts on a hill at the middle level (`--tile TYPE LEVEL` to choose).
  - J / K: previous / next level; U: next type. Each respawns the robot at the tile centre and prints the tile's
    slope and tilt.
  - For a flat-trained run, the tilt termination is raised by the grid's steepest tilt, so standing on a 25° face is
    not a fall.
  - The keys stop at the command ranges the curriculum reached (`--configured_ranges` to go beyond).
  - Use it to watch a P2 and a P3.1 policy on the same tile.
- **`go2_feet_plot.py <grid npz> --tile TYPE LEVEL`**:
  - clearance above the ground under each foot. On rough and obstacle tiles the curve jumps where the ground under a
    swinging foot steps;
  - the speed is measured along the ground under the foot (since P3.2, 5.1.2).
- **`go2_fsm_sim.py --slope 15`**:
  - the robot lies aligned with the slope, facing +x (uphill), stands up, hands over to the policy, walks uphill with
    the stick, sits down;
  - heights are logged above the ground and tilts from the ground normal, so the checks (standing: z > 0.25 m, tilt
    < 10°) keep their meaning;
  - `--slope -15` faces downhill.

  For scale, the zero-action policy (default pose with the policy gains) on ±15°: stand-up ends at 0.33 m with ≤ 3°
  tilt, then the soft policy gains let it sag 8–11 cm and pitch 14–31° from the normal (facing downhill is worse).
  A trained terrain policy should hold the hand-over far better. The bench is a way to check that.

### 4.7 Knobs

**Terrain keys** (`--set env.terrain.KEY=VALUE`; G = generation key: changes the heights, so a new grid, cached):

| Key | Default | Unit | G | Meaning / when to change |
|---|---|---|---|---|
| `types` | flat .10, rough .20, hill .175, pit .175, wave .15, obstacles .20 | env shares | G | one column per type; shares need not sum to 1 |
| `copies` | 1 | | G | columns per type (more random instances, wider grid) |
| `num_levels` | 10 | | G | rows; levels are d = l/(L − 1) |
| `tile_size` | 8.0 | m | G | legged_gym's value; also sets `up_distance` (half) |
| `border` | 5.0 | m | G | flat margin; legged_gym uses 25 m, which costs RAM at build (4.8) |
| `horizontal_scale` | 0.1 | m | G | cell size; 0.2 for smoke tests only (coarse rough tiles) |
| `vertical_scale` | 0.005 | m | | unit of the heightfield passed to Genesis (heights stay floats) |
| `seed` | 0 | | G | terrain generation only (not the training seed) |
| `rough_amp` | (0.005, 0.03) | m | G | ± height at level 0 → level 9 |
| `rough_cell` | 0.2 | m | G | spacing of the random heights |
| `max_slope_deg` | 25.0 | deg | G | hill and pit faces at the top level; also sets the tilt termination |
| `platform` | 2.0 | m | G | flat centre of hill, pit and obstacle tiles |
| `slope_noise` | 0.0 | m | G | ± rough noise on hill and pit faces (legged_gym "rough slope": 0.05) |
| `wave_amp` | (0.0, 0.15) | m | G | wave amplitude |
| `wave_length` | 4.0 | m | G | snapped to whole half-waves per tile |
| `obstacle_height` | (0.02, 0.10) | m | G | h; steps are ±h and ±h/2 |
| `obstacle_size` | (0.3, 1.0) | m | G | rectangle sides |
| `obstacle_count` | 50 | | G | per tile |
| `spawn_xy_noise` | 0.5 | m | | ± around the tile centre (keep < platform/2) |
| `random_yaw` | True | | | random heading at each reset |
| `spawn_tile` | None | | | e.g. `["hill", 9]`: every env on that tile, curriculum off (`go2_teleop.py`) |
| `curriculum` | True | | | False: levels stay where they start |
| `max_init_level` | 4 | | | start levels 0..this (legged_gym: 5); 0 = everyone starts easy |
| `up_distance` | None (= tile/2 = 4 m) | m | | lower (3.0) when terrain progress is slow at small command ranges (4.1.4) |
| `down_ratio` | 0.5 | | | lower (0.3) = harder to move down |
| `down_min_cmd_dist` | 1.0 | m | | judge progress only if the episode asked for at least this |
| `fall_moves_down` | True | | | False: falls alone do not move down |
| `slope_deg` | 15.0 | deg | | mode `"slope"` only (`go2_metrics.py --slope`, `go2_teleop.py --slope`, `go2_fsm_sim.py --slope`) |
| `mode` | `"grid"` | | | `"slope"` = one tilted plane (evaluation and bench) |

Recipes:
```bash
--terrain --set "env.terrain.types={'flat':0.1,'hill':0.45,'pit':0.45}"   # slopes only: does the slope frontier move?
--terrain --set env.terrain.max_slope_deg=20 env.terrain.num_levels=8      # gentler top, 2.9 deg per level
--terrain --set env.terrain.max_init_level=0                               # strict curriculum: everyone at level 0
--terrain --set env.terrain.curriculum=False env.terrain.max_init_level=9  # no curriculum: uniform levels (ablation)
--terrain --set env.terrain.slope_noise=0.03                               # rough faces
```
The last-but-one line is the ablation for the thesis: the same tiles, levels drawn uniformly and never changed.

**Command curriculum** (`--cmd-curriculum`, then `--set command.curriculum.KEY=VALUE`):

| Key | Default | Meaning |
|---|---|---|
| `start` | vx ±0.5, vy ±0.25, wz ±0.5 | initial ranges (within the configured ones) |
| `step` | vx 0.1, vy 0.05, wz 0.1 | growth per side and check |
| `threshold` | 0.8 | score needed to grow (0.8 ≈ 0.24 m/s mean error) |
| `interval_s` | 20.0 | time between checks (one episode) |
| configured ranges | `command.lin_vel_x_range` [−1, 1], `lin_vel_y_range` [−0.5, 0.5], `ang_vel_range` [−1, 1] | the maximum; widen them (e.g. vx [−1.5, 1.5]) to let the curriculum go further |

**Other P3.1 knobs**

| Knob | Where | Default | Effect |
|---|---|---|---|
| tilt margin | `TERRAIN_TILT_MARGIN_DEG` (`go2_train.py`), or `--set env.termination_if_*` | 30° (limit 55°) | lower = terminate earlier on slopes |
| reset with forward kinematics | `--set env.reset_forward=True/False` | on for terrain, off on flat | 4.1.8 |
| terrain build fix | `TERRAIN_FAST_BUILD` (`go2_env.py`) | True | False = Genesis' own build (minutes, GBs) for comparison |
| `--through` | `go2_metrics.py` | 0.75 | share of the planned distance that counts as "through" |
| `--grid_cmd` | `go2_metrics.py` | 0.5 0 0 | 1.0 0 0 = the same tiles at twice the speed; 0 0.3 0 = sideways |
| `--fall_deg` | `go2_metrics.py` | 10 | + the ground's tilt (4.1.6) |
| `--warmup_s`, `--measure_s` | `go2_metrics.py` | grid: 1 s and "half a tile"; flat / slope: 2 s and 10 s | |
| `--types`, `--levels` | `go2_metrics.py` | all | subsets (faster) |

### 4.8 Build cost and memory
Genesis turns the heightfield into a mesh at `scene.build()`, once per process. Every training, evaluation and teleop
start pays it. Measured here: env with 1 robot, CPU backend, 2 CPU cores and 7 GB of RAM, Genesis' kernels already
compiled (the first run after installing Genesis compiles them once, which adds about a minute):

| Ground | Vertices | Env build | Peak RAM |
|---|---|---|---|
| flat plane (P2) | – | 15 s | 1.5 GB |
| smoke grid (3 levels, 0.2 m cells) | 50k | 15 s | 1.7 GB |
| `--check --genesis` grid (5 levels, 4 m border) | 270k | 23 s | 3.0 GB |
| **default** (10 levels, 0.1 m cells, 5 m border) | **523k** | **35 s** | **4.3 GB** |
| default with a 20 m border | 1.06M | – | > 6.4 GB: killed |
| without the support-field fix, 245k vertices | 245k | 45 s, of which 35 s in that table | + a 4 GB array |

- The cost scales with the vertex count: (2·border + levels·tile)/cell × (2·border + columns·tile)/cell.
  - Halving `horizontal_scale` quadruples it.
  - `copies 2` doubles the columns.
- The number of envs does not change it: the terrain is one entity shared by all envs.
- RAM is the limit to watch. Your laptop has more cores, so expect a shorter build.

### 4.9 Symptoms → fixes
| Symptom | Likely cause | Fix |
|---|---|---|
| `Curriculum/terrain_level` drops toward 0 in the first few hundred iterations | robots do not walk yet; falls move them down | normal; it should turn around once they walk. If it never does: `fall_moves_down=False` or `down_ratio=0.3` |
| levels barely rise while falls are rare | "up" needs 4 m in one episode; at small command ranges that is rare (4.1.4) | `--set env.terrain.up_distance=3.0`; compare `go2-p31t` |
| `level_flat` ≈ 6 but `level_hill` / `level_pit` stuck low | the slopes are the problem | `Curriculum/slope_*_deg` tells the slope; check `tau_peak` uphill in `--slope` tables; consider more hill/pit share or more iterations. [P3v2] Also the energy term (4.5): train `go2-p31-noE` |
| `level_obstacles` stuck at 4–6 | blind steps near the swing height | expected to some extent; `--clearance 0.09` (higher swing) is the first knob; a height scan belongs to P3.3's teacher |
| `cmd_*_max` stall at the start ranges | `cmd_score_*` < 0.8 | `--set command.curriculum.threshold=0.7`; check tracking on flat tiles first |
| flat tables worse than `$P2` after terrain training | terrain cost flat performance, or training noise | [P3v2] first check against the P2 band (`$P2S2`) and the other P3.1 seeds; if it holds: more flat share, or `--init-from` a P2 policy |
| every robot falls in the first second of `--terrain grid` | a fall criterion too strict for that policy | read `fall_step`; try `--fall_deg 15` |
| `through` n/a | turning command or < 0.1 m/s | expected (4.4.5) |
| top levels fail mostly with `--keep_dr` | μ < 0.47 cannot hold 25° (4.1.3) | read the nominal evaluation for the policy, the DR one for robustness |
| teleop on the terrain counts falls standing still on a face | a flat-trained run's tilt limit | already raised by the grid's steepest tilt; check the printed "tilt termination" |
| build takes minutes, or RAM runs out | grid too fine or border too wide (4.8); or the build fix is not active | check for the `[go2_env] Genesis X: SupportField.activate differs` message (section 10) |

### 4.10 Done when
- `go2-p31` (3 seeds) reaches a `Curriculum/terrain_level` near its ceiling (6.0), with hill and pit near the top.
- Its heat map shows `through` ≥ 80 % up to level 7–9 on hill and pit, and clearly to the right of `$P2`'s.
- Its flat tables are within the P2 band (`$P2`, `$P2S2`; `go2_compare.py --preset gait`).
- The ±20° slope tables track (`vx_meas` near the command, no falls at the standing bin).
- The unseen-tile grid is within a few % of the own tiles.
- `go2_export.py` deploys the reached ranges; the P2 sim-to-sim loop still passes in unitree_mujoco on flat ground.

Results log to fill in (G = `go2_metrics.py`; "frontier" = last level with `through` ≥ 80 %):

| Run | Flags | Seeds | terrain_level (TB) | frontier hill / pit / obst | through hill 9 / pit 9 | flat lin_err 0.5 | flat CoT 0.5 | slope +20 vx_meas 0.5 | slope −20 vx_meas 0.5 | cmd ranges reached |
|---|---|---|---|---|---|---|---|---|---|---|
| $P2, $P2S2 | (P2) | 1, 2 | – | | | | | | | – |
| go2-p31c | --cmd-curriculum | | – | | | | | | | |
| go2-p31t | --terrain | | | | | | | | | – |
| go2-p31 | both | 1, 2, 3 | | | | | | | | |
| go2-p31i | both, --init-from $P2 | | | | | | | | | |

---

## 5. P3.2 Foot slip from ground truth, and the foot-slip penalty decision

Goal, from the plan:
1. The harness measures **each foot's stance slip from simulation ground truth**, correctly on slopes and terrain.
2. With that measurement, decide whether to train with a **generic foot-slip penalty** (Isaac Lab Spot
   `foot_slip_penalty`), and record the decision and its evidence for the config card (P3.4).

### 5.1 What the code does, and why

#### 5.1.1 One slip measure, from the simulator
`go2_terrain.contact_slip_speed(v, ω, n, R)` is the single definition, used by the env, the harness and the foot plot:

```
contact point      p = centre − R·n                       (R = 0.023 m, the foot sphere; n = ground normal)
its velocity       v_cp = v_centre + ω × (−R·n)           (v_centre, ω: foot link, world frame, from Genesis)
slip               |v_cp − (v_cp·n)·n|                    (the part along the ground; the normal part is removed)
```

- A foot that rolls without slipping gives 0: the calf turning over a planted foot is not slip.
- A foot sinking into the soft contact moves along n: also 0.
- n is the normal of the heightfield triangle under the foot (`ground.normal_at()`), i.e. the face Genesis collides
  with. Flat ground: (0, 0, 1). Tilted plane: constant. Grid: per triangle.
- Checked here:
  - Genesis' own contact normal at touchdown agrees with the lookup's within 0.03° on faces up to 25°
    (`go2_terrain.py --check --genesis`).
  - A robot sliding down a 20° plane at friction 0.1 (zero actions): foot slip 2.62 m/s for a base sliding at
    2.60 m/s along the slope. The old horizontal formula read 0.939 of that (cos 20° = 0.940).

What was wrong with the P1 metric on terrain (it is kept, unchanged, for flat recordings):
- it took the horizontal part: × cos θ on a slope, −6 % at 20°, −9 % at 25°;
- its contact point was straight under the centre instead of along −n;
- it counted part of the sinking as horizontal speed on a slope;
- it averaged the four feet.

#### 5.1.2 What the harness reports
`go2_metrics.py` prints two slip tables (stance = |F| > 3 N, first sample of each stance dropped, as in P1):

**"Foot slip"** (all feet together):

| Column | Meaning | Read it as |
|---|---|---|
| `slip_v[cm/s]` | mean contact-point slip in stance | the headline number. On flat ground: exactly the P1 value |
| `slip_d[cm/step]` | slip distance per complete stance | `slip_v` × stance time; n/a without complete stances |
| `slip_ratio[-]` | 1 − distance gained / commanded (P1) | tracking + slip, not slip alone |
| `slip_share[%]` | **new**: share of stance samples above 2 cm/s (`SLIP_SHARE_V`) | events vs creep: a low `slip_v` with a high share = short skids; a high `slip_v` with a low share = a few big slides |
| `pen_xy[cm/s]` | **new**: what the penalty sums in mode `xy`, per step, mean over the window | × W = the reward the penalty takes per second |
| `pen_t[cm/s]` | **new**: the same in mode `tangent` | `pen_xy − pen_t` ≈ the foot's rolling (R × ω of the calf), which `xy` also prices |

**"Foot slip per foot"**: one column per foot, `slip_v / slip_d / slip_share`. Means only; the CSV has the mean and std
of each (`slip_v_FL[cm/s]`, `slip_d_FL[cm/step]`, `slip_share_FL[%]`, ...).

```
--- Foot slip per foot: slip_v [cm/s] / slip_d [cm/step] / slip_share [%] (bin means; std in the CSV) ---
   vx     vy     wz  ood                 FL                 FR                 RL                 RR
 0.50   0.00   0.00        1.20 / 0.31 / 18   1.15 / 0.30 / 17   0.62 / 0.17 / 6    0.66 / 0.18 / 7
```
(an illustration of the layout, not a measurement)

What to look for:
- **Which leg slips.** On a slope I expect the front feet to slip more facing downhill (they brake) and the hind feet
  facing uphill (they push). That is a hypothesis; the per-foot columns test it.
- **Left vs right** (FL vs FR, RL vs RR) on the straight bins: a difference above noise is an asymmetric gait (P2.3).
- **Standing bins** (0, 0, 0) are the noise floor: a standing policy reads ~0.01 cm/s.

#### 5.1.3 The penalty and its two modes
`--foot-slip [W]` adds `foot_slip = −W` (default `FOOT_SLIP_WEIGHT` = 0.1) to the rewards:

```
mode "xy" (default, Isaac Lab Spot foot_slip_penalty):  Σ_feet 1[|F| > 1 N] · |v_centre,xy|
mode "tangent" (--set reward.foot_slip_mode=tangent):  Σ_feet 1[|F| > 1 N] · slip   (= self.foot_slip)
```

- The weight: Spot uses −0.5 next to tracking 5 + 5. The same share of our 1.5 + 0.75 is 0.11, hence 0.1.
- It is a gentle term, as in Spot: two stance feet at 5 cm/s cost 0.1 × 0.1 = 0.01 per second, against 2.25 for perfect
  tracking.
- The contact is this step's, unfiltered, at the reward's threshold (`contact_force_threshold`, 1 N, as Spot).
- It is negative, so the P1 penalty curriculum ramps it: × 0.3 at the start, × 1 after 400 iterations.
- **Why two modes.** The Go2's foot body is the sphere *centre*. In stance it moves at R·ω of the calf even when the
  foot does not slip: about 4.6 cm/s for a calf turning at 2 rad/s. So `xy` prices the leg's rotation over a planted
  foot as well as slip, and on a slope it sees the slip × cos θ. `tangent` prices slip only. Spot's term is `xy`,
  and the plan asked for the generic term, so `xy` is the default. `pen_xy − pen_t` tells you how much of `xy` is
  rolling.

#### 5.1.4 What does not change
- Without `--foot-slip` a run trains exactly as in P3.1 (and P2 on flat ground), bit for bit. Checked: 68 tensors after
  3 iterations, and identical observations, rewards and resets over 60 steps with pushes.
- Flat-ground CSVs: all P1/P2 columns identical (same checksum); P3.2 adds 30 columns.
- Terrain CSVs: `slip_v` and `slip_d` change by design (along the ground now); every other column is as in P3.1.
- So your existing `go2-p31` runs can be the "without" arm of the experiment: re-run `go2_metrics.py` on them with the
  P3.2 code (the new columns), no retraining.

### 5.2 First session: smoke tests (CPU, ~10 min)
```bash
python go2_terrain.py --check                  # last line: ok [P3.2] contact slip: 0 rolling or sinking ...
python go2_terrain.py --check --genesis        # ok [P3.2] contact normal at touchdown ...: max 0.03 deg (1-2 min)
python go2_train.py -e smoke-p32 --cpu -b 32 --max-iterations 3 --terrain --foot-slip
python go2_metrics.py -e smoke-p32 --ckpt 2 --cpu --reps 2 --slope 15 --warmup_s 0.2 --measure_s 0.3
```
In the training log, the P3 block has the P3.2 line:
```
--- terrain, curricula, foot slip, privileged observations (P3) ---
...
P3.2 foot-slip penalty -0.1 x sum over feet in contact (|F| > 1 N) of the planar foot speed (Isaac Lab Spot, mode xy) [m/s]
P3.3 privileged observations off (actor and critic read the policy observation)
```
- TensorBoard: `Episode/rew_foot_slip` appears (≤ 0; about 0 in a 3-iteration run, whose robots barely stand).
- The metrics run: a 3-iteration policy falls on a 15° slope within half a second, hence the short window. The point is
  that both slip tables print. `n/a` in `slip_d` = no complete stance in the window.

### 5.3 Training runs (the A/B)
```bash
# A: without the penalty. Your go2-p31, -s2, -s3 if they used these flags (P3.2 code = P3.1 code, bit for bit)
python go2_train.py -e go2-p32       $BASE            --terrain --cmd-curriculum
python go2_train.py -e go2-p32-s2    $BASE --seed 2   --terrain --cmd-curriculum
python go2_train.py -e go2-p32-s3    $BASE --seed 3   --terrain --cmd-curriculum
# B: + the Spot penalty, W = 0.1
python go2_train.py -e go2-p32-slip    $BASE          --terrain --cmd-curriculum --foot-slip
python go2_train.py -e go2-p32-slip-s2 $BASE --seed 2 --terrain --cmd-curriculum --foot-slip
python go2_train.py -e go2-p32-slip-s3 $BASE --seed 3 --terrain --cmd-curriculum --foot-slip
# only if B is inconclusive (see 5.6):
python go2_train.py -e go2-p32-slipt $BASE --terrain --cmd-curriculum --foot-slip --set reward.foot_slip_mode=tangent
python go2_train.py -e go2-p32-slip5 $BASE --terrain --cmd-curriculum --foot-slip 0.5
```
Same `--max-iterations` for all of them (the A/B compares the same checkpoint).
- [P3v2] 3 seeds per arm. P2 measured training noise far above the rule's 5 % tolerances; with 2 + 2 seeds each cost
  check can still fire by chance 1 time in 6, with 3 + 3 1 time in 20 (5.4.2).
- [P3v2] Both arms carry `$BASE`, so energy is on in both: the foot-slip penalty and the energy term both price foot
  motion, and the record shows `stride` and `swing_h` on every world (5.4.3).

### 5.4 Evaluation and the decision

#### 5.4.1 Commands
```bash
for R in go2-p32 go2-p32-s2 go2-p32-s3 go2-p32-slip go2-p32-slip-s2 go2-p32-slip-s3; do
    python go2_metrics.py -e $R --ckpt $CKPT                         # flat
    python go2_metrics.py -e $R --ckpt $CKPT --terrain grid          # own tiles
    for S in 15 -15 25 -25; do python go2_metrics.py -e $R --ckpt $CKPT --slope $S; done
done
python go2_compare.py logs/go2-p32/metrics_ckpt${CKPT}_seed0_slope+25.csv \
    logs/go2-p32-slip/metrics_ckpt${CKPT}_seed0_slope+25.csv --preset slip
python go2_compare.py logs/go2-p32/metrics_ckpt${CKPT}_seed0_grid.csv logs/go2-p32-slip/metrics_ckpt${CKPT}_seed0_grid.csv \
    --preset slip --rows "hill 6" "pit 6" "rough 9"
python go2_decide.py slip --off go2-p32 go2-p32-s2 go2-p32-s3 --on go2-p32-slip go2-p32-slip-s2 go2-p32-slip-s3 \
    --ckpt $CKPT
```

#### 5.4.2 The rule (agree on it before looking at the numbers)
`go2_decide.py slip` reduces every CSV to the mean of its moving bins (command ≥ 0.1 m/s or 0.1 rad/s; per type on the
grid) and applies, with thresholds as constants in its header:

| Criterion | Pass when | Constant |
|---|---|---|
| slip drops (the benefit) | on every terrain world (grid, each slope; flat if none), `slip_v` of B is ≥ 30 % below A, and every B seed is below every A seed | `SLIP_DROP` |
| no foot worse (a cost) | no foot's `slip_v` is worse by more than 10 % with every B seed above every A seed | `FOOT_WORSE` |
| tracking (a cost) | not: flat `lin_err` of B > A × 1.05 + 0.005 m/s with every B seed above every A seed | `LIN_ERR_TOL` |
| getting through (a cost) | not: `through[%]` lower by more than 5 points with every B seed below every A seed, on any grid type or slope | `THROUGH_TOL` |
| energy (a cost) | not: flat CoT of B > A × 1.05 with every B seed above every A seed | `COT_TOL` |

All pass → **KEEP** (train v1.0 with the penalty), else **DROP**. If you change a threshold, do it before the runs end
and say so with `--note "..."`; the record keeps the note.

[P3v2] Why a cost needs both conditions: the tolerances (5 %) are far below the training noise P2 measured (3.4), so a
mean beyond the tolerance alone is often noise, and with 2 seeds per arm the old rule would often DROP for noise.
"Every B seed worse than every A seed" is a rank condition: with no real effect it happens by chance 1 / C(n_A + n_B,
n_B), i.e. 1 in 6 with 2 + 2 seeds and 1 in 20 with 3 + 3. The benefit already asked for the same separation. The
price: a small real cost that stays inside the seed spread passes; that is what the seeds can resolve, and the record
shows every seed, so read it.

#### 5.4.3 Reading the output
```
=== P3.2 foot-slip penalty: --off go2-p32 go2-p32-s2 go2-p32-s3 | --on go2-p32-slip ... | ckpt 1999 ===
worlds (moving bins): flat, grid, slope+15, slope-15, slope+25, slope-25
3 + 3 seeds: with no real effect, the arms separate in a given direction by chance 1 in 20 (a cost needs that AND its
tolerance)
world     metric               --off (per seed)  --on (per seed)  change
slope+25  slip_v[cm/s]         ...               ...              -41 %
slope+25  slip_v_FL[cm/s]      ...
...
  [pass] slip drops on slope+25: 41 % (needs 30 %), every --on seed below every --off seed: True
  [pass] tracking: lin_err on flat 0.0817 vs 0.0700 m/s; beyond 0.0785: True, every --on seed worse: False
  [FAIL] getting through: worst through change -8.0 points (grid pit, every --on seed lower: True), tolerance -5;
         failing: grid pit -8.0
decision: DROP (a criterion fails)
  --on runs: foot_slip weight -0.1, mode xy, contact threshold 1.0 N
decision record written to decisions/slip.md (commit it; the P3.4 config card includes it)
```
(the layout; the numbers are made up)

- The evidence rows also show `slip_share`, `pen_xy`, `pen_t` per world, and the gait as side effects that are not
  judged: `stride`, `swing_h`, `step_freq` on flat, [P3v2] `stride` and `swing_h` on every terrain world.
- [P3v2] "beyond ...: True, every --on seed worse: False" (the tracking line above) is a mean beyond the tolerance
  inside the training noise: it passes. Read such lines in the evidence rows anyway, per seed.
- [P3v2] With energy on in both arms, compare `swing_h` and `stride` with P2's band (6.9–8.7 cm swing at 0.5 m/s): a
  clearly lower swing with the penalty is the interaction P2 warned about, even when every criterion passes.
- `decisions/slip.md` holds the runs, the rule, every number, the decision and the flags it implies. Commit it:
  `logs/` is git-ignored, `decisions/` is not. P3.4's `go2_card.py` puts it on the config card.

### 5.5 What I expect (hypotheses, not measurements)
- With the [S2R] contact model (elliptic cone, impratio 100) stance feet barely creep on flat ground, so the penalty
  should matter little there. On 20–25° faces and at low friction the feet do slip, so most of any effect should
  show on the slopes and the hill/pit tiles.
- At W = 0.1 the term is small next to tracking. A drop of slip by ≥ 30 % from it alone would surprise me; the more
  likely outcomes are "no clear effect" (DROP, and v1.0 stays simpler) or a modest drop with a gait change.
- `xy` will also push the policy to rotate its calves less in stance (`pen_xy − pen_t`): watch `stride` and `step_freq`.

### 5.6 Knobs

| Knob | Where | Default | Effect |
|---|---|---|---|
| penalty on/off, weight | `--foot-slip [W]` | off; 0.1 when given alone | the A/B switch; 0.1 ≈ Spot's share of the tracking weight, 0.5 ≈ 4.5× |
| mode | `--set reward.foot_slip_mode=tangent` | `xy` | slip only, rolling not priced |
| contact threshold | `--set reward.contact_force_threshold=...` | 1.0 N | **also** the air-time contact (P1.1) and the P3.3 slip entry: change it for all or for none |
| harness slipping threshold | `SLIP_SHARE_V` (`go2_metrics.py`) | 0.02 m/s | what `slip_share` counts |
| harness stance threshold | `CONTACT_N` (`go2_metrics.py`) | 3 N | as in P1; changing it changes every slip and gait column |
| decision thresholds | `go2_decide.py` header | 30 %, 10 %, 5 %, 5 points, 5 % | fix them before the numbers; [P3v2] a cost also needs the seed ranges separated |
| seeds per arm | the runs you pass | 3 | [P3v2] chance separation 1 in 20 (2 + 2: 1 in 6) |

If B is inconclusive:
- **Slip did not drop, nothing else moved**: the term is too weak. One run at W = 0.5 (`go2-p32-slip5`) tells whether a
  stronger penalty does anything before you conclude.
- **Slip dropped but the gait changed** (shorter strides, higher step frequency, lower `through` on pits): try
  `tangent` (`go2-p32-slipt`). It does not price the calf's rotation.
- **Slip dropped on slopes only, the rule fails on the grid**: read the grid per type. On rough and obstacle tiles,
  slip is dominated by feet landing on edges; a penalty cannot fix the blind policy's missing terrain knowledge.

### 5.7 Symptoms → fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| `slip_v` on a slope differs from your P3.1 CSV of the same checkpoint | by design: along the ground now | compare P3.2 CSVs with P3.2 CSVs only |
| flat `slip_v` differs from P3.1 | not expected | check that both CSVs come from the same `--reps`, `--seed`, `--init_noise` (the checksum line must match) |
| `go2_decide.py`: `no go2_metrics.py CSV common to every run` | a world is missing for some run, or `--ckpt` differs | run the 5.4.1 loop for every run; tagged CSVs (`--tag`) are ignored on purpose |
| `n/a` in `slip_v_*` per foot | that foot had no stance sample (3 N) in the window | a lame leg, or a fall during the warm-up (`falls` column) |
| no `Episode/rew_foot_slip` in TensorBoard | the run has no `--foot-slip` | check the P3.2 line of the start-up block |
| `reward.foot_slip_mode must be 'xy' or 'tangent'` | a typo in `--set` | |

### 5.8 Done when
- The harness reports per-foot tangential slip; flat values are unchanged (checked here).
- The A/B decision is taken by `go2_decide.py slip` on 3 seeds per arm, from the same checkpoint.
- `decisions/slip.md` is committed: on/off, weight, mode, contact threshold, and the numbers behind it.

---

## 6. P3.3 Privileged observations for the critic and a teacher

Goal, from the plan: a privileged observation vector that the **critic** reads (asymmetric actor-critic), and that a
**teacher** policy can read (for a later teacher–student distillation). The deployed actor stays blind.

### 6.1 What the code does, and why

#### 6.1.1 Two observation groups
- With `obs_cfg["privileged"]` set, `Go2Env.get_observations()` returns `TensorDict({"policy": [N, 45],
  "privileged": [N, P]})`. Without it, `"policy"` only, as in every run before P3.3.
- rsl_rl 5.5.1 picks the groups of each network with `train_cfg["obs_groups"]`. `MLPModel` concatenates the listed groups
  in order (`policy` first, so the first 45 inputs keep the P1 layout).
- `go2_train.py --privileged critic` sets `{"actor": ["policy"], "critic": ["policy", "privileged"]}`;
  `--privileged teacher` gives the actor `["policy", "privileged"]` as well.
- legged_gym has the same hook (`num_privileged_obs`: "if not None a priviledge_obs_buf will be returned by step()").

Why a privileged critic:
- PPO's update is driven by the advantages, which come from the critic's value estimates.
- Under randomization and terrain, the same proprioceptive observation can lead to very different returns: a slippery
  env, a heavy base, a weak motor, a pit ahead. A critic that sees friction, mass, CoM, motor strength, the true
  velocity and the contacts can tell them apart, so its estimates (and the gradient) are less noisy.
- The critic is not deployed: nothing changes for the actor, the export or the robot. This is the asymmetric
  actor-critic of Pinto et al. 2017.

Why a teacher:
- Lee et al. 2020 train a privileged teacher, then distill it into a proprioceptive student. The teacher learns
  faster because it sees what the student must infer.
- P3.3 makes the teacher trainable. The distillation (rsl_rl 5.5.1 `Distillation`, observation sets `"student"` /
  `"teacher"`) is a later step.
- A teacher cannot be deployed: `go2_export.py` refuses it.

#### 6.1.2 The entries
`go2_env.PRIVILEGED_SIZES`; `--privileged-entries` picks and orders them (default `PRIVILEGED_DEFAULT`, 36 values: the
P3.3 list).

| Entry | Size | Scaling (fixed constants) | Source in the env | Mirror (`--symmetry`) |
|---|---|---|---|---|
| `base_lin_vel` | 3 | body frame × 2.0 | `base_lin_vel` (the robot only estimates it) | (vx, −vy, vz) |
| `friction` | 1 | (μ − 0.8) / 0.4 | P1.2 `friction` | same |
| `added_mass` | 1 | (m − 0.5 kg) / 1.5 | P1.2 `added_mass` | same |
| `com_shift` | 3 | body frame / 0.03 m | P1.2 `com_shift` | (x, −y, z) |
| `motor_strength` | 12 | (k − 1) / 0.1 | P1.2 `motor_strength`, policy joint order | legs swap |
| `foot_forces` | 12 | body frame / 100 N, clipped ±3 | net contact force per foot | feet swap, y flips |
| `foot_slip` | 4 | / 0.2 m/s, clipped at 5 | P3.2 `foot_slip` | feet swap |
| `gains` (opt.) | 2 | (factor − 1) / 0.15 | P1.2 Kp, Kd | same |
| `latency` (opt.) | 1 | 0–1 policy step | P1.2 `latency_steps` | same |
| `foot_clearance` (opt.) | 4 | / 0.1 m, clipped −1…3 | foot-sphere bottom above the ground under it | feet swap |
| `height_scan` (opt.) | 187 | clip(base z − 0.3 − h, −1, 1) × 5 | ground at legged_gym's 17 × 11 points around the base | y index flips |

Notes:
- All noise-free. The policy group keeps its noise; the privileged part is computed after the noise draw, so the
  random stream (and a `--privileged` run's policy observations) is the same as without it.
- Fixed scales, not the run's randomization ranges, so a value means the same in every run (wider ranges simply
  give values beyond ±1). rsl_rl's `obs_normalization` is off by default.
- Vectors are in the **body frame**: the left–right mirror is then a fixed permutation and sign. A world-frame force
  would need the heading to be mirrored.
- At a reset the robot respawns in the air at rest: the velocity, foot forces and slip read 0. The randomized
  parameters are those of the new episode.
- Height scan: legged_gym's grid (x −0.8…0.8 m, y −0.5…0.5 m every 0.1 m, rotated by the base yaw only) and formula,
  with the Go2's standing height 0.3 m where legged_gym subtracts 0.5 m (ANYmal). On flat ground at 0.3 m every point
  reads 0. It is the one entry that tells a teacher what the terrain ahead looks like; without it a teacher is "blind
  but omniscient about itself".
- `foot_clearance` reads the foot positions again at observation time, so after a reset with `reset_forward` (terrain)
  it is the respawned feet. With the flat-ground default (`skip_forward`) the first value of an episode is the last
  step's.

#### 6.1.3 Symmetry
- rsl_rl's symmetry extension augments the whole observation TensorDict, critic inputs included.
- So `go2_symmetry.compute_symmetric_states()` now mirrors every group, entry by entry, with the rules in the table.
- An entry without a rule is an error, not a silent wrong mirror.
- `python go2_symmetry.py --privileged [--slope 10]` checks every rule in a mirrored rollout. Env 0 gets a random CoM
  shift and random motor strengths, env 1 their mirror. The physics stays mirrored only if those two rules are right,
  and every block of both groups is compared.

#### 6.1.4 What does not change
- Without `--privileged`, nothing (no group, same weights bit for bit).
- With `--privileged critic`:
  - the actor (45 inputs), the export, `deploy_cfg.json` (one record line added) and the robot side are unchanged;
  - `go2_metrics.py`, `go2_teleop.py`, `go2_eval.py` work as before: they pass the env's whole observation and the
    policy picks its group.

### 6.2 First session: smoke tests (CPU, ~10 min)
```bash
python go2_symmetry.py --privileged                  # every block ~1e-3 or below (flat)
python go2_symmetry.py --privileged --slope 10       # every block ~1e-2 or below (slope)
python go2_train.py -e smoke-p33 --cpu -b 32 --max-iterations 3 --terrain --privileged critic --symmetry
python go2_train.py -e smoke-p33t --cpu -b 32 --max-iterations 3 --privileged teacher \
    --privileged-entries base_lin_vel friction foot_forces foot_slip height_scan
python go2_export.py -e smoke-p33 --ckpt 2           # exports; deploy_cfg["source"]["critic_privileged"]
python go2_export.py -e smoke-p33t --ckpt 2          # refused: the actor reads ['policy', 'privileged']
python go2_metrics.py -e smoke-p33t --ckpt 2 --cpu --reps 1    # a teacher in the harness (sym_err included)
```
What to read:
- The start-up block: `P3.3 privileged group (36 values: base_lin_vel 3, friction 1, ...) read by critic`; for a
  teacher, `read by actor and critic  -> TEACHER: the actor cannot be deployed`.
- The model print-out: `Actor Model: ... Linear(in_features=45, ...)` and `Critic Model: ... Linear(in_features=81,
  ...)` for the default entries (45 + 36). A teacher with the height scan: 45 + 3 + 1 + 12 + 4 + 187 = 252 for both.
- The mirror check: the columns are the largest difference and, in brackets, the largest value of each block. On
  the slope, `foot_forces` and `foot_slip` reach ~5e-3: contacts can differ by one solver iteration between the two
  envs. A wrong rule gives a difference near the size of the signal (checked by breaking rules on purpose: 0.1 to
  1.6).

### 6.3 Training runs
```bash
# the critic A/B: same flags as the P3.2 arm you keep (A = go2-p32 or go2-p32-slip, 3 seeds), + --privileged critic
P32="--terrain --cmd-curriculum"            # + --foot-slip if the P3.2 decision was KEEP
python go2_train.py -e go2-p33    $BASE          $P32 --privileged critic
python go2_train.py -e go2-p33-s2 $BASE --seed 2 $P32 --privileged critic
python go2_train.py -e go2-p33-s3 $BASE --seed 3 $P32 --privileged critic
# optional: the critic with the height scan too (it sees the terrain under and ahead of the robot)
python go2_train.py -e go2-p33h $BASE $P32 --privileged critic --privileged-entries base_lin_vel friction added_mass \
    com_shift motor_strength foot_forces foot_slip height_scan
# optional: a teacher, for a later distillation (not deployable)
python go2_train.py -e go2-p33t $BASE $P32 --privileged teacher --privileged-entries base_lin_vel friction \
    added_mass com_shift motor_strength foot_forces foot_slip height_scan
```
- `--symmetry` combines with both (the privileged group is mirrored).
- `--lcp` / `--caps-*` combine with both. For a teacher, the CAPS noise on the privileged part is 0.
- `--rnn` combines with both: the RNN reads the concatenated groups (checked: critic LSTM(81, 256)).
- GPU memory: the rollout storage holds every group. Default entries: 4096 × 24 × 36 floats = 14 MB; with the height
  scan, 88 MB. Negligible next to the simulator.

### 6.4 Evaluation and the decision

#### 6.4.1 TensorBoard (the main evidence for a critic)
| Tag | Privileged critic vs A | Why |
|---|---|---|
| `Train/mean_reward` | rises earlier and/or ends higher | the point of the change |
| `Loss/value` | lower | the critic sees more; **not** a better policy by itself |
| `Curriculum/terrain_level`, `level_<type>` | higher or faster | the terrain curriculum is the best "how much did it learn" curve on terrain |
| `Curriculum/cmd_*_max` | at least as fast | |
| `Policy/mean_std` | similar | a much faster collapse can mean overconfident advantages |
| `Loss/surrogate`, `Loss/learning_rate` | similar | the adaptive learning rate follows the KL; a jumpy KL means unstable advantages |

#### 6.4.2 Commands and rule
```bash
for R in go2-p33 go2-p33-s2 go2-p33-s3; do
    python go2_metrics.py -e $R --ckpt $CKPT
    python go2_metrics.py -e $R --ckpt $CKPT --terrain grid
    for S in 15 -15 25 -25; do python go2_metrics.py -e $R --ckpt $CKPT --slope $S; done
done
python go2_decide.py critic --off go2-p32 go2-p32-s2 go2-p32-s3 --on go2-p33 go2-p33-s2 go2-p33-s3 --ckpt $CKPT
```
`go2_decide.py critic` reads the TensorBoard files of the six runs and their harness CSVs. Each criterion fails only
when B is beyond the tolerance **and** every B seed is worse than every A seed ([P3v2], 5.4.2):

| Criterion | Fails when | Constant |
|---|---|---|
| learns at least as fast | the iteration at which `Train/mean_reward` first reaches 90 % of A's final value: B's mean over seeds more than 5 % later, every B seed later than every A seed | `LEARN_REACH`, `LEARN_TOL` |
| ends at least as high | B's `Train/mean_reward` over the last 5 % of the iterations below A − 2 %, every B seed below every A seed | `FINAL_TOL` |
| terrain level (when logged) | B's final `Curriculum/terrain_level` below A − 0.2, every B seed below every A seed | `LEVEL_TOL` |
| tracking, getting through, energy | as for the slip rule (5.4.2) | |

No criterion fails → **KEEP**. `Loss/value` is printed, not judged. [P3v2] So are the commands P2 left weak: flat
`lin_err` over the slow and lateral bins (planar command ≤ 0.3 m/s) and `yaw_err` turning in place. P2 measured 0.16
for 0.25 m/s, 0.22 for 0.3 m/s lateral and a constant −0.05 m/s shortfall forward; a critic that sees the base velocity
is one candidate fix, and these rows say whether it is one. It writes `decisions/critic.md`.

#### 6.4.3 A teacher
- The harness measures a teacher like any policy (`go2_metrics.py` builds the privileged group).
- It is an upper bound for what a student could reach. For example: the grid frontier of `go2-p33t` vs `go2-p33`
  tells how much the height scan is worth on each tile type.
- No export. `go2_export.py` stops with "distill a student that reads 'policy' only".

### 6.5 What I expect (hypotheses, not measurements)
- A privileged critic should help most where returns depend on hidden state: low-friction envs on slopes, heavy or
  weak robots, and the terrain tiles (the critic "knows" a fall is coming). So I would look for the gain in
  `Curriculum/terrain_level` (faster climb) rather than in the flat tracking reward.
- With only the default entries the critic does not see the terrain. `go2-p33h` (height scan in the critic) is the
  variant that can predict "a pit ahead".
- If the curves are within seed noise, the rule keeps it: it costs nothing on the robot. The rows of P2's weak
  commands (6.4.2) are where I would look for a gain on flat ground.

### 6.6 Knobs

| Knob | Where | Default | Notes |
|---|---|---|---|
| who reads the group | `--privileged critic` / `teacher` | off | `--set train.obs_groups.critic=[...]` works too |
| which entries, in which order | `--privileged-entries ...` | `PRIVILEGED_DEFAULT` (7 entries, 36 values) | order = vector layout; changing it changes the critic's input layout (start a new run) |
| entry scales | `go2_env._privileged_obs()` | see 6.1.2 | fixed constants; change both the code and its comment |
| height scan grid | `HEIGHT_SCAN_X`, `HEIGHT_SCAN_Y`, `HEIGHT_SCAN_OFFSET` (`go2_env.py`) | 17 × 11 points, 0.1 m, offset 0.3 m | the size follows; the mirror reads `env.height_scan_shape` |
| a new entry | `PRIVILEGED_SIZES` + `_privileged_obs()` + `go2_symmetry._privileged_mirror()` | | the mirror check fails until the rule exists |

### 6.7 Symptoms → fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| `obs_cfg['privileged']: unknown entries` | a typo in `--privileged-entries` | the message lists the known names |
| `obs_groups ['critic'] read 'privileged' but obs.privileged lists no entry` | `--set train.obs_groups...` without entries | use `--privileged critic` |
| `go2_symmetry: no mirror rule for the privileged entry 'x'` | a new entry without its rule | add it to `_privileged_mirror()` |
| `go2_symmetry: privileged observation of size ...` | `PRIVILEGED_SIZES` and the code disagree | sizes and rules must match |
| a critic run cannot be loaded by a script (size mismatch in the critic) | the script was given another `obs_cfg` | load with the run's own `cfgs.pkl` (all scripts do) |
| `go2_export.py`: "the actor reads ... not only 'policy'" | a teacher run | by design; distill a student first |
| `Loss/value` much lower but the reward not better | the critic fits better, the policy does not benefit | the decision rule judges the reward and the harness, not the value loss |

### 6.8 Done when
- `--privileged critic` and `teacher` train; the mirror check passes for every entry (flat and slope); the export of a
  critic run is unchanged and a teacher's is refused (all checked here).
- The critic A/B is decided by `go2_decide.py critic` on 3 seeds per arm, and `decisions/critic.md` is committed;
  v1.0 follows it (the config card checks).

---

## 7. P3.4 Baseline v1.0: final run, config card, checked export, harness report, git tag

Goal, from the plan: freeze one policy as **Baseline v1.0**, the reference for everything after P3:
- the final run, with the P3.1–P3.3 choices;
- a one-page config card;
- an export checked against the training model (TorchScript and ONNX);
- the harness report;
- a git tag.

P3.4 adds tools, not training features: a run with the same flags trains as before, bit for bit (checked, see
`CHANGES.md`).

### 7.1 What the code does, and why

#### 7.1.1 Provenance: what trained the policy (`go2_provenance.py`, `manifest.json`)
`go2_train.py` writes `logs/<exp>/manifest.json` when a run starts. P3.4 completes it:
- git commit, dirty flag (tracked files changed), untracked `.py` files (code a checkout of the commit would lack);
- the command line (`argv`) and every argument, host, GPU, package versions, start time;
- a **code fingerprint**: the sha256 of every `.py` file of the code folder.

Changes in behaviour:
- Until P3.3 the manifest was skipped entirely when a `git` call failed. Now the git fields are just `null`.
- A dirty tree leaves `logs/<exp>/git_diff.patch` (what a checkout of the commit would lack) and a warning at the
  start of training. Untracked `.py` files get a warning too.
- `--set` can now be given several times; the lists add up. Until P3.3 a second `--set` silently replaced the first,
  which mattered as soon as the decisions add their own `--set` (e.g. `reward.foot_slip_mode=tangent`) to
  `--set env.reset_forward=True`.

Why the fingerprint: git says which commit a run started from, not whether a file was edited after training.
`go2_card.py` compares the fingerprint with the files as they are now; `python go2_provenance.py logs/<exp>` does the
same by hand.

The `git/` folder in every run is rsl_rl's: it stores a diff only for repositories registered with
`runner.add_git_repo_to_log()`. By default that is rsl_rl itself, installed by pip outside any repository, so the folder
stays empty and the log says `Could not find git repository in .../rsl_rl/__init__.py. Skipping.` (harmless). The
manifest and `git_diff.patch` record your code instead.

#### 7.1.2 The checked export (`go2_export.py --onnx`)
- `policy.onnx`: rsl_rl 5.5.1 `export_policy_to_onnx()`, opset 18.
  - MLP: input `obs` [1, D], output `actions` [1, 12].
  - Recurrent: also `h_in` [layers, 1, hidden] (+ `c_in` for an LSTM) → `h_out` (`c_out`). The caller carries them
    from call to call and zeroes them at a reset, as `reset()` does for the TorchScript export.
  - **One file.** With torch 2.15 (dynamo exporter, the default since 2.9), `torch.onnx.export` writes the weights to
    a separate `policy.onnx.data` (`external_data=True` is its default, and rsl_rl keeps it). `go2_export.py` folds
    them back into `policy.onnx`, so the file is self-contained and its sha256 covers the weights.
- Both files are checked against the training policy (deterministic mean), one observation at a time as on the robot,
  on two input sets:
  - 100 random observations, from a reset (the P1.6 check);
  - **the observations the policy itself meets**: a rollout of the training policy in the nominal world, 8 envs × 200
    steps (`--check-envs`, `--check-steps`), commands resampled as in training. Each env's sequence is replayed
    through the export, with a reset wherever the env was reset.
- Any difference above `EXPORT_TOL` (1e-4) stops the script. Measured here (MLP, LSTM, GRU, several exports): every
  error below 3e-7, i.e. float32 round-off.
- `exported/export_check.json`: every error, the sha256 of `policy.pt`, `policy.onnx` and `deploy_cfg.json`, the
  ONNX inputs and outputs, the run's git commit. `deploy_cfg.json` → `source` gets the hashes, the errors, the commit
  and the export time.
- An export without `--onnx` deletes a stale `policy.onnx` (and `.data`), so the folder never holds two networks.
- Why the rollout check: random observations miss the states the policy really meets, and a recurrent export's state
  handling (carry, reset) only shows on sequences.
- Why ONNX: runtimes without PyTorch (onnxruntime in C++, TensorRT); and torch 2.15 already warns that
  `torch.jit.load` is deprecated. The robot side (`go2_fsm.py`) still runs `policy.pt`, which loads fine.

#### 7.1.3 The harness report (`go2_report.py`)
One command runs the standard evaluation set and collects one summary in `logs/<exp>/report/`.

| World | `go2_metrics.py` | Headline numbers |
|---|---|---|
| `flat` | nominal world | per key command (0 0 0, 0.5 0 0, 1 0 0, −0.5 0 0, 0 0.3 0, 0 0 0.5): falls, lin_err, yaw_err, vx_meas, CoT, slip_v |
| `flat_dr` | `--keep_dr --tag dr` | the same, under the training randomization |
| `grid` | `--terrain grid` | per tile type: the **frontier**, through at the top level, falls; mean through |
| `grid_unseen` | `--terrain grid --terrain_set seed=1 --tag unseen` | the same, on new random tiles |
| `slope±10`, `±20`, `±25` | `--slope D` | at 0.5 m/s: vx_meas, lin_err, through, falls, CoT, slip_v; standing: falls, creep |
| gate (optional) | `go2_sim_suite.py --suite gate`, with `--deploy-python` | runs as expected (needs unitree_mujoco and an export) |

- **Frontier** = the last level of the run of levels, from the lowest, with `through` ≥ 80 % (`FRONTIER_THROUGH`).
  A type that fails its lowest level has no frontier ("none" in the card, −1 in the console and in the headline).
- Each `go2_metrics.py` runs as a subprocess with its own Genesis; its output goes to `report/logs/<world>.log`.
- Outputs: `REPORT.md` (tables), `summary.json` (everything, plus the headline numbers; strict JSON, n/a is `null`),
  `plots/` (heat maps, gait and spectra plots).
- `--ref EXP` adds a table "Against EXP": every headline number of both runs, the change and which way is better. The
  same command then measures every later candidate against v1.0.
- `--collect` rebuilds the report from existing CSVs; `--skip-existing` evaluates only the missing worlds. The envs
  per bin are read from the CSVs (the denominator of `falls`), so a collected report states what was measured.
- [P3v2] **Seed band** (`--seeds EXP-s2 EXP-s3`): the other training seeds of the configuration, whose reports exist.
  `REPORT.md` gets a table of every headline number per run, with the mean and the range (max − min), and
  `summary.json` a `"band"`. Their configurations are checked against this run's: `cfgs.pkl` and the manifest's
  arguments, except names and seed; any difference is printed as a `WARNING` and recorded.
- [P3v2] **Verdicts.** With a band on either side, `--ref` gives each headline number a verdict: better / worse only
  when the two sides' seed ranges do not overlap, else "within noise". One run on each side gets no verdict, and
  the report says why (P2: one candidate against one baseline run is not a result).
- [P3v2] **The gate suite runs in the unitree_mujoco environment**: `--deploy-python` (default `$DEPLOY_PY`) starts
  `go2_sim_suite.py`. Before the evaluation starts, the report checks the export, the unitree_mujoco folder and that
  this python imports `unitree_sdk2py`, so a wrong environment fails at once instead of after the evaluation.
  `go2_sim_suite.py` refuses a second battery while one runs (P2: two in parallel make the results invalid).
- Time: 32 to 49 s per world on 2 CPU cores here, with `--reps 2` and tile subsets. On your GPU with the default
  `--reps 20` I would expect 10–20 min for the whole set (not measured).

#### 7.1.4 The config card (`go2_card.py`)
`logs/<exp>/CONFIG_CARD.md`, one page (about 60 lines). It reads only the run's files: `cfgs.pkl`, `manifest.json`,
the checkpoint, `exported/`, `report/summary.json` and `decisions/`. Sections:
1. **Freeze checklist**, each line `ok` or `NO` with the reason:

   | Line | ok when |
   |---|---|
   | code committed | the manifest has a commit; at training time the tree was clean and held no untracked `.py` file |
   | code unchanged | every `.py` file still has the fingerprint the run recorded |
   | export checked | `exported/` is from this checkpoint, passed every check, has `policy.onnx`, and its files still have the recorded sha256 |
   | report complete | `report/summary.json` is from this checkpoint, has all 10 worlds, every tile type and level the run trained on, and ≥ 20 envs per bin (`REPORT_MIN_REPS`) |
   | seed band | [P3v2] the report holds a seed band of ≥ 3 training seeds (`BAND_MIN_RUNS`), all of one configuration |
   | decision slip / critic | the record exists and the run follows it (penalty weight and mode; privileged entries) |
   | reset_forward | `env.reset_forward` is True |

2. **Identity and training**: commit, host, GPU, versions, the exact command, PPO settings, networks and parameter
   counts.
3. **World**: physics, the [S2R] contact model, the randomization, the terrain and the mean level the curriculum
   reached per type, the command ranges (reached, deployed).
4. **Policy interface and rewards**: observation layout and noise, action, every reward weight and its parameters
   (the source of each term is in `CHANGES.md`, "Rewards and terminations per phase").
5. **Results**: the headline numbers of the report (a slope table included); [P3v2] with a seed band, the mean
   [min–max] of the key numbers.
6. **Export**: sha256 and errors of each file, ONNX inputs and outputs.
7. **Decisions**: P3.2, P3.3 (decision, date, runs, flags, criteria), `reset_forward`.
8. **Reproduce**: `git checkout <commit>` and the command line.

`--freeze DIR` copies the card and the small records into `DIR` (git-tracked): `deploy_cfg.json`,
`export_check.json`, `REPORT.md`, `summary.json`, the decisions, `manifest.json`, `cfgs.pkl` and `cfgs.json`. It then
prints the git and gh commands. The binaries (policy files, checkpoint, `terrain.npz`) go to the GitHub release,
because `logs/` and `*.pt` are git-ignored. `--strict` makes any `NO` an error (exit 1), and then `--freeze` copies
nothing: a baseline is frozen clean, or not at all.

#### 7.1.5 The decisions feed the run (`go2_decide.py flags`)
`python go2_decide.py flags` prints the `go2_train.py` flags of the KEEP decisions in `decisions/` (nothing for a
DROP). On stderr it says what each record decided, and warns when a record is missing. So the v1.0 command line is
built from the records, and the card checks that the run followed them.

### 7.2 First session: a dry run of the freeze (CPU, ~15 min)
Run the whole chain once on a tiny run, before spending GPU hours. Nothing here is committed.
```bash
python go2_provenance.py                    # git state and versions as the tools see them
python go2_train.py -e smoke-v1 --cpu -b 32 --max-iterations 3 --terrain --cmd-curriculum \
    --set env.reset_forward=True $(python go2_decide.py flags)
python go2_export.py -e smoke-v1 --ckpt 2 --onnx --check-envs 2 --check-steps 50
python test_go2_deploy.py --policy logs/smoke-v1/exported
python go2_report.py -e smoke-v1 --ckpt 2 --cpu --reps 2 --grid-types hill pit rough --grid-levels 0 9
python go2_card.py -e smoke-v1 --ckpt 2 --name "Smoke v1" --freeze /tmp/smoke-v1-freeze
python go2_provenance.py logs/smoke-v1      # "code unchanged since training"
```
Check:
- The start of training prints both `--set` lines, and a `WARNING` if your tree is dirty or has untracked `.py` files.
- The export prints four errors around 1e-7; `logs/smoke-v1/exported/` holds `policy.pt`, `policy.onnx` (one file,
  no `.data`), `deploy_cfg.json`, `export_check.json`.
- The report runs 10 worlds and prints the headline numbers (a 3-iteration policy does not walk: falls, n/a and
  "none" frontiers are expected).
- On the card, expect: `export checked` ok, `code unchanged` ok, `reset_forward` ok; `report complete` **NO**
  (2 envs per bin, tile subsets: by design, a quick report cannot pass for a baseline's); `seed band` NO (one run);
  `code committed` NO if your tree was dirty; decisions NO until `decisions/` holds both records.
- `/tmp/smoke-v1-freeze/` holds the copies; the console shows the git and gh commands it would have you run.

### 7.3 The freeze, step by step
```bash
# 0) P3.2 and P3.3 decided: decisions/slip.md and decisions/critic.md exist (sections 5.4, 6.4)
python go2_decide.py flags                  # e.g. "--foot-slip 0.1" or nothing
# 1) Commit the code that will train v1.0 (the manifest records the commit; a dirty tree is flagged on the card)
git add -A && git commit -m "Code of Baseline v1.0"
# 2) The final run, + seeds 2 and 3 for the thesis
V1FLAGS="--terrain --cmd-curriculum --set env.reset_forward=True $(python go2_decide.py flags)"
python go2_train.py -e go2-v1.0    $BASE          $V1FLAGS
python go2_train.py -e go2-v1.0-s2 $BASE --seed 2 $V1FLAGS
python go2_train.py -e go2-v1.0-s3 $BASE --seed 3 $V1FLAGS
# 3) Export, checked (TorchScript + ONNX), and the deploy-side tests on it
python go2_export.py -e go2-v1.0 --ckpt $CKPT --onnx
python test_go2_deploy.py --policy logs/go2-v1.0/exported
# 4) The harness reports: the other seeds first, then v1.0 with its seed band and the gate suite (unitree_mujoco env)
for S in s2 s3; do python go2_report.py -e go2-v1.0-$S --ckpt $CKPT; done
python go2_report.py -e go2-v1.0 --ckpt $CKPT --seeds go2-v1.0-s2 go2-v1.0-s3 \
    --gate ~/unitree_mujoco --deploy-python $DEPLOY_PY
# 5) The card, frozen into baselines/v1.0/; then run the git / gh commands it prints (7.5)
python go2_card.py -e go2-v1.0 --ckpt $CKPT --name "Baseline v1.0" --freeze baselines/v1.0 --strict
```
Notes:
- `$BASE` already holds `--seed 1`; a second `--seed` later on the line wins.
- `reset_forward=True` removes the reset kick on flat tiles too (4.1.8). The freeze is the moment to take it, since
  it ends bit-identity with P2 anyway.
- Train from a clean tree: commit first, then do not edit any `.py` until the card is written. The card checks both.
- [P3v2] The seed band of step 4 is v1.0's training noise: every later candidate is judged against it (7.6). Write
  the mean and the range in the thesis next to v1.0's numbers (results log in 7.9).
- The MuJoCo gate runs alone: do not start another `go2_sim_suite.py` battery (or `p2_overnight.sh` stage) meanwhile.
- Add the v1.0 row to your `runs.md`: run, tag, training commit, checkpoint, flags, the card's headline numbers.

### 7.4 Reading the outputs

#### 7.4.1 The export
```
exported to logs/smoke-p32/exported: policy.pt, policy.onnx, deploy_cfg.json, export_check.json (MLP)
  policy.pt   max |export - policy|: random_100 0.00e+00, rollout_8x200 1.34e-07
  policy.onnx max |export - policy|: random_100 1.79e-07, rollout_8x200 1.19e-07
```
(a smoke run here)
- Expect ~1e-7: float32 round-off.
- 1e-5 to 1e-4 would point at a different code path (e.g. an observation normalizer that was not exported). Above
  1e-4 the script stops and writes `"passed": false`.
- The rollout check is the one that matters for a recurrent policy: it replays real sequences, with resets.
- torch 2.15 prints a `FutureWarning` that `torch.jit.load` is deprecated: expected, `policy.pt` still loads.

#### 7.4.2 `REPORT.md`
- One table per world, with the same metrics as sections 4.4 and 5.4.
- Read the flat tables first: they are the P1/P2 tables, so v1.0 can be compared with every earlier run.
- Then the frontiers (own vs unseen tiles: within a level means no overfitting to the tile instances), then the
  slopes (up vs down at the same angle).
- [P3v2] The seed band: the range column is the training noise of each number. A number whose range is wide
  (e.g. the frontier of one tile type moving by 2 levels between seeds) cannot support a claim of a 1-level gain.
- With `--ref`, the last table lists every headline number of both sides (mean [min–max] with a band), the change of
  the means (falls in percentage points), the verdict and which way is better. Only "better" / "worse" are results;
  "within noise" means the seeds cannot tell.

#### 7.4.3 The card
- Read the checklist first. Every `NO` names its reason. For a baseline, all lines must be `ok` (`--strict`).
- Then check the **Command** line. It is exactly what trained the policy: flags, iterations, envs, seed.
- The **Decisions** section must agree with the command (the checklist verifies it).

### 7.5 Git tag and release
`go2_card.py --freeze baselines/v1.0` prints, for example:
```bash
git add baselines/v1.0 decisions
git commit -m "Baseline v1.0: go2-v1.0 checkpoint 1999 (config card, report, export hashes)"
git tag -a baseline-v1.0 -m "Baseline v1.0: go2-v1.0 checkpoint 1999, trained from <commit>"
git push origin HEAD baseline-v1.0
gh release create baseline-v1.0 logs/go2-v1.0/exported/policy.pt logs/go2-v1.0/exported/policy.onnx \
    logs/go2-v1.0/exported/deploy_cfg.json logs/go2-v1.0/terrain.npz logs/go2-v1.0/model_1999.pt \
    --verify-tag --title "Baseline v1.0" --notes-file baselines/v1.0/CONFIG_CARD.md
```
- The tag points at the commit that adds `baselines/v1.0/` on top of the training commit. The card names the
  training commit, and "code unchanged" proves that no `.py` file changed in between.
- `--verify-tag` makes gh use the tag you pushed; without it, gh would create a missing tag from the default branch's
  latest state.
- The release holds what git does not: the policy files, `deploy_cfg.json`, the checkpoint (to fine-tune or distill
  from later) and the exact grid. The card is its description.
- Later, to check a checkout against the baseline: `git checkout baseline-v1.0 && python go2_provenance.py
  logs/go2-v1.0`.

### 7.6 Later candidates against v1.0
```bash
for R in go2-v2-s2 go2-v2-s3; do python go2_report.py -e $R --ckpt $CKPT; done
python go2_report.py -e go2-v2 --ckpt $CKPT --seeds go2-v2-s2 go2-v2-s3 --ref baselines/v1.0
```
- The "Against" table shows every headline number with both bands, the change of the means and the verdict.
- Train candidates with the same `$V1FLAGS` plus the change under test, on 3 seeds, as for P3.2 and P3.3. A single
  candidate run against the v1.0 band gets "better" / "worse" only outside the whole band.
- `--ref` takes a run name, a run folder, or a frozen baseline folder: `--ref baselines/v1.0` compares with the
  `summary.json` frozen in git, on any machine. It is read before the evaluation starts, so a typo fails at once.

### 7.7 Knobs

| Knob | Where | Default | Effect |
|---|---|---|---|
| ONNX export | `go2_export.py --onnx` | off | needed for a clean freeze (the card checks it) |
| check size | `--check-envs`, `--check-steps` | 8 × 200 | longer = more of the state space replayed |
| tolerance | `EXPORT_TOL` (`go2_export.py`) | 1e-4 | float32 round-off is ~1e-7 |
| evaluation size | `go2_report.py --reps` | 20 | envs per bin, as in `go2_metrics.py` |
| evaluation seed | `go2_report.py --seed` | 0 | the same for every run you compare |
| the standard set | `WORLDS`, `SLOPES` (`go2_report.py`), `STANDARD_WORLDS` (`go2_card.py`) | 10 worlds | change both lists together |
| frontier threshold | `FRONTIER_THROUGH` (`go2_report.py`) | 80 % | |
| key commands | `FLAT_BINS` (`go2_report.py`) | 6 bins | rows of the flat tables |
| baseline report size | `REPORT_MIN_REPS` (`go2_card.py`) | 20 | below it, "report complete" is NO |
| seed band | `go2_report.py --seeds`; `BAND_MIN_RUNS` (`go2_card.py`) | 3 runs | [P3v2] the card's `seed band` line |
| band keys on the card | `BAND_KEYS` (`go2_card.py`) | 5 numbers | [P3v2] shown as mean [min–max] |
| gate python | `go2_report.py --deploy-python`, `$DEPLOY_PY` | this python | [P3v2] the unitree_mujoco + unitree_sdk2py environment |
| card title, tag | `go2_card.py --name`, `--tag` | run name | the tag is derived from the name |
| release files | `RELEASE_FILES` (`go2_card.py`) | policy files, deploy_cfg, terrain + the checkpoint | |

### 7.8 Symptoms → fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| `--onnx: No module named 'onnxscript'` (or `onnx`) | torch ≥ 2.9 exports through onnxscript | `pip install onnx onnxscript onnxruntime` |
| `exported network does not match the training policy` | a module the export does not carry, or an rsl_rl version change | compare `random_100` and `rollout_*`; check that `rsl-rl-lib` is 5.5.1 |
| a `policy.onnx.data` next to `policy.onnx` | exported by rsl_rl directly, not through `go2_export.py` | re-export with `go2_export.py --onnx` (one file) |
| card: **NO** code committed | trained from a dirty tree (`git_diff.patch` holds the diff), or with untracked `.py` files | commit, retrain; or freeze knowingly, with the patch in `baselines/` |
| card: **NO** code unchanged | a `.py` file was edited after training | `python go2_provenance.py logs/<exp>` lists them; revert them (`git stash`) or retrain |
| card: **NO** report complete | a world failed or was not run; tile subsets; `--reps` below 20 | `report/logs/<world>.log`; rerun with the defaults (`--skip-existing` for missing worlds only) |
| card: **NO** decision slip / critic | the run does not follow `decisions/` | retrain with `$(python go2_decide.py flags)` |
| card: Command "unknown (manifest before P3.4)" | a run trained before P3.4 | fine for comparisons; a baseline should be trained with the P3.4 code |
| `--gate`: no export | the gate runs the exported policy in unitree_mujoco | `go2_export.py --onnx` first |
| `--gate: ... cannot import unitree_sdk2py` | the report would start the gate with the Genesis python (P2: two environments) | `--deploy-python` or `DEPLOY_PY=...` (3.1) |
| `another go2_sim_suite.py battery is running` | a second MuJoCo batch while one runs | wait for it; results of two parallel batches are invalid |
| `WARNING: EXP is not this run's configuration: [...]` | `--seeds` names a run with other flags | only seeds of one configuration form a band; the card says NO |
| card: **NO** seed band | no band, fewer than 3 runs, or not one configuration | report the seeds, then rebuild with `--collect --seeds ...` |
| gate suite fails to start | unitree_mujoco or DDS setup | run `go2_sim_suite.py --suite gate` by hand, then `go2_report.py --collect --gate-summary out/suite_*/summary.csv` |
| `--ref: no summary.json for EXP` | the reference run has no report on this machine | `--ref baselines/v1.0` (the frozen summary), or run `go2_report.py` on it |
| a second `--set` seemed ignored (runs before P3.4) | argparse kept only the last `--set` list | fixed: the lists add up now |

### 7.9 Done when
- `go2_card.py --strict` passes: every checklist line `ok` (with the seed band).
- The tag `baseline-v1.0` is pushed, and the release on it holds `policy.pt`, `policy.onnx`, `deploy_cfg.json`, the
  checkpoint and `terrain.npz`, with the card as its notes.
- The card, the release and the tag point to each other: training commit in the card, card in the release, release
  on the tag.

Results log to fill in (from `REPORT.md` and the card):

| Run | Seed | flat lin_err 0.5 | flat_dr falls | frontier hill / pit / obst (own) | mean through own / unseen | vx_meas ±20° | through ±25° | gate |
|---|---|---|---|---|---|---|---|---|
| go2-v1.0 | 1 | | | | | | | |
| go2-v1.0-s2 | 2 | | | | | | | – |
| go2-v1.0-s3 | 3 | | | | | | | – |
| band | mean [min–max] | | | | | | | – |

---

## 8. Reference: configuration keys and constants

### env_cfg (P3.1)
| Key | Default | Set by |
|---|---|---|
| `terrain` | None (flat) | `--terrain` → `{"mode": "grid"}`; `--set env.terrain.KEY=VALUE` (4.7) |
| `termination_if_roll_greater_than`, `termination_if_pitch_greater_than` | 30 (P1.2) | 55 with `--terrain` (`max_slope_deg` + `TERRAIN_TILT_MARGIN_DEG`), unless `--set` |
| `reset_forward` | absent → on for terrain, off on flat | `--set env.reset_forward=True` |

### command_cfg (P3.1)
| Key | Default | Set by |
|---|---|---|
| `curriculum` | None | `--cmd-curriculum` → `CMD_CURRICULUM` (4.7) |
| `lin_vel_x_range`, `lin_vel_y_range`, `ang_vel_range` | [−1, 1], [−0.5, 0.5], [−1, 1] | the curriculum's maximum |

### reward_cfg, obs_cfg, train_cfg (P3.2, P3.3)
| Key | Default | Set by |
|---|---|---|
| `reward_scales["foot_slip"]` | absent (off) | `--foot-slip [W]` → −W (W = `FOOT_SLIP_WEIGHT` = 0.1 when given alone) |
| `foot_slip_mode` | absent → `"xy"` | `--set reward.foot_slip_mode=tangent` |
| `contact_force_threshold` | 1.0 N (P1.1) | also the contact of the penalty and of the `foot_slip` entry |
| `obs_cfg["privileged"]` | absent (no group) | `--privileged critic` or `teacher` → `PRIVILEGED_DEFAULT`, or `--privileged-entries` |
| `train_cfg["obs_groups"]` | actor and critic `["policy"]` | critic: `["policy", "privileged"]`; teacher: the actor too |

### Checkpoint `infos["curriculum"]` (every P3.1 checkpoint)
`{"env_steps", "command_curriculum": bool, "command_ranges": {lin_vel_x, lin_vel_y, ang_vel},
"terrain": {"types", "num_levels", "max_slope_deg", "key", "level_counts": {type: [count per level]}}}` (terrain
only on grid runs). Read it with:
```bash
python -c "import torch; print(torch.load('logs/go2-p31/model_1999.pt', map_location='cpu', weights_only=False)['infos'])"
```

### Constants
- `go2_terrain.py`:
  - `TERRAIN_DEFAULTS`, `TILE_TYPES`, `GENERATION_KEYS`;
  - `GENERATOR_VERSION` (1; bump it when you change a generator, so caches regenerate);
  - `FOOTPRINT_HALF` (0.4 m), `BODY_LENGTH` (0.4 m).
- `go2_train.py`: `CMD_CURRICULUM`, `TERRAIN_TILT_MARGIN_DEG` (30°); [P3.2] `FOOT_SLIP_WEIGHT` (0.1); [P3.3]
  `PRIVILEGED_DEFAULT` (7 entries, 36 values).
- `go2_env.py`: `TERRAIN_FAST_BUILD` (True), `skip_terrain_support_field()`, `_SUPPORT_FIELD_SOURCES`; [P3.3]
  `PRIVILEGED_SIZES`, `HEIGHT_SCAN_X` / `HEIGHT_SCAN_Y` (17 × 11 points), `HEIGHT_SCAN_OFFSET` (0.3 m).
- `go2_metrics.py`: `THROUGH_FRAC` (0.75), `TERRAIN_TABLES`; [P3.2] `SLIP_SHARE_V` (0.02 m/s), `FOOT_TABLES`,
  `CONTACT_N` (3 N, P1).
- `go2_decide.py`: `SLIP_DROP` (0.30), `FOOT_WORSE` (0.10), `LIN_ERR_TOL` (5 %, 0.005 m/s), `THROUGH_TOL` (5 points),
  `COT_TOL` (5 %), `LEARN_REACH` (0.9), `LEARN_TOL` (5 %), `FINAL_TOL` (2 %), `LEVEL_TOL` (0.2), `MOVING` (0.1).
  [P3v2] A cost also needs every --on seed worse than every --off seed (`all_worse()`; chance rate `chance()`).
- [P3.4] `go2_export.py`: `EXPORT_TOL` (1e-4).
- [P3.4] `go2_report.py`: `WORLDS`, `SLOPES` (±10, ±20, ±25°), `FLAT_BINS` (6 commands), `FLAT_METRICS`,
  `SLOPE_METRICS`, `FRONTIER_THROUGH` (80 %), `HEADLINE_BETTER`.
- [P3.4] `go2_card.py`: `STANDARD_WORLDS` (the 10 worlds), `REPORT_MIN_REPS` (20), `RELEASE_FILES`; [P3v2]
  `BAND_MIN_RUNS` (3), `BAND_KEYS` (the 5 numbers shown with their band).
- [P3v2] `go2_report.py`: `SEED_KEYS` (what may differ between the seeds of a band: run names, seed).

### `report/summary.json` "band" (P3 v2, with `--seeds`)
`{runs: [exp, ...], n, same_config: true | false | null (not checked), config_differs: {run: [keys]}, warnings: [...],
stats: {headline key: {values: [per run], mean, min, max, share}}}`; `share` marks counts "a/b", stored as shares.

### `manifest.json` (every run; complete from P3.4)
| Field | Content |
|---|---|
| `git_hash`, `git_dirty`, `git_untracked_py` | commit at the start; tracked files changed; `.py` files git does not track (`null` without git) |
| `argv`, `args` | the command line; every argument after parsing |
| `created`, `host`, `gpu` | start time (ISO, with time zone), machine, GPU name (`null` on CPU) |
| `python`, `torch`, `cuda`, `genesis`, `rsl_rl` | versions |
| `code_sha256` | `{file.py: sha256[:16]}` for every `.py` file of the code folder |
| `cfgs` | the five configs, as in `cfgs.pkl` |

### `exported/export_check.json` (P3.4)
`{exp_name, ckpt, exported, tolerance, passed, max_abs_error: {file: {random_100, rollout_NxT}}, sha256: {policy.pt,
policy.onnx, deploy_cfg.json}, recurrent, obs_dim, onnx_io: {inputs: {name: shape}, outputs}, git_commit, git_dirty}`

### Files a P3 run writes
`logs/<exp>/`:
- `cfgs.pkl`, `model_*.pt` (with `infos["curriculum"]`), `terrain.npz` (grid runs), `git/` (rsl_rl's, empty: 7.1.1);
  [P3.4] `manifest.json`, `git_diff.patch` (dirty tree only);
- after evaluation: `metrics_*_grid.{csv,npz}`, `terrain_*_grid.png`, `gait_*_grid.png`, `metrics_*_slope±D.*`
  ([P3.2] terrain `.npz` also hold `foot_normal`);
- `exported/` after export ([P3.3] `deploy_cfg["source"]["critic_privileged"]` for a privileged-critic run; [P3.4]
  `policy.onnx` with `--onnx`, `export_check.json`);
- [P3.4] `report/` (`REPORT.md`, `summary.json`, `plots/`, `logs/<world>.log`) after `go2_report.py`; `CONFIG_CARD.md`
  after `go2_card.py`.

`decisions/` (repository root, committed): `slip.md`, `critic.md`, written by `go2_decide.py`. [P3.4]
`baselines/<version>/` (committed): the card and the small records, written by `go2_card.py --freeze`.

---

## 9. Reference: metrics glossary (P3)

| Column / output | Where | Definition | Better |
|---|---|---|---|
| `terrain`, `level` | grid tables, CSV | the tile of the bin | – |
| `dist[m]` | Progress (terrain modes) | straight-line horizontal distance from the start point at the end of the valid window (at the fall for a robot that fell) | higher (≤ 4 m on the grid at 0.5 m/s) |
| `through[%]` | Progress | 100 × share of robots that never fell and reached `--through` × the planned distance (command speed × warm-up + measurement); n/a below 0.1 m/s or when turning | higher |
| `z[m]`, `z_err[cm]` | Tracking (terrain modes) | base height above the mean ground under the feet; error vs 0.30 m + dh | z_err lower |
| `swing_h[cm]` | Gait (terrain modes) | peak clearance above the ground under the foot, per swing | near the target |
| `falls` | every table | fallen / envs; fall = base contact, or tilt > `--fall_deg` + the ground's tilt | lower |
| `ood` `*` | every table | the command is outside the training ranges (the ranges reached by the command curriculum, when the checkpoint has them) | – |
| heat map | `terrain_*_grid.png` | `through[%]` per (type, level), with falls % | darker, further right |
| `slip_v[cm/s]` | Foot slip | [P3.2] mean speed of the contact point along the ground in stance (\|F\| > 3 N, first sample of each stance dropped); on flat ground the P1 value | lower |
| `slip_d[cm/step]` | Foot slip | slip distance per complete stance | lower |
| `slip_share[%]` | Foot slip | [P3.2] share of stance samples sliding faster than 2 cm/s | lower |
| `pen_xy[cm/s]`, `pen_t[cm/s]` | Foot slip | [P3.2] the foot-slip penalty's sum per step (feet with \|F\| > 1 N) in mode `xy` (foot-centre planar speed) and `tangent` (contact-point slip); × W = reward per second | lower; the gap ≈ rolling |
| `slip_v_FL[cm/s]` … `slip_share_RR[%]` | Foot slip per foot (printed `v / d / share`), CSV | [P3.2] the same per foot | lower; left ≈ right |
| `sym_err[rad]` | Gait pattern & symmetry | P2.3; [P3.3] mirrors every group the actor reads | lower |
| `decisions/<rule>.md` | `go2_decide.py` | [P3.2/P3.3] runs, rule, evidence, KEEP / DROP, implied flags | – |
| frontier | `go2_report.py`, card | [P3.4] per tile type, the last level of the run of levels, from the lowest, with `through` ≥ 80 %; "none" (−1) when the lowest level fails | higher |
| mean through | `go2_report.py` | [P3.4] mean `through[%]` over every tile of a grid world | higher |
| `max \|export − policy\|` | `go2_export.py`, `export_check.json` | [P3.4] largest action difference between an exported file and the training policy, per input set (`random_100`, `rollout_NxT`) | ~1e-7; the script stops above 1e-4 |
| change, better | `REPORT.md` with `--ref` | [P3.4] candidate − reference (falls: percentage points of the share); the direction that counts as better | – |
| seed band | `REPORT.md`, `summary.json`, card | [P3v2] a headline number over the training seeds of one configuration: per run, mean, range (max − min) | a narrow range |
| verdict | `REPORT.md` with `--ref` | [P3v2] better / worse when the two sides' seed ranges do not overlap; "within noise" otherwise; "–" for one run on each side | – |
| `Curriculum/*` | TensorBoard | the tags of 3.5 | see 4.5 |

Rules of thumb:
- CoT uphill ≈ CoT flat + sin θ.
- `dist` on a face ≈ `vx_meas` × cos θ × time.
- `terrain_level` ceiling = 6.0 with 10 levels.

---

## 10. Troubleshooting

| Message / symptom | Cause | Fix |
|---|---|---|
| `ValueError: unknown terrain keys [...]` | typo in `--set env.terrain.KEY` | the message lists the valid keys |
| `tile_size must be a multiple of horizontal_scale` | e.g. 8 m at 0.3 m | pick a divisor |
| `terrain types must be a non-empty {type: share > 0} dict` | a type name or share wrong | types: flat, rough, hill, pit, wave, obstacles |
| `[go2_env] Genesis X: SupportField.activate differs from 1.4.2; terrain built the default way` | another Genesis version: the build fix is not installed | 1.4.3+ has the fix upstream (fine); otherwise expect a slow build and +4 GB |
| killed during `scene.build()` | RAM: grid too large (4.8) | smaller `border`, fewer levels, `horizontal_scale` 0.1 not 0.05 |
| `XML ... version` error on `plane.urdf` | Genesis' bundled `plane.urdf` (flat ground only) | `python fix_plane_urdf.py` (your runs.md note) |
| `height field (...) does not match the config` | a `terrain.npz` with a different layout given as `file` | let `make_ground` regenerate, or use the matching config |
| cache gives old tiles after editing a generator | cache key unchanged | bump `GENERATOR_VERSION` or delete `~/.cache/go2_terrain` |
| `--slope and --terrain grid exclude each other` | both given | one per evaluation |
| `--grid_cmd: give a planar speed of at least 0.1 m/s` | `--grid_cmd 0 0 0.5` | the tiles need walking; turning-only is not a grid test |
| `--types [...]: not in this grid` / `--levels: this grid has levels 0 .. L-1` | subsets outside the grid | check `python go2_terrain.py --run logs/<exp>` |
| `--tile: ... is not a --terrain grid recording` | feet plot on a flat or slope `.npz` | use `--cmd` there |
| `no tile hill 9 in ...` | the evaluation used `--types` / `--levels` | the message lists the tiles it has |
| `(no curriculum state in that checkpoint)` | `--init-from` a P1/P2 checkpoint | expected: levels and ranges start from their defaults |
| `size mismatch` in `--init-from` | different network: observation size (`--height-cmd`, `--gait wtw`) or `--rnn` | same P1/P2 flags as the source run |
| export prints smaller command ranges than configured | the command curriculum stopped there | train longer, or lower the threshold; the robot is limited to them on purpose |
| teleop J / K / U do nothing | no `--terrain` | `--terrain` (the keys are only registered then) |
| `--set command.curriculum.threshold=...` seems to do nothing | given without `--cmd-curriculum` | add the flag (3.2) |
| every robot "falls" on flat tiles but not on others in `--terrain grid` | a policy at the edge of the 10° limit: flat tiles get no tilt allowance | `--fall_deg`; check the same policy on the flat tables |
| [P3.2] `reward.foot_slip_mode must be 'xy' or 'tangent'` | typo in `--set` | |
| [P3.2] `go2_decide.py: no go2_metrics.py CSV common to every run` | a world or the checkpoint is missing for some run | run the evaluation loop of 5.4.1 for every run (tagged CSVs are ignored) |
| [P3.3] `obs_cfg['privileged']: unknown entries [...]` / `listed twice` | `--privileged-entries` typo | the message lists the known entries |
| [P3.3] `obs_groups [...] read 'privileged' but obs.privileged lists no entry` | `--set train.obs_groups...` alone | `--privileged critic` |
| [P3.3] `go2_symmetry: no mirror rule for the privileged entry` | a new entry without its mirror | add it to `_privileged_mirror()` (6.6) |
| [P3.3] `go2_export.py`: the actor reads `['policy', 'privileged']` | a teacher run | by design: distill a student first |
| [P3.3] `--privileged-entries needs --privileged critic or teacher` | entries without a reader | add `--privileged` |
| [P3.4] `--onnx: No module named 'onnxscript'` (or `onnx`, `onnxruntime`) | torch ≥ 2.9 exports through onnxscript; the check runs onnxruntime | `pip install onnx onnxscript onnxruntime` |
| [P3.4] `exported network does not match the training policy (worst ... > 0.0001)` | the export differs from the training policy | 7.8 |
| [P3.4] `WARNING: uncommitted changes; the run records them in .../git_diff.patch` | dirty tree when training starts | fine for tests; commit before the baseline run |
| [P3.4] `WARNING: untracked .py files [...]` | a `.py` file git does not track | commit it, or list it in `.gitignore` |
| `Could not find git repository in .../rsl_rl/__init__.py. Skipping.` | rsl_rl's own code-state logging (7.1.1) | harmless |
| [P3.4] `--ref: no summary.json for EXP` | no report of the reference on this machine | `--ref baselines/v1.0`, or run `go2_report.py` on it |
| [P3.4] `--gate: no export in ...` | the gate suite runs the exported policy | `go2_export.py --onnx` first |
| [P3.4] a `NO` line on the config card | see the reason on the line | 7.8 |
| [P3v2] `--gate: ... cannot import unitree_sdk2py` | the gate would run with the Genesis python | `--deploy-python` / `$DEPLOY_PY` (3.1) |
| [P3v2] `another go2_sim_suite.py battery is running (... is locked)` | two MuJoCo batches at once | one at a time; the lock goes when the first ends |
| [P3v2] `WARNING: EXP is not this run's configuration: [...]` | `--seeds` lists a run with other flags | only seeds of one configuration form a band |
| [P3v2] `--seeds: no report for EXP` | that seed has no `report/summary.json` | run `go2_report.py -e EXP --ckpt ...` first |

---

## 11. Sources

- Rudin, Hoeller, Reist, Hutter, "Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning",
  CoRL 2021 (PMLR 164, 2022), arXiv:2109.11978: the game-inspired terrain curriculum.
- legged_gym:
  - `legged_gym/utils/terrain.py` (`Terrain`, `curiculum()`, `make_terrain()`);
  - `legged_gym/envs/base/legged_robot.py` (`_update_terrain_curriculum`, `update_command_curriculum`,
    `_get_env_origins`, `privileged_obs_buf`);
  - `legged_robot_config.py`: terrain 8 m tiles, 10 rows, 25 m border, 0.1 m / 0.005 m scales,
    `max_init_terrain_level` 5; commands `resampling_time` 10 s, `heading_command` True, `curriculum` False,
    `max_curriculum` 1.
- Isaac Gym `terrain_utils.py` (bundled in Genesis 1.4.2 as `ext/isaacgym/terrain_utils.py`): `random_uniform_terrain`,
  `pyramid_sloped_terrain`, `discrete_obstacles_terrain`, `wave_terrain` (a sum of a sine and a cosine).
- Isaac Lab:
  - `isaaclab/terrains/terrain_generator.py` and `height_field/hf_terrains.py` (bilinear pyramid slopes);
  - `locomotion/velocity/mdp/curriculums.py` (`terrain_levels_vel`: up beyond half the tile size, down below half of
    |command| × episode length);
  - Spot `mdp/rewards.py` (`foot_slip_penalty`: planar foot velocity × contact above the threshold) and
    `spot/flat_env_cfg.py` (`foot_slip` −0.5, threshold 1.0; tracking 5.0 + 5.0);
  - `locomotion/velocity/mdp/rewards.py` (`feet_slide`: the same quantity, threshold 1.0).
- Genesis 1.4.2, installed package:
  - `options/morphs.py` (`Terrain`, `Plane`), `utils/terrain.py` (heightfield to mesh, triangle split);
  - `engine/solvers/rigid/collider/support_field.py` (`SupportField.activate`), `mpr.py` and `gjk_support.py`
    (`_func_support_prism`);
  - `engine/entities/rigid_entity/rigid_entity.py` (`set_qpos`, `skip_forward`, `get_dofs_position`).
- Genesis 1.4.3 release: terrain geoms skipped when building the support field.
- rsl-rl-lib 5.5.1, installed package:
  - `runners/on_policy_runner.py` (`save` / `load` with infos, `export_policy_to_onnx`, opset 18);
  - `utils/utils.py` (`resolve_obs_groups`);
  - `algorithms/ppo.py` (observation sets `actor` / `critic`) and `algorithms/distillation.py` (`student` /
    `teacher`).
- Lee, Hwangbo, Wellhausen, Koltun, Hutter, "Learning Quadrupedal Locomotion over Challenging Terrain", Science
  Robotics 2020, arXiv:2010.11251: privileged teacher and proprioceptive student.
- Pinto, Andrychowicz, Welinder, Zaremba, Abbeel, "Asymmetric Actor Critic for Image-Based Robot Learning", RSS 2018,
  arXiv:1710.06542: a critic with full state, an actor with partial observations.
- [P3.2] Isaac Lab Spot `foot_slip_penalty` (`config/spot/mdp/rewards.py`): contact = max over the force history of
  \|F\| > threshold; penalty = Σ contact × \|body_lin_vel_w[:2]\|. Weight −0.5, threshold 1.0, bodies `.*_foot`
  (`config/spot/flat_env_cfg.py`).
- [P3.3] legged_gym `legged_robot_config.py`: `num_privileged_obs` ("if not None a priviledge_obs_buf will be returned
  by step()"), `measured_points_x` (−0.8…0.8) / `measured_points_y` (−0.5…0.5), `obs_scales.height_measurements` 5.0;
  `legged_robot.py`: heights = clip(base z − 0.5 − h, −1, 1) × scale, points rotated by the base yaw
  (`quat_apply_yaw`).
- [P3.3] rsl-rl-lib 5.5.1, installed package: `models/mlp_model.py` (`get_latent`: groups concatenated in
  `obs_groups` order; `obs_normalization` False by default), `extensions/symmetry.py` (`augment_batch`,
  `compute_loss`: the whole observation TensorDict), `storage/rollout_storage.py` (every group stored),
  `algorithms/ppo.py` (`construct_algorithm`, default sets `actor` / `critic`).
- [P3.2] Genesis 1.4.2, installed package: `RigidEntity.get_contacts()` (contact `"normal"` and `"position"`;
  `collider.py` contact data), `get_links_ang()`, `get_links_net_contact_force()`.
- [P3.4] rsl-rl-lib 5.5.1, installed package: `runners/on_policy_runner.py` (`export_policy_to_jit`,
  `export_policy_to_onnx` with opset 18, `add_git_repo_to_log`), `models/mlp_model.py` and `models/rnn_model.py`
  (`as_onnx`: input and output names), `utils/logger.py` (`_store_code_state`: `git/<repo>.diff`).
- [P3.4] torch 2.15, installed package: `onnx/__init__.py` (`export`: `dynamo=True` by default since 2.9,
  `external_data=True`), `onnx/_internal/exporter/_core.py` (imports onnxscript), `jit/_serialization.py` (the
  `torch.jit.load` deprecation warning). PyTorch 2.8 release notes draft,
  [pytorch-labs/torch-release-notes#17](https://github.com/pytorch-labs/torch-release-notes/pull/17): `dynamo=True` by
  default from 2.9, TorchScript "nearing end-of-life".
- [P3.4] onnx 1.23.2, installed package: `external_data_helper.py`. ONNX Runtime
  [Python API](https://onnxruntime.ai/docs/api/python/api_summary.html): `InferenceSession`, `get_inputs`, `run`.
- [P3.4] GitHub CLI manual, [gh release create](https://cli.github.com/manual/gh_release_create): assets, `--title`,
  `--notes-file`, `--verify-tag` (otherwise a missing tag is created from the default branch's latest state).
- [P3v2] `P2_SUMMARY.md` (your P2 results, 2026-11-06): the final configuration, the run names, the training-noise
  band, the two conda environments, one MuJoCo batch at a time, energy against terrain, the commands still weak.
