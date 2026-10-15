# Test and tinker guide (companion to CHANGES.md)

`CHANGES.md` says what changed in each phase and why. This file says how to run each phase, which knobs to turn, and how to read the results.

Everything here was run on CPU with Genesis 1.4.2 and rsl-rl-lib 5.5.1, except training to convergence (that needs your GPU). So this guide gives no "good values" for a trained policy: compare your runs with each other.

Contents
- 0. Ground rules
- 1. Files at a glance
- 2. First session on your machine (smoke tests)
- 3. On/off switches, all in one place
- 4. Reading the training logs (TensorBoard)
- 5. Reading `go2_metrics.py`, comparing runs with `go2_compare.py`
- 6. P1.1 stand still
- 7. P1.2 domain randomization and actuation
- 8. P1.3 safety and limits
- 9. P1.4 smooth actions
- 10. P1.5 stand-up, sit-down, deployment
- 11. P1.6 body height, keyboard, RNN
- 12. PPO and network knobs
- 13. Recipes: add a reward, a metric, a command; fine-tune; make a video
- 14. Troubleshooting
- 15. Experiment log template

## 0. Ground rules

- **Where to run.** Run each command from the folder that holds the `go2_*.py` files. Training creates `logs/<name>/` there, and every other script reads it.
  - To tinker, work in `p1_6/`. Every knob below exists there, and its default flags reproduce P1.5 exactly (same weights, bit for bit).
  - To compare phases, train each phase from its own snapshot folder (`p1_1/` ... `p1_6/`), or train in `p1_6/` with the flags of each phase where they exist.
  - The commands below are written for `p1_6/`. Some options only exist there: `--tag` (metrics), `--fallen` (eval), `go2_compare.py`, `go2_teleop.py`.
- **One name per variant.** `go2_train.py` **deletes `logs/<name>/`** if it exists. Use a new `-e` for every variant, e.g. `go2-p13-coll5`.
- **Checkpoints.** Every 100 iterations, plus the last one. A 1000-iteration run has `model_0.pt` ... `model_900.pt` and `model_999.pt`. There is no `model_1000.pt`: always `--ckpt 999` for the last one.
- **What a run was trained with.** The five config dicts are saved in `logs/<name>/cfgs.pkl` at training time. Eval, metrics, export and teleop read that file, so editing `go2_train.py` afterwards does not change an existing run. To print it:
  ```
  python -c "import pickle; e,o,r,c,t = pickle.load(open('logs/go2-p16h/cfgs.pkl','rb')); print(c); print(r['reward_scales']); print(t['actor'])"
  ```
- **Where the knobs live.**
  - Values: `get_cfgs()` and `get_train_cfg()` in `go2_train.py`. They form 5 dicts: `env_cfg`, `obs_cfg`, `reward_cfg`, `command_cfg`, `train_cfg`.
  - Reward functions: `_reward_<name>()` in `go2_env.py`. A term runs only if its name is a key of `reward_scales`. Delete the key to turn it off (a weight of 0 also works, but the term is still computed and logged as 0).
  - The env multiplies every weight by `dt` (0.02 s), so a weight is "per second".
- **One change at a time, same seed.** Compare a variant with its baseline at the same `--seed` (default 1). When the difference is small, train both again with `--seed 2`. Two seeds of the same config can differ as much as a small reward change does.
- **Same evaluation for every run.** Use the `go2_metrics.py` defaults (nominal world, 20 envs per command bin, 10 s, seed 0) unless the test needs otherwise. When you evaluate the same checkpoint a second way (e.g. `--keep_dr`), add `--tag dr`, or the second run overwrites the first CSV.
- **Smoke test before a GPU run.** `--cpu -b 64 --max-iterations 3` catches a config typo in 1 to 2 minutes (section 2).
- **Windows.** The commands are the same in PowerShell. `VAR=value python ...` (headless rendering, section 11) is Linux syntax and only needed on a Linux machine without a display.

## 1. Files at a glance

| file | does | reads | writes |
|---|---|---|---|
| `go2_train.py` | trains (configs at the top) | – | `logs/<exp>/cfgs.pkl`, `model_*.pt`, TensorBoard events |
| `go2_env.py` | the environment: commands, PD at 200 Hz, randomization, rewards, terminations | – | – |
| `go2_eval.py` | viewer, one robot, until you close it | `cfgs.pkl`, `model_<ckpt>.pt` | – |
| `go2_metrics.py` | evaluation harness: tables per command bin | same | `metrics_*.csv`, `metrics_*.npz`, `gait_*.png`, `spectra_*.png` in `logs/<exp>/` |
| `go2_compare.py` | puts metrics CSVs side by side (no Genesis) | CSVs | – |
| `go2_teleop.py` | keyboard control in the viewer, video | `cfgs.pkl`, `model_<ckpt>.pt` | `.mp4` |
| `go2_export.py` | export for the robot | same | `logs/<exp>/exported/policy.pt`, `deploy_cfg.json` |
| `go2_fsm.py` | robot-side state machine (numpy, torch; no Genesis) | `exported/` | – |
| `go2_fsm_sim.py` | Genesis bench of the state machine | `exported/` (or none) | `fsm_<scenario>.png` |
| `go2_deploy_sdk2.py` | adapter to the robot (unitree_sdk2_python) | `exported/` | – |
| `go2_commands.py` | command deadzone and gamepad mapping, shared by training and robot | – | – |
| `go2_smooth_ppo.py` | PPO with LCP / CAPS (only with `--lcp`, `--caps-*`) | – | – |

## 2. First session on your machine (smoke tests)

These are the checks run here. Run them once before long trainings. The smoke policies are untrained: they fall and drift. These runs only check that the code paths work. Each script first spends a few seconds to a minute compiling Genesis kernels.

```
python go2_train.py -e smoke --cpu -b 64 --max-iterations 3
python go2_metrics.py -e smoke --ckpt 2 --cpu --reps 1 --warmup_s 0.2 --measure_s 0.6
python go2_metrics.py -e smoke --ckpt 2 --cpu --reps 1 --warmup_s 0.2 --measure_s 0.6
python go2_train.py -e smoke-hl --cpu -b 64 --max-iterations 3 --height-cmd --rnn lstm
python go2_metrics.py -e smoke-hl --ckpt 2 --cpu --reps 2 --warmup_s 0.5 --measure_s 2
python go2_export.py -e smoke-hl --ckpt 2
python go2_fsm_sim.py -e smoke-hl
python go2_teleop.py -e smoke-hl --ckpt 2
python go2_compare.py logs/smoke/metrics_ckpt2_seed0.csv logs/smoke-hl/metrics_ckpt2_seed0.csv
```

What you should see:
- training: `Learning iteration 2/3`, then `model_0.pt` and `model_2.pt` in `logs/smoke/`.
- the two metrics runs: the same `checksum:` line (determinism).
- `smoke-hl`: the printed models contain `LSTM(46, 256)` (46 inputs = 45 + the height command).
- its metrics: 15 rows, the last 4 with `dh` −0.06 and 0.03.
- export: `policy.pt (recurrent), deploy_cfg.json   (max |jit - policy| over 100 steps: 0.00e+00)`.
- bench: `[fsm   8.600 s] body height offset -1 cm` ... `-3 cm`, and `observation check: 275 policy steps, max |deploy obs - Go2Env obs| = 1.79e-07`.
- teleop: the key list in the terminal. Click the viewer window, press Up twice: `set: vx +0.20 vy +0.00 wz +0.00 dh +0 cm  -> policy gets: vx +0.20 ...`. Close the window to stop.
- compare: a `falls` table, then one table per metric with columns A and B.

## 3. On/off switches, all in one place

| switch | where | default | effect |
|---|---|---|---|
| `--height-cmd` | `go2_train.py` flag | off | 4th command: body-height offset `dh` (P1.6) |
| `--rnn lstm` / `--rnn gru` | flag | off | recurrent actor and critic (P1.6) |
| `--recovery` | flag | off | 15 % fallen starts + recovery rewards (P1.5) |
| `--lcp X`, `--caps-t X`, `--caps-s X` | flags | 0 = off | smoothness regularizers (P1.4); MLP only |
| `--cpu` | flag | off | CPU backend, for smoke tests |
| `obs_cfg["add_noise"]` | `go2_train.py`, `get_cfgs()` | `True` | observation noise (P1.2) |
| `env_cfg["domain_rand"]` keys | same | all 9 present | delete a key to turn that randomization off; `{}` = none (P1.2) |
| `domain_rand["push_interval_s"]` | same | 15 | 0 = no pushes |
| `reward_cfg["only_positive_rewards"]` | same | `True` | step reward clipped at 0 (P1.3) |
| `env_cfg["terminate_after_contacts_on"]` | same | `["base"]` | `[]` = no base-contact termination (P1.3) |
| `command_cfg["rel_standing_envs"]` | same | 0.1 | 0 = no exact-zero command envs (P1.1) |
| `LIN_DEADZONE`, `ANG_DEADZONE` | `go2_commands.py` | 0.2, 0.2 | 0 = no deadzone, in training **and** on the robot (P1.1) |
| any reward term | `reward_scales` | – | on when its key is present |
| `enable_self_collision` | `go2_env.py`, `RigidOptions` | `False` | legs collide with each other; may need a larger `max_collision_pairs` (20) |
| `schedule` | `train_cfg["algorithm"]` | `"adaptive"` | `"fixed"` keeps `learning_rate` constant (section 12) |
| `obs_normalization` | `train_cfg["actor"]`, `["critic"]` | absent = `False` | rsl_rl running mean/std of the observation; exported with the policy (not tried here) |
| eval `--cmd`, `--dr`, `--fallen` | `go2_eval.py` | off | pin the command; training randomization (no fallen starts); every episode starts fallen |
| metrics `--keep_dr`, `--tag`, `--init_noise 0` | `go2_metrics.py` | off | training randomization; file-name suffix; identical starts |
| teleop `--dr`, `--headless`, `--record` | `go2_teleop.py` | off | training randomization; no viewer; video |
| bench `--dr`, `--viewer`, `--scenario estop`, `--handover-s 0` | `go2_fsm_sim.py` | off | see section 10 |

## 4. Reading the training logs (TensorBoard)

`tensorboard --logdir logs`, then tick the runs to compare. The console of `go2_train.py` prints the same numbers.

| scalar | what it is | how to read it |
|---|---|---|
| `Train/mean_reward` | mean return (sum of the step rewards) of the last 100 episodes that ended | rises, then plateaus. Only comparable between runs with the same reward terms and weights. ≥ 0 with `only_positive_rewards` (recovery episodes excepted). |
| `Train/mean_episode_length` | mean length of the same 100 episodes, in policy steps; at most about 1000 (20 s / 0.02 s) | near 1000 = episodes end by time-out, not by falling. It starts low because the first episodes start at a random point (`init_at_random_ep_len`); after that, a drop means terminations (tilt, base contact). |
| `Train/mean_reward/time` | the same, against training time [s] | compare an RNN with an MLP on this axis too: RNN iterations are slower |
| `Episode/rew_<term>` | weight × term, per second, over the episodes that ended | see below |
| `Episode/recovery_success` | P1.5 `--recovery`: share of recovery episodes that end upright, base above 0.2 m | should climb toward 1. P1.6 fixed its logging (it read far too low in P1.5). |
| `Loss/value` | critic loss | drops and stays bounded. Spikes follow sudden reward changes. |
| `Loss/surrogate` | PPO policy loss | small and noisy around 0 |
| `Loss/entropy` | entropy of the action distribution, summed over the 12 actions (not a loss, despite the tag) | 17.0 at the start (std 1.0), then decreasing as `Policy/mean_std` decreases |
| `Loss/learning_rate` | adaptive schedule (`desired_kl` 0.01) | if it collapses toward 1e-5, the updates overshoot the KL target |
| `Policy/mean_std` | std of the action noise | decreases from 1.0 (`init_std`). If it stays near 1 or grows, the policy does not commit to actions. |
| `Perf/total_fps`, `Perf/collection_time`, `Perf/learning_time` | throughput; seconds per iteration for simulation and for the PPO update | collection grows with physics cost (P1.2: 4 solver steps per policy step), learning with the update cost (RNN, LCP, CAPS) |
| `Loss/grad_penalty`, `Loss/caps_temporal`, `Loss/caps_spatial` | P1.4 regularizers, before their coefficient | logged only when their coefficient is > 0 |

**`Episode/rew_<term>` in numbers.** The env adds weight × term × dt at every step, and divides the episode total by the episode length in seconds (20 s). For an episode that lasts its full 20 s, that is weight × (mean of the term). Episodes that end early give smaller magnitudes.
- Tracking terms: divide by the weight to get the mean of exp(−error² / 0.25), in [0, 1]. For a steady error e: 0.9 ≈ 0.16 m/s, 0.8 ≈ 0.24 m/s, 0.5 ≈ 0.42 m/s.
- Squared penalties: divide by the weight to get the mean of the squared quantity. Example: `rew_base_height` = −0.02 with weight −50 → mean (z − target)² = 0.0004 m² → 2 cm RMS height error.
- Rough cross-check: for full-length episodes and no clipping, `Train/mean_reward` ≈ 20 × the sum of the `Episode/rew_*` values.
- **Dilution.** Each step logs the mean over the envs that reset at that step, and 0 when none did; TensorBoard shows the average over all steps of the iteration. With 4096 envs, about 4 envs reset per step and the effect is about 2 %. With 64 envs (smoke tests), the values are pulled toward 0. Compare `Episode/*` only between runs with the same `-b`.
- To size a new penalty, start with a weight that makes its `rew_` a few % of `rew_tracking_lin_vel` early in training, then adjust.

## 5. Reading `go2_metrics.py`, comparing runs with `go2_compare.py`

**What the run does.**
- One env per (command bin × rep). Each env keeps its command for the whole run.
- 2 s of warm-up (discarded), then 10 s measured.
- Nominal world: no randomization, no observation noise, no pushes, no fallen starts, 20 ms latency.
- Initial joint angles get ±0.1 rad of seeded noise.
- A fall is a tilt over 10° (`--fall_deg`) or a base contact (P1.3+ runs). After a fall, that env's data stops counting, so each mean only covers the time before the fall.
- Every cell is mean ± std over the envs of the bin (20 by default). The std is the spread between envs, not a confidence interval on the mean. The standard error of the mean is std / √20 ≈ 0.22 × std.
- Recurrent policies get their memory cleared at every reset.

**Options**

| option | default | use |
|---|---|---|
| `--ckpt` | 100 | the checkpoint; 999 is the last one of a 1000-iteration run |
| `--seed` | 0 | joint-noise and simulator seed. Same seed + same checkpoint = same numbers. |
| `--reps` | 20 | envs per bin |
| `--warmup_s`, `--measure_s` | 2, 10 | transient discarded, window measured |
| `--keep_dr` | off | training randomization, noise and pushes (fallen starts stay off). Pushes come every 15 s, so add `--measure_s 20` to get one. |
| `--latency_ms` | 20 | latency in the nominal world. **Values above 20 are clamped to 20** (one policy step; section 7). |
| `--fall_deg` | 10 | fall threshold |
| `--init_noise` | 0.1 | rad, 0 = identical starts |
| `--tag` | none | suffix of the output names, e.g. `--tag dr` → `metrics_ckpt999_seed0_dr.csv` |
| `--cpu` | off | CPU backend |
| `--log_dir` | `logs/<exp>` | |

**Output**
- Printed tables, plus `logs/<exp>/metrics_ckpt<N>_seed<S>[_<tag>].csv` (one row per bin, `<metric>_mean` and `<metric>_std` columns).
- `gait_*.png` and `spectra_*.png` with the same name pattern.
- `metrics_*.npz`: every raw signal, for your own plots (section 13).
- `checksum`: the sum of every measured base velocity. Same code, checkpoint and options → same number. Use it to check that a code change leaves the simulation unchanged.

**Columns shared by every table**

| column | meaning |
|---|---|
| `vx vy wz` | the bin's command [m/s, m/s, rad/s] |
| `dh` | P1.6 runs with `--height-cmd`: body-height offset [m] |
| `ood` | `*` = outside the run's training ranges: expect worse numbers there |
| `falls` | fallen envs / envs in the bin |

**Tracking & stability**

| column | meaning | how to read it |
|---|---|---|
| `lin_err[m/s]` | mean of ‖v_xy − cmd_xy‖ | never 0: the base speed oscillates within each stride. Compare runs. |
| `yaw_err[rad/s]` | mean of \|wz − cmd_wz\| | same |
| `vx_meas`, `vy_meas`, `wz_meas` | mean measured velocities | bias. Example: `vx_meas` 0.85 at vx 1.0 = 15 % short. At (0, 0, 0), anything far from 0 = drift. |
| `z[m]` | mean base height (P1.6) | the PD stance alone (default pose, kp 20) sags to 0.27 m, so the policy has to push to reach the 0.30 m target |
| `z_err[cm]` | mean \|z − (0.30 + dh)\| (P1.6) | height tracking. Runs without the height command are measured against 0.30 m. |

**Smoothness**

| column | meaning | how to read it |
|---|---|---|
| `act_rate[rad/s]` | RMS over joints of how fast the joint targets change | the gait needs some; jitter adds a lot |
| `act_acc[rad/s2]` | RMS second difference of the targets / dt² | what the P1.4 `action_smoothness` reward penalizes |
| `hf>10Hz[%]` | share of the targets' signal power above 10 Hz, mean over joints | the gait (≈1–3 Hz and its first harmonics) sits below 10 Hz; above is jitter. Envs that fell are left out (n/a if all fell). |
| `tau_rms[Nm]` | RMS torque over joints and time, from the 200 Hz signal | motor heating goes with torque² |
| `tau>10Hz[Nm]` | RMS of the torque content above 10 Hz, 200 Hz signal | vibration and heat that do no useful work |

**Actuation limits**

| column | meaning | how to read it |
|---|---|---|
| `tau_peak[%lim]` | highest \|torque\| / limit, any joint, sampled once per policy step | close to 100 = the policy uses the whole motor |
| `tau_sat[%time]` | share of time at least one joint is at ≥ 95 % of its limit | should be 0 in nominal walking; above 0, the policy relies on the clip |
| `jlim_hits[/min]` | entries into the last 0.02 rad of a joint's URDF range, per minute | should be 0 |

**Energy**

| column | meaning | how to read it |
|---|---|---|
| `power[W]` | mean of Σ\|τ·q̇\| over joints (no energy recovered) | |
| `speed[m/s]` | mean planar speed | |
| `CoT[-]` | power / (m g speed) | lower = more efficient. Compare at the same command. n/a below 0.1 m/s commanded. |

**Gait**

| column | meaning | how to read it |
|---|---|---|
| `duty[-]` | share of time each foot is on the ground (> 3 N), mean over the 4 feet | 1.0 = never lifts (standing). A trot is about 0.5. |
| `step_freq[Hz]` | touchdowns per foot per second | ≈ 0 when standing still: the P1.1 test |
| `swing_h[cm]` | mean peak foot height over the complete swings | low values = shuffling, toe-stubbing risk |
| `nonfoot[%time]` | share of time any other link (calf, thigh, base) touches the ground | should be 0 |

**Foot slip**

| column | meaning | how to read it |
|---|---|---|
| `slip_v[cm/s]` | mean speed of the contact point during stance (rolling removed) | low = feet planted |
| `slip_d[cm/step]` | distance slid per stance | |
| `slip_ratio[-]` | 1 − distance gained / distance commanded, along the command | 0 = exact, > 0 = falls short, < 0 = overshoots. It includes tracking error, not only slip. n/a below 0.1 m/s. |

**Plots**
- `gait_*.png`: one representative env per bin (the first that never fell), 3 s. A bar = foot on the ground.
  - Trot: diagonal pairs (FL + RR, FR + RL) alternate.
  - Standing still: four full bars.
  - Tapping at zero command: short gaps in the bars.
- `spectra_*.png`: power spectral density of the joint targets (left, 0–25 Hz) and of the torques (right, 0–100 Hz), for (0, 0, 0), (0.5, 0, 0), (1, 0, 0) and (0, 0, 0.5).
  - The gait shows as peaks at the step frequency and its harmonics, left of the dashed 10 Hz line.
  - To the right, the curve should drop by orders of magnitude.
  - A bump near 25 Hz in the targets = an action flipping every policy step.

**Comparing runs**

```
python go2_compare.py logs/go2-p13/metrics_ckpt999_seed0.csv logs/go2-p14/metrics_ckpt999_seed0.csv
python go2_compare.py A.csv B.csv C.csv --cols lin_err act_acc "tau>10Hz" --rows "0 0 0" "0.5 0 0"
python go2_compare.py A.csv B.csv --all
python go2_compare.py logs/go2-p16h/metrics_ckpt999_seed0.csv logs/go2-p16h/metrics_ckpt999_seed0_dr.csv
```
- The first file is the reference (A). Each other cell shows the value and its change from A: in %, or as a difference `d...` when A is ~0 or the metric is compared to a command (`vx_meas`, `vy_meas`, `wz_meas`, `z`).
- `~` after a change = within 2 standard errors of A (computed from the stds and env counts in the CSVs): the difference may be noise. More `--reps` narrows it. A second training seed is the real test.
- Rows are matched on the command, so a 3-command run compares with a 4-command one on the shared rows (dh = 0).
- Default columns: `lin_err yaw_err z_err act_acc hf>10Hz tau>10Hz tau_peak CoT slip_v`. `--cols` matches the start of a name (`slip` gives the three slip columns).

## 6. P1.1 stand still at zero command

**Commands**
```
python go2_train.py -e go2-p11
python go2_eval.py -e go2-p11 --ckpt 999 --cmd 0 0 0      # does it stand still?
python go2_eval.py -e go2-p11 --ckpt 999 --cmd 0.5 0 0    # walking
python go2_eval.py -e go2-p11 --ckpt 999 --cmd 0.1 0 0    # inside the 0.2 deadzone: becomes (0, 0, 0)
python go2_eval.py -e go2-p11 --ckpt 999 --cmd 0 0 0.5    # turning in place
python go2_eval.py -e go2-p11 --ckpt 999                  # random commands as in training, new one every 4 s
python go2_metrics.py -e go2-p11 --ckpt 999
```

**Knobs**

| knob | where | default | effect |
|---|---|---|---|
| `lin_vel_x_range`, `lin_vel_y_range`, `ang_vel_range` | `command_cfg` | ±1, ±0.5, ±1 | the command space. Keep the metrics grid inside it (`ood` column). |
| `LIN_DEADZONE`, `ANG_DEADZONE` | `go2_commands.py` | 0.2, 0.2 | smaller commands become an exact 0. The robot side imports the same constants. |
| `rel_standing_envs` | `command_cfg` | 0.1 | share of the envs given exactly (0, 0, 0) at each resampling |
| `stand_still_scale` | `reward_cfg` | 5.0 | multiplies `similar_to_default` at zero command while the base is slow |
| `stand_still_vel_threshold` | `reward_cfg` | 0.5 m/s | above this base speed, the ×5 is off (lets the robot step after a push) |
| `feet_air_time` | `reward_scales` | 1.0 | pays (air time − target) at each touchdown; off at zero command |
| `feet_air_time_target` | `reward_cfg` | 0.5 s | shorter swings are penalized, longer ones rewarded |
| `contact_force_threshold` | `reward_cfg` | 1 N | foot contact for the air-time reward |
| `tracking_lin_vel`, `tracking_ang_vel` | `reward_scales` | 1.0, 0.5 | |
| `tracking_sigma` | `reward_cfg` | 0.25 | width of the tracking kernel: smaller = stricter |
| `resampling_time_s` | `env_cfg` | 4.0 | how often the command changes in training |
| `feet_height_target` | `reward_cfg` | 0.075 | unused: no reward reads it |

**What to read**
- Metrics, row (0, 0, 0): `step_freq` ≈ 0, `duty` ≈ 1.0, small `slip_v`, `lin_err` and `act_rate`, `falls` 0.
- Viewer, `--cmd 0 0 0`: no foot tapping, no slow drift. Without `--cmd`: clean switches between walking and standing.
- TensorBoard: `Episode/rew_feet_air_time` positive and rising.

**Experiments**
1. `rel_standing_envs` 0 / 0.1 / 0.2: standing quality against walking quality.
2. `stand_still_scale` 1 / 5 / 10.
3. Deadzone 0 vs 0.2 (`go2_commands.py`), with the (0.25, 0, 0) row in view.
4. `tracking_sigma` 0.25 vs 0.15.

**If it goes wrong**
- Taps its feet at zero command: raise `stand_still_scale`; check that `rel_standing_envs` > 0.
- Shuffles at low speed (row 0.25: `duty` near 1, low `step_freq`, high `slip_v`): try `feet_air_time_target` 0.3–0.4 s. That is a hypothesis to test, not a known fix.

## 7. P1.2 domain randomization and actuation

**Commands**
```
python go2_train.py -e go2-p12
python go2_metrics.py -e go2-p12 --ckpt 999                                          # nominal world
python go2_metrics.py -e go2-p12 --ckpt 999 --keep_dr --measure_s 20 --tag dr        # training randomization, with a push
python go2_metrics.py -e go2-p12 --ckpt 999 --latency_ms 0 --tag lat0                # latency sweep: 0, 10, 20
python go2_metrics.py -e go2-p12 --ckpt 999 --latency_ms 10 --tag lat10
python go2_metrics.py -e go2-p11 --ckpt 999                                          # P1.1 run in the same world
python go2_compare.py logs/go2-p12/metrics_ckpt999_seed0.csv logs/go2-p12/metrics_ckpt999_seed0_dr.csv
python go2_eval.py -e go2-p12 --ckpt 999 --dr                                        # watch the pushes (every 15 s)
python go2_eval.py -e go2-p12 --ckpt 999 --latency-ms 0                              # --latency-ms here, --latency_ms in metrics
```

**Knobs** (`env_cfg`, unless stated)

| knob | default | effect |
|---|---|---|
| `sim_dt`, `decimation` | 0.005, 4 | physics at 200 Hz, policy at 50 Hz. Changing them changes the policy period: leave them unless that is the experiment. |
| `kp`, `kd` | 20, 0.5 | policy PD gains; exported to `deploy_cfg.json`, so the robot side follows |
| `armature` | 0.01 | rotor inertia per joint (Menagerie) |
| `torque_limits` | hip/thigh 23.7, calf 45.43 Nm | clip and solver force range |
| `latency_s` | 0.02 | latency when it is not randomized |
| `domain_rand.friction_range` | [0.5, 1.25] | ground and robot |
| `domain_rand.added_mass_range` | [−1, 2] kg | on the base |
| `domain_rand.com_shift_range` | ±0.03 m | base CoM, each axis |
| `domain_rand.kp_range`, `kd_range` | ×[0.85, 1.15] | per env, per episode |
| `domain_rand.motor_strength_range` | ×[0.9, 1.1] | per env, joint and episode |
| `domain_rand.latency_range_s` | [0, 0.02] | per env and episode, 5 ms steps |
| `domain_rand.push_interval_s` | 15 | 0 = no pushes |
| `domain_rand.max_push_vel_xy` | 1.0 m/s | |
| `termination_if_roll/pitch_greater_than` | 30° | |
| `obs_cfg.add_noise`, `obs_cfg.noise_scales` | True; ang_vel 0.2, gravity 0.05, dof_pos 0.01, dof_vel 1.5 | |
| `obs_cfg.noise_level` | 1.0 (not in the config) | add `"noise_level": 0.5` to `obs_cfg` to halve every noise |
| `reward_scales.orientation` | −2.5 | |

- To turn one randomization off, delete its key from `domain_rand`. The env only randomizes what is listed. Without mass and CoM, batched link info is off too, so training runs faster.
- **Latency is capped at 20 ms.** The env holds the previous target for k physics steps of the current policy step, so k ∈ {0..4}, i.e. 0–20 ms. Larger values are clamped. Testing longer delays needs an action buffer across policy steps (not implemented).

**What to read**
- Nominal vs `--keep_dr` tables of the same checkpoint: the gap in `falls` and `lin_err` is what the randomization costs.
- Latency sweep: `lin_err` and `hf>10Hz` against latency.
- Viewer `--dr`: the recovery steps after each push.

**Experiments**
1. DR ablation: train with `domain_rand = {}` and with the default; evaluate both with `--keep_dr --measure_s 20`.
2. Push strength: `max_push_vel_xy` 0.5 / 1.0 / 1.5 → `falls` with `--keep_dr --measure_s 20`.
3. Latency: train with `latency_range_s` [0, 0] vs [0, 0.02]; evaluate at 0, 10 and 20 ms.
4. Friction: [0.5, 1.25] vs [0.3, 1.5] → slip columns with `--keep_dr`.
5. Gains: `kp` 20 / 30 (with `kd` 0.5 / 0.75). A different actuator for the policy: compare tracking and `tau>10Hz`, and keep the gains the robot will run.

**If it goes wrong**
- Training does not converge with DR: narrow the ranges (friction, latency, pushes first), then widen them again.
- Fine in nominal, bad with `--keep_dr`: find the culprit by retraining without one randomization at a time.

## 8. P1.3 safety and limits

**Commands**
```
python go2_train.py -e go2-p13
python go2_metrics.py -e go2-p13 --ckpt 999
python go2_metrics.py -e go2-p13 --ckpt 999 --keep_dr --measure_s 20 --tag dr
python go2_compare.py logs/go2-p12/metrics_ckpt999_seed0.csv logs/go2-p13/metrics_ckpt999_seed0.csv --cols tau_peak tau_sat jlim nonfoot power CoT lin_err
```

**Knobs**

| knob | where | default | effect |
|---|---|---|---|
| `penalize_contacts_on` | `env_cfg` | [thigh, calf, base] | link-name substrings counted by `collision` |
| `terminate_after_contacts_on` | `env_cfg` | [base] | contact ends the episode |
| `dof_vel_limits` | `env_cfg` | hip/thigh 30.1, calf 15.7 rad/s | |
| `soft_dof_pos_limit`, `soft_dof_vel_limit` | `reward_cfg` | 0.9, 0.9 | share of the range where the limit penalties start |
| `collision_force_threshold`, `termination_contact_force` | `reward_cfg` | 0.1 N, 1 N | |
| `only_positive_rewards` | `reward_cfg` | True | step reward clipped at 0 |
| `lin_vel_z`, `ang_vel_xy` | `reward_scales` | −2, −0.05 | base bouncing and rocking |
| `torques`, `dof_acc` | `reward_scales` | −1e-5, −2.5e-7 | effort and joint acceleration |
| `collision` | `reward_scales` | −1 | per touching link |
| `dof_pos_limits`, `dof_vel_limits` | `reward_scales` | −10, −1 | beyond the soft limits |
| `action_rate` | `reward_scales` | −0.01 | |
| `dof_vel` | `reward_scales` | absent | joint speed; legged_gym's weight is 0. Add it to try. |

**What to read**
- Actuation limits: `tau_sat` and `jlim_hits` should be 0, `tau_peak` well below 100.
- Gait: `nonfoot` 0.
- Energy: `power` and `CoT` against P1.2.
- TensorBoard: `Episode/rew_collision` and `rew_dof_pos_limits` going to 0; `Train/mean_episode_length` high.

**Experiments**
1. `torques` −1e-5 vs −2e-4 (unitree_rl_gym's Go2 value): `CoT`, `tau_rms` against `lin_err`.
2. `only_positive_rewards` False: watch `Train/mean_episode_length` early in training, when the penalties can make an early end attractive.
3. `collision` −1 vs −5.
4. `soft_dof_pos_limit` 0.9 vs 0.8.

## 9. P1.4 smooth actions

**Commands**
```
python go2_train.py -e go2-p14                                   # action_smoothness -0.1
python go2_train.py -e go2-p14-lcp --lcp 0.002                   # + LCP gradient penalty
python go2_train.py -e go2-p14-caps --caps-t 0.1 --caps-s 0.5    # + CAPS
python go2_metrics.py -e go2-p14 --ckpt 999                      # Smoothness table + spectra plot
python go2_compare.py logs/go2-p13/metrics_ckpt999_seed0.csv logs/go2-p14/metrics_ckpt999_seed0.csv logs/go2-p14-lcp/metrics_ckpt999_seed0.csv --cols lin_err act_rate act_acc hf tau_rms "tau>10Hz"
```

**Knobs**

| knob | where | default | effect |
|---|---|---|---|
| `action_smoothness` | `reward_scales` | −0.1 | second difference of the joint targets: targets jitter, spares the gait |
| `action_rate` | `reward_scales` | −0.01 | first difference |
| `--lcp` | flag | 0 | LCP gradient penalty (paper: 0.002) |
| `--caps-t`, `--caps-s` | flags | 0, 0 | CAPS temporal and spatial (paper, Ant: 0.1, 0.5) |
| `HF_CUTOFF_HZ` | `go2_metrics.py` | 10 | the gait / jitter boundary of the metrics |
| `SPECTRA_BINS` | `go2_metrics.py` | 4 commands | the commands drawn in the spectra plot |

**What to read**
- Smoothness table: `act_acc`, `hf>10Hz` and `tau>10Hz` should drop from P1.3 to P1.4 while `lin_err` stays put.
- Spectra plot: less power right of the 10 Hz line, same peaks left of it.
- TensorBoard (LCP / CAPS runs): `Loss/grad_penalty`, `Loss/caps_*` decreasing; `Perf/learning_time` up (1.4× for LCP, 2.1× for CAPS in a CPU smoke run).

**Experiments**
1. `action_smoothness` 0 / −0.1 / −0.3: where does tracking start to suffer?
2. LCP 0.002 vs none, at the same seed.
3. CAPS after LCP, if LCP is not enough.

**If it goes wrong**
- Sluggish tracking (`lin_err` up, `vx_meas` short): the smoothness weight is too high.
- Buzzing or warm motors on the robot while the sim tables look clean: the sim may miss the cause (motor model, latency). Check `tau>10Hz` with `--keep_dr` first.

## 10. P1.5 stand-up, sit-down, deployment

**Commands**
```
python go2_export.py -e go2-p15 --ckpt 999                 # logs/go2-p15/exported/{policy.pt, deploy_cfg.json}
python go2_fsm_sim.py                                      # no policy (zero actions): the state machine alone
python go2_fsm_sim.py -e go2-p15                           # stand up -> policy, stick forward -> sit down
python go2_fsm_sim.py -e go2-p15 --scenario estop          # damping (L1+Y) while standing
python go2_fsm_sim.py -e go2-p15 -n 16 --dr                # 16 robots, training randomization
python go2_fsm_sim.py -e go2-p15 --handover-s 0            # no gain ramp at the hand-over
python go2_fsm_sim.py -e go2-p15 --viewer
python go2_train.py -e go2-p15rec --recovery               # learned fall recovery (optional)
python go2_eval.py -e go2-p15rec --ckpt 999 --fallen       # every episode starts fallen, command 0 (P1.6 flag)
# robot side, in this order
python go2_deploy_sdk2.py --policy logs/go2-p15/exported --sim            # unitree_mujoco
python go2_deploy_sdk2.py --policy logs/go2-p15/exported --net <iface>    # robot on a stand, then on the ground
```

The `full` scenario: L1+R2 (stand up) at 1 s, L1+A (policy) at 4 s, left stick forward 6–8.5 s, D-pad down ×3 at 8.6–9.0 s (P1.6, ignored by 3-command policies), L1+R1 (sit down) at 10 s.

**What to read in the bench output**
- `[fsm t] ...` lines: the mode changes, in order.
- `observation check: max |deploy obs - Go2Env obs|`: about 1e-7 (float32 rounding). Around 1e-3 or more means the robot side builds a different observation (joint order, scales, gravity): fix that before going near the robot.
- Phases table: base height at the start and end of each mode, worst tilt, torque peak (% of limit), worst |target − joint|.
- Checks:
  - stand-up: n/n robots standing;
  - hand-over: base drop in the first 1.5 s of policy mode;
  - sit-down: sink speed;
  - e-stop: sink speed.
- `fsm_full.png` (or `fsm_estop.png`) next to the run: the same, over time.

**Knobs** (`FsmConfig` in `go2_fsm.py`, unless stated)

| knob | default | effect |
|---|---|---|
| `stand_kp`, `stand_kd` | (60, 80, 80), (5, 4, 4) | stand-up and sit-down gains, hip/thigh/calf |
| `passive_kd` | 3 | damping mode |
| `lie_s`, `stand_s`, `restand_s` | 1, 1, 1 s | stand-up segments |
| `ready_hold_s` | 0.5 s | L1+A accepted after this long standing |
| `handover_s` | 0.5 s | gain ramp into policy mode |
| `sit_s`, `sit_hold_s` | 2, 0.5 s | |
| `max_tilt_rad` | 1.0 | policy mode → damping beyond this tilt |
| `joint_limit_margin` | 0.2 rad | policy mode → damping this far outside a joint range |
| `height_step` | 0.01 m | P1.6 D-pad step |
| `combos` | L1+R2, L1+A, L1+R1, L1+Y | gamepad combinations |
| `LIE_POSE_LEG` | (0, 1.36, −2.65) | lying pose |
| `sticks_to_command(stick_deadzone=0.1)` | `go2_commands.py` | stick deadzone |
| `recovery.init_prob`, `spawn_height` | `env_cfg` | 0.15 with `--recovery`, 0.5 m |
| `rec_upright`, `rec_height`, `rec_pose` | `reward_scales` with `--recovery` | 1.0, 1.0, 0.5 |

**Experiments**
1. `--handover-s` 0 / 0.5 / 1.0 with your trained policy: base drop and tilt at the hand-over.
2. `-n 16 --dr`: does every robot stand up and hand over cleanly?
3. Recovery: `Episode/recovery_success` against the walking metrics of the same run (does learning to get up cost walking quality?).

## 11. P1.6 body height, keyboard, RNN

**Commands**
```
# training
python go2_train.py -e go2-p16h  --height-cmd
python go2_train.py -e go2-p16r  --rnn lstm
python go2_train.py -e go2-p16g  --rnn gru
python go2_train.py -e go2-p16hr --height-cmd --rnn lstm
python go2_train.py -e go2-p16r128 --rnn lstm --rnn-hidden 128      # smaller, faster memory
python go2_train.py -e go2-p16r96  --rnn lstm --steps-per-env 96    # longer backprop-through-time window
python go2_train.py -e go2-p16hr15 --height-cmd --rnn lstm --max-iterations 1500   # last checkpoint: 1499

# viewer
python go2_eval.py -e go2-p16h --ckpt 999 --cmd 0 0 0 -0.08          # lowest stance
python go2_eval.py -e go2-p16h --ckpt 999 --cmd 0 0 0 0.04           # tallest stance
python go2_eval.py -e go2-p16h --ckpt 999 --cmd 0.5 0 0 -0.05        # walking crouched
python go2_eval.py -e go2-p16h --ckpt 999 --cmd 0.5 0 0              # 3 values: dh = 0
python go2_eval.py -e go2-p16h --ckpt 999                            # random commands, heights included

# keyboard and video
python go2_teleop.py -e go2-p16h --ckpt 999                          # click the viewer window, then use the keys
python go2_teleop.py -e go2-p16h --ckpt 999 --dr                     # with training randomization and pushes
python go2_teleop.py -e go2-p16h --ckpt 999 --cmd 0.3 0 0 -0.03      # start from this command
python go2_teleop.py -e go2-p16h --ckpt 999 --record drive.mp4       # + video from a camera following the robot
python go2_teleop.py -e go2-p16h --ckpt 999 --headless --cmd 0.5 0 0 -0.04 --duration 10 --record crouch.mp4
PYOPENGL_PLATFORM=egl python go2_teleop.py ...                       # same, on Linux without a display

# metrics
python go2_metrics.py -e go2-p16h --ckpt 999                         # + height bins, z and z_err columns
python go2_metrics.py -e go2-p16r --ckpt 999 --keep_dr --measure_s 20 --tag dr
python go2_compare.py logs/go2-p15/metrics_ckpt999_seed0.csv logs/go2-p16h/metrics_ckpt999_seed0.csv logs/go2-p16r/metrics_ckpt999_seed0.csv

# deployment
python go2_export.py -e go2-p16hr --ckpt 999       # recurrent: "max |jit - policy| over 100 steps" in a row
python go2_fsm_sim.py -e go2-p16hr                 # D-pad down x3 at 8.6-9.0 s: "body height offset -3 cm"
```

**Teleop keys** (viewer window focused; the list is printed at start)

| keys | command |
|---|---|
| Up / Down | vx ±0.1 m/s |
| Left / Right | vy ±0.1 m/s (left = +) |
| Q / E | wz ±0.1 rad/s (Q = turn left) |
| T / G | dh ±1 cm (runs trained with `--height-cmd`) |
| Space | vx = vy = wz = 0 (keeps dh) |
| Backspace | reset the robot and the policy's memory |

- The terminal prints `set: ... -> policy gets: ...` at each change. The second part is after the training deadzone: one press of Up (0.1 m/s) shows `vx +0.00` there, two presses show `vx +0.20`.
- A key that would leave the training range prints `vx is at its training limit (+1.00)` and changes nothing.
- Genesis' own keys keep working: A C D F H I L O P V W Z F11 (I shows them in the window).
- Stop: close the window or Ctrl+C in the terminal. Both finish the video file and print the simulated time and the fall count.

**Knobs: body height**

| knob | where | default | effect |
|---|---|---|---|
| `--height-cmd` | flag | off | 4 commands instead of 3 |
| `body_height_range` | `command_cfg` | [−0.08, 0.04] m | sampled offsets. Reach (IK, feet at their default positions): front calf at its soft limit at +6.9 cm, at its URDF limit at +7.8 cm. Downward, −12 cm puts the calf at −2.2 rad, still inside its range (−2.72). |
| `obs_scales["body_height"]` | `obs_cfg` | 5.0 | dh × 5 in the observation (±0.4) |
| `base_height` | `reward_scales` | −50 | weight of the height error |
| `base_height_target` | `reward_cfg` | 0.30 m | dh is an offset from it |
| `similar_to_default` | `reward_scales` | −0.1 (×5 standing) | measured from the IK pose of the commanded height |
| `HEIGHT_GRID` | `go2_metrics.py` | dh −0.06, +0.03 at 0 and 0.5 m/s | extra metrics bins; keep them inside `body_height_range` |
| `height_step` | `FsmConfig` | 0.01 m | D-pad step on the robot |
| `GO2_THIGH_LENGTH`, `GO2_CALF_LENGTH` | `go2_env.py` | 0.213 m | geometry, not a knob |

**Knobs: RNN**

| knob | where | default | effect |
|---|---|---|---|
| `--rnn` | flag | off | `lstm` or `gru`, for actor and critic |
| `--rnn-hidden` | flag | 256 | size of the memory |
| `rnn_num_layers` | `go2_train.py`, `main()` | 1 | stacked RNN layers (edit the line) |
| `--steps-per-env` | flag | 48 with an RNN (24 MLP) | rollout length = the longest span a gradient reaches back: 48 steps = 0.96 s |
| `hidden_dims` | `train_cfg` actor / critic | [512, 256, 128] | MLP after the RNN |
| `num_mini_batches` | `train_cfg["algorithm"]` | 4 | with an RNN it splits the envs into groups of whole trajectories |

**Knobs: teleop** (`go2_teleop.py`)

| knob | default | effect |
|---|---|---|
| `LIN_STEP`, `ANG_STEP`, `HEIGHT_STEP` | 0.1, 0.1, 0.01 | change per key press |
| `CHASE_OFFSET` | (−1.6, −1.2, 0.9) m | camera position relative to the robot (viewer and video) |
| `FOLLOW_SMOOTHING` | 0.9 | camera low-pass per physics step (closer to 1 = smoother, more lag) |
| `--record_res` | 1280 720 | video size; 50 fps = one frame per policy step |
| `--latency_ms` | 20 | latency in the nominal world |
| `KeyboardCommand.keybinds()` | see keys | to remap, avoid Genesis' keys: A C D F H I L O P V W Z F11 (+ R S if imgui-bundle is installed) |

**What to read**
- Height:
  - metrics, `z_err` at the height bins (dh −6 and +3 cm) and at dh = 0;
  - `z[m]` should move by about dh between the bins of the same velocity;
  - the dh = 0 rows next to a P1.5 run (`go2_compare.py`): the extra command should not cost walking quality;
  - `tau_sat` and `jlim_hits` in the crouched and tall rows: they show when a height is too extreme;
  - TensorBoard: `Episode/rew_base_height` / −50 = mean squared height error.
- RNN:
  - `Train/mean_reward` against iterations and against training time (`Train/mean_reward/time`);
  - `Perf/learning_time`;
  - metrics in the nominal world and with `--keep_dr --tag dr`. The memory should help most under randomization, where it can infer friction, mass and latency from the recent history (a hypothesis to test);
  - latency sweep (0, 10, 20 ms).
- Teleop: the terminal line; the fall count at the end.

**Experiments**
1. Height range: [−0.08, 0.04] vs [−0.05, 0.05] (walk-these-ways' default) vs [−0.12, 0.04]. Keep `HEIGHT_GRID` inside the narrowest one to compare.
2. `base_height` −50 vs −20 (your pre-summer weight) with `--height-cmd`: height tracking against walking quality.
3. MLP vs LSTM vs GRU, same seed, 1000 and 1500 iterations, nominal and `--keep_dr`.
4. `--steps-per-env` 24 / 48 / 96 with the LSTM.
5. `--height-cmd` with `--recovery`: recovery episodes get a zero command, dh included.

**If it goes wrong**
- The robot ignores dh (`z_err` ≈ |dh|): check that `cfgs.pkl` has `num_commands` 4 (section 0). Then compare `Episode/rew_base_height` with the other terms.
- It falls when crouched and walking: the crouched gait is harder. Look at `tau_sat` and `jlim_hits` in those rows, train longer, or narrow the range.
- The RNN learns slower: compare at equal training time, give it more iterations, try `--rnn gru` or `--rnn-hidden 128`.
- Keys do nothing: click the viewer window first (keyboard focus). The terminal must show the key list at start.
- `--cmd: this run takes VX VY WZ (no DH...)`: the run was trained without `--height-cmd`.
- Empty or corrupt video: end the run by closing the window or with Ctrl+C, so the file is finished. A killed process loses it. On Linux without a display, add `PYOPENGL_PLATFORM=egl`.
- The viewer runs slower than real time while recording: the camera renders 50 frames per simulated second on top of the viewer. Here, with software OpenGL (no GPU), that was about 2 video frames per second of wall time. `--headless` is faster.

## 12. PPO and network knobs

`get_train_cfg()` in `go2_train.py`. These are rsl_rl's settings; the defaults come from the Genesis Go2 example.

| knob | default | effect |
|---|---|---|
| `learning_rate` | 1e-3 | starting value; the adaptive schedule then moves it |
| `schedule`, `desired_kl` | "adaptive", 0.01 | after each mini-batch: rate ÷ 1.5 when the KL divergence of the update exceeds 2 × `desired_kl`, × 1.5 when it is below half of it, within [1e-5, 1e-2]. "fixed" keeps it constant. |
| `num_learning_epochs`, `num_mini_batches` | 5, 4 | passes over the rollout, and mini-batches per pass. More mini-batches = smaller ones = less memory. |
| `clip_param` | 0.2 | PPO ratio clip |
| `entropy_coef` | 0.01 | exploration bonus. Higher keeps `Policy/mean_std` up longer. |
| `gamma`, `lam` | 0.99, 0.95 | discount (horizon ≈ 1 / (1 − γ) = 100 steps = 2 s) and GAE |
| `value_loss_coef`, `use_clipped_value_loss`, `max_grad_norm` | 1.0, True, 1.0 | |
| `actor` / `critic` `hidden_dims`, `activation` | [512, 256, 128], elu | |
| `distribution_cfg.init_std`, `std_type` | 1.0, "scalar" | initial action noise; "log" learns log(std) instead |
| `num_steps_per_env` | 24 (48 with `--rnn`) | rollout length per env and iteration. Rollout size = `-b` × this. |
| `save_interval` | 100 | checkpoint every N iterations |
| `-b` (`--num-envs`) | 4096 | parallel envs: throughput and GPU memory |

## 13. Recipes

**Add a reward term**
1. In `go2_env.py`, write `_reward_<name>(self)`, returning one value per env (shape [num_envs]). Use the buffers updated in `step()`: `base_lin_vel`, `dof_pos`, `torques`, `feet_contact`, `link_contact_force`, ...
2. In `go2_train.py`, add `"<name>": <weight>` to `reward_scales`. Negative for a penalty.
3. It is logged as `Episode/rew_<name>`. Size the weight as in section 4.

**Add a metric**
1. In `go2_metrics.py`, write `m_<name>(rec, dt)` returning `{"<name>[unit]": tensor [N]}`. `rec` holds the recordings ([T, N, ...] tensors, listed in `run_rollout()`), `rec["valid"]` masks the time after a fall. `masked_mean()` does the averaging.
2. Add the function to a table in `TABLES`. The CSV and `go2_compare.py` pick it up.
3. If it needs a signal that is not recorded, add it to `log` in `run_rollout()`.

**Read the raw recordings** (the `.npz`)
```
import numpy as np
d = np.load("logs/go2-p16h/metrics_ckpt999_seed0.npz")
grid, bin_id = d["grid"], d["bin_id"]                 # commands of the bins; bin of each env
env = int(np.flatnonzero(bin_id == 11)[0])            # first env of bin 11 (dh -0.06 at rest, 4-command runs)
z = d["base_pos"][:, env, 2]                          # its base height over time [T]
print(grid[11], z.mean(), z.std())
```
Keys: `cmd, lin_vel, ang_vel, valid, q_target, dof_pos, dof_vel, torque, torque_sub, base_pos, foot_pos, foot_vel, foot_ang, foot_force, other_force, fall_step, q_lo, q_hi, tau_lim, mass, z_target, bin_id, grid`.

**Add a command** (as P1.6 did for the height)
- `grep -n "\[P1.6\]" *.py` lists every place the height command touches (PowerShell: `Select-String -Pattern "\[P1.6\]" -Path *.py`).
- Those places are the checklist for any new command:
  - env: scales and limits, observation noise size, sampling, the reward that uses it;
  - training config and flag;
  - eval `--cmd`, metrics grid and columns;
  - export (`deploy_cfg.json`);
  - robot side: `go2_fsm.py` builds the command;
  - teleop keys.

**Fine-tune from a checkpoint** (tested here on a smoke run)
- In `go2_train.py`, just before `runner.learn(...)`, add:
  ```
  runner.load("logs/go2-p15/model_999.pt")   # another run's folder: go2_train.py deletes logs/<exp_name>/
  ```
- Train under a new `-e`. The network must have the same shape (same number of commands, same `--rnn`, same `hidden_dims`), or loading fails with a size mismatch.
- The optimizer state and the iteration counter come along: `--max-iterations 500` from `model_999.pt` runs iterations 999 to 1498 and ends with `model_1498.pt`.
- Remove the line afterwards.

**Make a video of any run**
- `python go2_teleop.py -e <run> --ckpt 999 --headless --cmd 0.5 0 0 --duration 10 --record walk.mp4`. Works for 3- and 4-command runs, MLP or RNN.
- 1280×720 at 50 fps by default; `--record_res 1920 1080` for more.

**Evaluate an old run in today's world**
- `go2_metrics.py` takes the run's config from its `cfgs.pkl`, but the physics (200 Hz, explicit PD, armature, limits) from the code. So a baseline or P1.1 run is measured in the same world as a P1.6 run.

## 14. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `Missing .../model_1000.pt` | the last checkpoint of 1000 iterations is `model_999.pt` | `--ckpt 999` |
| `size mismatch` (46 vs 45) loading a checkpoint | a model file loaded with another run's config (other command count) | load each checkpoint with its own run (`-e <run>` reads that run's `cfgs.pkl`) |
| my previous run disappeared | `go2_train.py` deletes `logs/<name>/` | a new `-e` per run |
| my nominal metrics CSV was overwritten | a second evaluation of the same checkpoint and seed | `--tag` on the second one |
| CUDA out of memory | rollout too large: `-b`, or an RNN (48 steps per env) | `-b 2048`, or raise `num_mini_batches` (smaller mini-batches) |
| `error: --lcp / --caps-* are implemented for MLP policies only` | `--rnn` with a P1.4 regularizer | use one or the other (CHANGES.md, P1.6) |
| `exported network does not match the training policy` | rsl_rl version change, or a modified model | re-export; check the versions saved in `deploy_cfg.json` (`source`) |
| latency sweep flat above 20 ms | values above 20 ms are clamped | section 7 |
| `[WARNING] Neutral robot position (qpos0) exceeds joint limits.` | the URDF's zero pose puts the calves at 0 rad, outside their range | harmless: the env sets the default pose at every reset |
| `FutureWarning: torch.jit.script is deprecated` | a deprecation notice of recent torch versions, printed at import | harmless |
| `Viewer closed.` traceback when you close the window of `go2_eval.py` | its loop runs until the viewer is gone (as in the Genesis example) | harmless; `go2_teleop.py` exits cleanly |
| CPU smoke test slow | CPU backend | `-b 32` to `-b 64`, `--max-iterations 3` |
| black viewer (remote session, WSL) | no OpenGL display | `go2_teleop.py --headless --record`, or metrics only |
| the same command gives different numbers | different `--seed`, `--reps`, world options or code | the checksum line tells whether the simulation changed |

## 15. Experiment log template

One row per evaluated checkpoint. Read the numbers off `go2_compare.py`.

| date | run (`-e`) | snapshot | change vs baseline | seed | ckpt | falls nominal / DR | lin_err @ 0.5 | z_err @ dh −6 | act_acc @ 0.5 | tau>10Hz @ 0.5 | CoT @ 0.5 | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | go2-p13 | p1_6 | baseline | 1 | 999 | | | | | | | |
| | go2-p13-coll5 | p1_6 | `collision` −5 | 1 | 999 | | | | | | | |
