# P2 · Gait quality: test & tinker guide (version 2, on the [S2R] P1)

Companion to the P2 code. This version sits on top of your reworked P1: the `[S2R]` sim-to-sim changes (joint losses,
MuJoCo-like contacts and lower legs, single-axis commands) and the `[fix]` changes (tracking 1.5 / 0.75, penalty
curriculum, air-time target 0.3 s, 2000 iterations). Part of the first P2 version is now in P1, so P2 adds less code
and its sections turn from "add this" into "check and tune this".

One section per part (P2.0 to P2.5), each with: what the code does and why, the commands, what to read in
`go2_metrics.py` (Genesis) and `go2_analyze.py` (Genesis bench / unitree_mujoco / robot), the knobs, symptoms and
fixes, and a "done when" check. Reference tables are at the end.

Contents
1. Files and install
2. Change map: what P2 adds to the new P1, per part
3. Workflow and conventions (run names, seeds, evaluation, comparing, knobs, TensorBoard, sim-to-sim loop)
4. P2.0 The sim-to-sim baseline (now in P1)
5. P2.1 Longer steps (air-time target and cap)
6. P2.2 Foot clearance (now in P1)
7. P2.3 Symmetry
8. P2.4 Gait pattern (only if needed)
9. P2.5 Energy
10. Final P2 run, report and sim-to-sim gate
11. Reference: configuration keys and constants
12. Reference: metrics glossary
13. Troubleshooting
14. Sources

---

## 1. Files and install

| File | Status vs your new P1 | Part |
|---|---|---|
| `go2_env.py` | modified | P2.1 air-time cap; P2.4 contact timers, Spot gait terms, WTW clock and schedule terms; P2.5 motor power and energy term |
| `go2_train.py` | modified | flags `--air-time --air-time-max --clearance --symmetry --gait --energy --set`, P2 constants, summary printout |
| `go2_symmetry.py` | **new** | P2.3 left-right mirror for rsl_rl + physics self-check |
| `go2_metrics.py` | modified | air_t, stride, diag_sync, air_cv, sym_err, air_asym, mirror-pair check, 200 Hz power for CoT |
| `go2_compare.py` | modified | `--preset gait` (also on `go2_analyze.py --csv` files) |
| `go2_export.py` | modified | gait-clock parameters and the joint model in `deploy_cfg.json` |
| `go2_fsm.py` | modified | `Policy` runs the gait clock of `--gait wtw` exports |
| `go2_fsm_sim.py` | modified | observation check with the clock (formula and timing) |
| `go2_analyze.py` | modified | per-phase air_t, diag_sync, air_cv, air_asym (estimated and true contacts), sched_match |
| `go2_deploy_sdk2.py` | modified | prints the gait clock in the start-up alignment block |
| `go2_sim_suite.py` | modified | gait column (air_t / diag_sync per moving segment) in `summary.md` / `summary.csv` |
| `test_go2_deploy.py` | modified | gait-clock and gait-column checks |
| `go2_feet_plot.py` | **new** | foot height / speed / contact plot from a `go2_metrics.py` `.npz` |
| `go2_commands.py`, `go2_eval.py`, `go2_smooth_ppo.py`, `go2_teleop.py`, `go2_inputs.py`, `go2_kinematics.py`, `go2_log.py`, `go2_mujoco_headless.py`, `go2_probe.py` | unchanged | |

- Put `go2_symmetry.py` next to the other scripts. rsl_rl imports it by name every time it builds PPO for a run trained
  with `--symmetry`, so `go2_eval.py`, `go2_metrics.py`, `go2_export.py` and `go2_teleop.py` need it too.
- Robot computer: copy the new `go2_fsm.py` (it only matters for `--gait wtw` exports, harmless otherwise), with
  `go2_commands.py`, `go2_inputs.py` and the exported folder as before.
- Without P2 flags, a run is exactly your P1 configuration: the P2 code adds buffers (timers, clock, power) but no
  reward term and no observation entry unless a flag asks for it.
- Old runs (P1 and first-version P2) still load and evaluate.
- All P2 markers are `[P2.0]` ... `[P2.5]` or `[P2]`: `grep -n "\[P2" *.py` lists every change.

Self-checks you can run any time (CPU):

```bash
python go2_symmetry.py            # mirror twice = identity; mirrored Genesis rollout stays mirrored
python go2_symmetry.py --clock    # same, with the P2.4 gait clock in the observation
python test_go2_deploy.py         # state machine, inputs, kinematics, log format, gait clock, gait columns
```

---

## 2. Change map: what P2 adds to the new P1, per part

### P2.0 Joint friction and damping → done in P1 `[S2R]`
Your P1 now has the joint losses (`GO2_JOINT_FRICTION` 0.2 Nm, `GO2_JOINT_DAMPING` 0.1 Nm·s/rad, `set_joint_losses()`,
optional per-env ranges), plus more than the first P2 version had: torsional and rolling foot friction, the elliptic
friction cone with impratio 100, the MJCF lower-leg mass, the contact friction range [0.4, 1.25] and single-axis
commands. **P2 adds no code here.** Section 4 turns it into the reference for every P2 comparison, in Genesis and in
unitree_mujoco.

### P2.1 Longer steps
- P1 `[fix]` already sets `feet_air_time_target` to 0.3 s (first P2 version: `--air-time`).
- P2 adds the cap `feet_air_time_max` in `_reward_feet_air_time()` and two flags: `--air-time T` (target, for sweeps)
  and `--air-time-max C` (cap). **Why the cap matters now:** without it the term always prefers fewer, longer steps
  (5.1); with it, the preferred swing duration is C.
- `go2_metrics.py`: `air_t[s]`, `stride[cm]`. `go2_analyze.py`: `air_t[s]` per phase, in every world.

### P2.2 Foot clearance → reward in P1 `[S2R]`
- P1 has `_reward_foot_clearance()` (the Spot kernel from the first P2 version) on by default: weight 1.0, std 0.02,
  target 0.075 m.
- P2 keeps your values. `--clearance H` now only changes the target (it no longer sets the weight); the ablation is
  `--set reward.reward_scales.foot_clearance=None`.
- `go2_feet_plot.py` (new): swing profile and dragged toes.

### P2.3 Symmetry
- `go2_symmetry.py` (new): `compute_symmetric_states(env, obs, actions)` for rsl_rl 5.5.1's `Symmetry` extension,
  `mirror_obs()`, `mirror_actions()`, a physics self-check (now against your MuJoCo-like contact model).
- `go2_train.py`: `--symmetry [aug|loss|both|log]`; refused with `--rnn` (rsl_rl raises for recurrent policies).
- `go2_metrics.py`: `sym_err[rad]`, `air_asym[%]`, mirror-pair check. `go2_analyze.py`: `air_asym[%]`.

### P2.4 Gait pattern (if needed)
- `go2_env.py`: contact timers (`feet_contact_time`, `feet_last_air_time`, `feet_last_contact_time`, Isaac Lab
  ContactSensor semantics); option A `_reward_gait()` + `_reward_air_time_variance()` (Spot); option B the WTW clock
  (`gait_targets()`, `gait_phase`, `clock_obs()`: 8 observation entries appended at the end) and
  `_reward_contact_schedule_force()` / `_vel()`.
- `go2_train.py`: `--gait spot` / `--gait wtw`. **New:** `--gait wtw` removes `feet_air_time` (P1's 0.3 s target is
  longer than the clock's 0.25 s swing, and WTW has no air-time term).
- Deploy side: `go2_export.py` writes the clock, `go2_fsm.Policy` runs it, `go2_fsm_sim.py` checks it,
  `go2_deploy_sdk2.py` prints it.
- `go2_metrics.py`: `diag_sync[%]`, `air_cv[%]`. `go2_analyze.py`: the same per phase, plus `sched_match[%]`
  (do the feet follow the clock, in unitree_mujoco too?).

### P2.5 Energy
- `go2_env.py`: `self.mech_power` = Σ_joints |τ·q̇| averaged over the 4 physics steps of each policy step (W);
  `_reward_energy()` returns it.
- `go2_train.py`: `--energy [W]` adds `energy: −W` (1.5e-3 when given without a value).
- `go2_metrics.py`: `power[W]` and `CoT[-]` use that 200 Hz average. `go2_analyze.py` already had
  `power[W]` = Σ|τ_est·q̇| from the logs, the same quantity.

### Weights rescaled with P1's tracking
P1 `[fix]` raised tracking from 1.0 + 0.5 to 1.5 + 0.75. The P2 weights keep their strength relative to tracking,
so they are 1.5× those of the first P2 version: Spot `gait` 1.0 → 1.5, WTW schedule terms −2 → −3, `--energy`
1e-3 → 1.5e-3. `air_time_variance` stays −1.0, which is now exactly unitree_rl_lab's Go2 value next to the same
tracking weights.

---

## 3. Workflow and conventions

### 3.1 Base command
Keep the P1 options you settled on in one variable, so every P2 run differs only by its P2 flags:

```bash
BASE="--seed 1"                    # + the P1 options you kept, e.g. --height-cmd
                                   #   (not --rnn: rsl_rl's symmetry does not support recurrent policies)
CKPT=1999                          # P1 now trains 2000 iterations: the last checkpoint is model_1999.pt
UM=~/unitree_mujoco                # your unitree_mujoco checkout
```

Run names used below: `go2-p20` (baseline), `go2-p21`, `go2-p22`, ... Sweep runs get a suffix (`go2-p21-c0.4`),
second seeds get `-s2` and `--seed 2`.

### 3.2 Seeds
`go2_compare.py`'s `~` means "within 2 standard errors over the evaluation envs". It says nothing about training
seeds, which usually vary more. Before you write a conclusion down, train the two configurations you compare with
2 seeds each (3 for the thesis) and check the difference holds for both.

### 3.3 Evaluation in Genesis
```bash
python go2_metrics.py -e go2-p21 --ckpt $CKPT                        # nominal world (no DR, no noise, 20 ms latency)
python go2_metrics.py -e go2-p21 --ckpt $CKPT --keep_dr --tag dr     # with the training randomization (robustness)
python go2_metrics.py -e go2-p21 --ckpt $CKPT --seed 1 --tag e1      # another evaluation seed
```
The evaluation uses the run's own world (joint losses, contact model and lower legs from its `cfgs.pkl`).
**Re-run the baseline with this `go2_metrics.py`**: power and CoT are measured at 200 Hz and new columns exist.
Never compare CSVs written by different versions of the script.

### 3.4 Comparing
```bash
python go2_compare.py logs/go2-p20/metrics_ckpt1999_seed0.csv logs/go2-p21/metrics_ckpt1999_seed0.csv --preset gait
python go2_compare.py A.csv B.csv C.csv --cols air_t stride lin_err CoT --rows "0.25 0 0" "0.5 0 0" "1 0 0"
```
`--preset gait` shows: lin_err, yaw_err, air_t, stride, step_freq, duty, swing_h, slip_v, sym_err, air_asym,
diag_sync, air_cv, sched_match, power, CoT (the columns a file has: `go2_analyze.py` CSVs have no stride, sym_err or
slip_v, and add the `_true` variants and sched_match).
The first file is the reference; `~` marks a change within noise.

### 3.5 Where the knobs are
1. **Flags** of `go2_train.py`: `--air-time T`, `--air-time-max C`, `--clearance H` (values required: P1 already sets
   the defaults), `--symmetry [mode]`, `--gait spot|wtw`, `--energy [W]` (recommended value when given alone).
2. **`--set GROUP.KEY=VALUE`** changes any config entry, after the flags (groups `env`, `obs`, `reward`, `command`,
   `train`). Values are Python literals; `None` on a reward scale removes the term, `None` on an env key switches an
   `[S2R]` term off:
   ```bash
   --set reward.reward_scales.foot_clearance=0.5 reward.foot_clearance_std=0.05
   --set reward.feet_air_time_max=0.4 env.joint_friction=0.3 env.domain_rand.friction_range=[0.5,1.25]
   --set reward.reward_scales.feet_air_time=None          # remove a term
   --set reward.penalty_scale_init=1.0                    # no penalty curriculum
   --set train.algorithm.symmetry_cfg.mirror_loss_coeff=0.3     # together with --symmetry loss (or both)
   ```
   Every value ends up in `cfgs.pkl`, so the run stays reproducible. The start of the training log prints a
   `--- gait quality (P2) ---` block with what is on, including the `[S2R]` world and the penalty curriculum: check it
   before leaving a run overnight.
3. **Constants** at the top of `go2_train.py` (`MIRROR_LOSS_COEF`, `ENERGY_WEIGHT`, `SPOT_GAIT`, `WTW_GAIT`,
   `WTW_CLOCK`) hold the values the bare flags use.

Two P1 mechanisms change how new terms behave:
- **Penalty curriculum** (`penalty_scale_init` 0.3, `penalty_ramp_steps` = 400 iterations): every negative-weight term
  is multiplied by k, rising from 0.3 to 1 over the first 400 iterations. That includes the P2 penalties
  (`air_time_variance`, `contact_schedule_*`, `energy`). Positive terms (`gait`, `foot_clearance`, `feet_air_time`)
  are not ramped.
- **`only_positive_rewards`**: the step reward is clipped at 0. A strong new penalty can make every step 0 (no
  signal); the curriculum softens this at the start.

### 3.6 TensorBoard
```bash
tensorboard --logdir logs
```
| Tag | Meaning |
|---|---|
| `Episode/rew_<term>` | per-term reward, already weighted: `weight × mean(term)` per second of episode. Penalties include the curriculum factor k (0.3 → 1 over 400 iterations). Early-terminated episodes read low. |
| `Train/mean_reward`, `Train/mean_episode_length` | overall progress; a flat 0 reward after adding a penalty = `only_positive_rewards` clipping everything (13) |
| `Loss/symmetry` | MSE between π(mirror(s)) and mirror(π(s)) in action units; logged whenever `--symmetry` is set (also `log`) |
| `Policy/mean_std` | action noise; should keep falling |
| `Perf/learning_time` | rises with `--symmetry aug/both` (mini-batches are twice as big) |

Conversions (after iteration 400, when k = 1): training-time motor power ≈ `−Episode/rew_energy / W`;
`sym_err` ≈ `0.25·√(Loss/symmetry)` (0.25 = `action_scale`; training batches vs evaluation states, so roughly).

### 3.7 Visual checks
```bash
python go2_teleop.py -e go2-p22 --ckpt $CKPT                                     # drive it (viewer)
PYOPENGL_PLATFORM=egl python go2_teleop.py -e go2-p22 --ckpt $CKPT --headless --cmd 0.5 0 0 --duration 10 \
    --record p22_0.5.mp4                                                       # video, no display
python go2_feet_plot.py logs/go2-p22/metrics_ckpt1999_seed0.npz --cmd 0.5 0 0 --target 0.075
python go2_mujoco_headless.py --um $UM --dt 0.002 --video p22_mj.mp4 --exit-idle 3 &    # unitree_mujoco video
python go2_deploy_sdk2.py --policy logs/go2-p22/exported --sim --scenario walk --profile fwd_back --yes
```
`gait_*.png` (from `go2_metrics.py`) shows the contact sequence per command. `go2_feet_plot.py` shows, per foot, the
clearance curve, the horizontal speed of the contact point (dashed, the quantity `slip_v` averages in stance) and the
stance (shaded, > 3 N). A dragged toe shows either as low and fast outside the shaded stance (light contact) or as
speed inside it (slip).

### 3.8 The sim-to-sim loop (one policy, two worlds, one table)
Every P2 part ends with the same check: does the gait change seen in Genesis survive in unitree_mujoco? The loop runs
the same scenario through the deploy path in both worlds, then reads both logs with the same estimators:

```bash
EXP=go2-p21; PROF=fwd_back                       # profiles: slow fwd_back lateral yaw stand60 (go2_inputs.PROFILES)
python go2_export.py -e $EXP --ckpt $CKPT
python test_go2_deploy.py --policy logs/$EXP/exported
# Genesis bench: the run's own world (joint losses, contact model), foot friction 1.0
python go2_fsm_sim.py -e $EXP --scenario walk --profile $PROF --log logs/deploy/genesis_${EXP}_$PROF.npz
# unitree_mujoco through go2_deploy_sdk2.py, foot friction 1.0 (the suite starts and stops the simulator)
python go2_sim_suite.py --policy logs/$EXP/exported --um $UM --only ${PROF}_mu1 --repeat 1 --out out/$EXP
MJ=$(ls -t logs/deploy/mujoco_*_suite_${PROF}_mu1_0.npz | head -1)
python go2_analyze.py logs/deploy/genesis_${EXP}_$PROF.npz $MJ logs/deploy/mjtruth_suite_${PROF}_mu1_0.npz \
    --csv out/$EXP --plots out/$EXP
python go2_compare.py out/$EXP/metrics_genesis_${EXP}_$PROF.csv out/$EXP/metrics_$(basename $MJ .npz).csv \
    --cols vx_meas lin_err air_t diag_sync air_cv air_asym swing_h step_freq duty power CoT
```
In the `go2_analyze.py` tables each profile segment is one table (A = Genesis bench, B = unitree_mujoco); in the
`go2_compare.py` output each segment's command is one row. Read the P2 columns of the part you just changed.

Which metric exists where:

| Metric | `go2_metrics.py` (Genesis, ground truth) | `go2_analyze.py` (bench, unitree_mujoco, robot) |
|---|---|---|
| `lin_err`, `vx_meas` | true velocity | true velocity in simulators, leg odometry on the robot (`vx_odo` everywhere) |
| `air_t`, `diag_sync`, `air_cv`, `air_asym` | contact = foot force > 3 N | contact estimated from joint torques (25 N on, 12 N off, 40 ms debounce); `_true` = simulator foot forces through the same thresholds |
| `swing_h` | foot-sphere height | leg kinematics, relative to the stance feet |
| `stride`, `sym_err`, `slip_*` | yes | no |
| `sched_match` (P2.4 B) | no | share of time the stance (estimated, and `_true`) equals the clock's |
| `power`, `CoT` | Σ\|τ·q̇\| at 200 Hz | Σ\|τ_est·q̇\| at the log rate |

Compare `go2_analyze.py` numbers between worlds (same estimator, its bias cancels), not with `go2_metrics.py`'s
absolute values. The bias is large for swing timing. Measured on the Genesis bench with an open-loop trot driven by a
2 Hz clock (feet lifted by thigh and calf, robot in place, so contacts are clean):

| column | true contacts (`_true`) | estimated from torques |
|---|---|---|
| `duty` | 0.58 | 0.74 |
| `air_t` | 0.21 s | 0.13 s |
| `step_freq` | 2.00 Hz | 2.00 Hz |
| `diag_sync` | 93 % | 93 % |
| `air_cv` | 4–8 % | 4–8 % |
| `sched_match` | 90 % | 76 % |

So frequency, synchrony and regularity survive the estimate; durations and duty do not (the torque estimate holds a
foot in stance while it unloads and lifts). In simulators read the `_true` columns for durations. On the robot only
the estimated ones exist: compare them with the estimated ones of the bench and unitree_mujoco, which is the
sim-to-real link.

---

## 4. P2.0 The sim-to-sim baseline (now in P1)

### 4.1 What P1 now contains, and why it matters for P2
- Joint friction 0.2 Nm and damping 0.1 Nm·s/rad (go2.xml), torsional 0.02 m and rolling 0.01 m foot friction, the
  elliptic cone with impratio 100, the go2.xml lower legs, contact friction [0.4, 1.25]. Your note in `go2_train.py`
  is the key result: together they make the Genesis bench reproduce unitree_mujoco for go2-p17h (slow profile
  +0.11 / −0.10 vs +0.12 / −0.10 m/s, step frequency 0.83 Hz in both, against 1.22 Hz in plain Genesis).
- For P2 this means two things. Gait metrics measured in Genesis now transfer to unitree_mujoco, so a P2 gain can be
  checked in both worlds. And the contact model changes the gait itself: the same policy (p17h, trained with
  Genesis' soft contacts) steps at 1.22 Hz in plain Genesis and at 0.83 Hz with the MuJoCo-like contacts.
  **Re-measure the gait on a baseline trained in the `[S2R]` world before tuning P2.1**: gait numbers from plain
  Genesis (p16, the first P2 version) no longer apply.
- Also new in P1 and relevant here: single-axis and small commands (better slow and lateral bins), tracking 1.5 / 0.75,
  the penalty curriculum, 2000 iterations.

### 4.2 Commands
```bash
# 1) The baseline: your P1 configuration, no P2 flag (2 seeds)
python go2_train.py -e go2-p20 $BASE
python go2_train.py -e go2-p20-s2 $BASE --seed 2
python go2_metrics.py -e go2-p20 --ckpt $CKPT
python go2_metrics.py -e go2-p20-s2 --ckpt $CKPT

# 2) The same in unitree_mujoco: the loop of 3.8 with PROF=fwd_back, then lateral and yaw
EXP=go2-p20; PROF=fwd_back   # ... 3.8

# 3) The H3 gate of the baseline (your suite)
python go2_sim_suite.py --policy logs/go2-p20/exported --um $UM --suite gate
```
If you already trained this configuration under another name, use that run instead of `go2-p20`.

### 4.3 What to read
- `go2_metrics.py`, Gait table: `air_t`, `step_freq`, `duty`, `swing_h` at 0.25 to 1 m/s. These are the numbers P2.1
  and P2.2 start from. Write them in the results log (10).
- `go2_analyze.py` (Genesis bench vs unitree_mujoco, fwd_back segments 1 and 3): `vx_meas` within a few cm/s, and
  `step_freq`, `air_t`, `diag_sync`, `swing_h` close between A and B. That is the condition for using Genesis numbers
  to predict MuJoCo behaviour in P2.1 to P2.5. Where they differ, the world gap is bigger than P2's effect: fix the
  world first (P1 `[S2R]`), not the gait.
- The gate: what passed for p17h should pass here.

### 4.4 Knobs (P1, listed for completeness)
| Knob | How | Default |
|---|---|---|
| joint losses | `--set env.joint_friction=0.3 env.joint_damping=0.2` | 0.2 Nm, 0.1 Nm·s/rad |
| joint-loss DR | `--set env.domain_rand.joint_friction_range=[0.05,0.4] env.domain_rand.joint_damping_range=[0.02,0.2]` | off |
| foot spin / rolling | `--set env.foot_torsional_friction=None env.foot_rolling_friction=None` | 0.02 / 0.01 m |
| cone, impratio | `--set env.friction_cone=pyramidal env.impratio=None` | elliptic, 100 |
| lower legs | `--set env.lower_leg_inertial=None` | go2.xml values |
| contact μ range | `--set env.domain_rand.friction_range=[0.5,1.25]` | [0.4, 1.25] |

### 4.5 Done when
`go2-p20` passes your gate, and the bench and unitree_mujoco agree on the gait columns at fwd_back (within what
you accept as a world gap). **`go2-p20` is the reference for every P2 comparison below.** Optional, free
monitoring: add `--symmetry log` to `go2-p20` to get a `Loss/symmetry` curve for the baseline (training unchanged).

---

## 5. P2.1 Longer steps

### 5.1 What the air-time term actually does
The legged_gym term pays, at each touchdown, `w·dt·(t_air − T)`. Summed over a second it is

    w·dt·[ (total airborne time of the 4 feet) − T × (number of touchdowns) ]

so **T is the price of one step, in seconds of swing**: a step shorter than T costs reward, a longer one earns.

At a fixed duty factor D and stride frequency f (swing s = (1 − D)/f), per second of walking:

    R = 4·w·dt·[ (1 − D) − T·f ]

R grows when f falls and when D falls. **Uncapped, the term always prefers fewer, longer steps and less ground
contact**; T only sets how expensive a step is. Kinematic reach, balance and tracking stop it somewhere, which is
where the gait ends up, not at T. So if your baseline's `air_t` sits well above 0.3–0.4 s at walking speeds, or slow
commands are tracked poorly with long, slow steps, the uncapped term is a likely cause.

With a cap C (`feet_air_time_max`), a touchdown pays at most `C − T`, and

    R = 4·w·dt·f·[ min(s, C) − T ]

rises with s up to s = C and falls after it (fewer steps, each paid no more). **The cap is the swing duration the term
prefers.** Isaac Lab's Spot air-time reward plays the same role with `mode_time` = 0.3 s (it pays air and contact
phases up to that duration, nothing beyond).

What the Go2 can reach: in a trot with duty D and swing time s at speed v, a foot sweeps `v·D·s/(1−D)` during stance.
The comfortable sweep is S ≈ 0.3 m (±15 cm around the hip, 0.213 m thigh and calf at 0.3 m base height). At D = 0.5
the longest swing is `s = S/v`: 0.6 s at 0.5 m/s, 0.3 s at 1 m/s. That is why T = 0.3 s (P1 `[fix]`) breaks even for
a 50 % trot at the top speed.

Reference swing durations: Unitree's model-based trot for this robot class has a 0.45 s period with 50 % stance,
i.e. 0.225 s swings (unitree_guide); walk-these-ways' Go1 default is 3 Hz (0.167 s swings) and samples 2–4 Hz;
Froude scaling of legged_gym's ANYmal value gives 0.39 s. So a cap between 0.3 and 0.45 s covers the useful range.

### 5.2 Commands
```bash
# 1) Read the baseline's swings (go2-p20 Gait table: air_t, step_freq, duty, at 0.25-1 m/s)
python go2_metrics.py -e go2-p20 --ckpt $CKPT

# 2) Sweep the cap (1 seed each), then repeat the best with a second seed
for C in 0.35 0.45 0.6; do python go2_train.py -e go2-p21-c$C $BASE --air-time-max $C; done
for C in 0.35 0.45 0.6; do python go2_metrics.py -e go2-p21-c$C --ckpt $CKPT; done
python go2_compare.py logs/go2-p20/metrics_ckpt1999_seed0.csv logs/go2-p21-c{0.35,0.45,0.6}/metrics_ckpt1999_seed0.csv \
    --cols air_t stride step_freq duty swing_h lin_err CoT slip_v --rows "0.25 0 0" "0.5 0 0" "1 0 0" "0 0.3 0" "0 0 0.5"

# 3) Optional: the target with the cap you kept (T moves the price of a step, C the preferred duration)
for T in 0.2 0.25; do python go2_train.py -e go2-p21-c0.45-t$T $BASE --air-time-max 0.45 --air-time $T; done

# 4) Sim-to-sim (3.8) for the run you keep, PROF=slow and fwd_back
```
Then keep the chosen run as `go2-p21`. If the baseline's `air_t` is already where you want it and tracking at slow
speed is fine, P2.1 is done without a change: write that down with the numbers.

### 5.3 What to read
| Metric | Expect with a cap C | Bad sign |
|---|---|---|
| `air_t[s]` (Gait) | moves toward C at 0.25–0.75 m/s | stays far above C: other terms drive the slow gait (check `similar_to_default`, clearance) |
| `step_freq[Hz]` | rises when C < the baseline's `air_t` | |
| `duty[-]` | 0.5–0.65 at walking speeds | > 0.7 with short `air_t` = shuffling; < 0.4 = flight phases |
| `lin_err`, `vx_meas` at 0.25 m/s | better or `~` (slow tracking was p17h's weak point: 0.11 m/s for 0.25 on the slow profile) | |
| `lin_err` at 1.0 m/s | `~` | worse: C or T too high for the top speed |
| `CoT[-]` | `~` (more steps, each cheaper) | up a lot: stepping got too fast |
| `slip_v`, `swing_h` | `~` | `swing_h` down: shorter swings leave less time to lift (P2.2 then needs more weight) |

Consistency checks for a periodic gait at straight walking:
`stride[cm] ≈ 100·vx_meas/step_freq` and `air_t ≈ (1 − duty)/step_freq`. If they disagree a lot, the gait is
irregular (`air_cv`, 12) or the bin mixes several behaviours.
`air_t` uses a 3 N contact threshold, the reward 1 N: the reward sees swings up to 1–2 policy steps (0.02–0.04 s)
shorter.
TensorBoard: `Episode/rew_feet_air_time` should end positive (on average, steps longer than T).
Sim-to-sim: `air_t` and `step_freq` of the slow and fwd_back segments move the same way in A (bench) and B
(unitree_mujoco).

### 5.4 Knobs
| Knob | How | Default | Reference values |
|---|---|---|---|
| cap C | `--air-time-max 0.45` | None (P1) | Spot `mode_time` 0.3 s |
| target T | `--air-time 0.25` | 0.3 s (P1 `[fix]`) | legged_gym 0.5 (ANYmal); Isaac Lab Go2 keeps 0.5 but lowers the weight |
| weight | `--set reward.reward_scales.feet_air_time=0.5` | 1.0 | Isaac Lab Go2 flat 0.25, rough 0.01; unitree_rl_lab Go2 0.1 (all with tracking 1.5) |
| contact threshold | `--set reward.contact_force_threshold=2.0` | 1.0 N | also changes the P2.4 timers |

### 5.5 Symptoms → fixes
| Symptom | Likely cause | Fix |
|---|---|---|
| very slow stepping (≤ 1 Hz), long `air_t`, poor slow tracking | uncapped term (5.1) | `--air-time-max 0.4` |
| high duty, short `air_t`, low `swing_h`, more `slip_v` | steps cost more than they earn | lower T (0.25, 0.2) |
| bouncing base, `duty` < 0.4, pronk-like gait plot | airborne time over-rewarded | lower the weight (0.5), or a cap |
| one foot rarely touches down (gait plot), `air_asym` high | held-up leg earns airborne time | the cap; P2.4's variance term also fixes it |
| tracking at 1 m/s degrades | T above what 1 m/s allows | lower T |

### 5.6 Done when
`air_t` where you want it (near C, or unchanged by choice), tracking and falls unchanged or better (`~`), on 2 seeds,
and the same direction of change in unitree_mujoco.

---

## 6. P2.2 Foot clearance (now in P1)

### 6.1 How the term works with your values
Isaac Lab Spot `foot_clearance_reward`, in P1 since `[S2R]`:

    r = exp( − Σ_feet (z_foot − z*)² · tanh(2·|v_foot,xy|) / 0.02 )        z* = 0.023 + feet_height_target

- Only feet that move horizontally count (`tanh` ≈ 1 above ~1 m/s, 0 at rest): swinging feet, and also **sliding**
  feet. A toe dragged along the ground is penalized as hard as a low swing.
- `feet_height_target` (0.075 m) is the clearance of the sphere bottom, the quantity `swing_h` measures at the peak of
  each swing. Unitree's own trot uses 0.08 m (unitree_guide `_gaitHeight`, WTW deployment default).
- Strength with std 0.02 and weight 1.0 (your comment in `go2_train.py`): two feet dragged at ground level at 0.5 m/s
  give 0.65, i.e. a cost of 0.35 per second, against 2.25 per second for perfect tracking (16 %). At 1 m/s: 0.58. A
  swing peaking at 4 cm instead of 7.5 cm gives 0.89 (two moving feet). With Spot's std 0.05 the same cases give
  0.84 and 0.95: your 0.02 is about 2.5× steeper.
- The kernel scores every moving instant, not the peak: lifting and landing feet move horizontally too, so a policy
  satisfies it best with a quick vertical lift and a flat top. `swing_h` therefore often ends a little above the
  target.
- Ground: world z = 0. On a heightfield or sand, subtract the ground height under each foot.

### 6.2 Commands
The term is in the baseline, so P2.2 is an ablation and a target sweep, not an addition:
```bash
# 1) What it buys: the baseline without it (needed for the thesis anyway)
python go2_train.py -e go2-p22-off $BASE <P2.1 flags> --set reward.reward_scales.foot_clearance=None
# 2) Target and width
python go2_train.py -e go2-p22-h0.06 $BASE <P2.1 flags> --clearance 0.06
python go2_train.py -e go2-p22-std05 $BASE <P2.1 flags> --set reward.foot_clearance_std=0.05
for r in go2-p22-off go2-p22-h0.06 go2-p22-std05; do python go2_metrics.py -e $r --ckpt $CKPT; done
python go2_compare.py logs/go2-p21/metrics_ckpt1999_seed0.csv logs/go2-p22-{off,h0.06,std05}/metrics_ckpt1999_seed0.csv \
    --cols swing_h slip_v slip_d nonfoot air_t lin_err power CoT
python go2_feet_plot.py logs/go2-p21/metrics_ckpt1999_seed0.npz --cmd 0.5 0 0 --target 0.075
python go2_feet_plot.py logs/go2-p22-off/metrics_ckpt1999_seed0.npz --cmd 0.5 0 0 --target 0.075
```
`<P2.1 flags>` = what you kept in P2.1 (e.g. `--air-time-max 0.45`), so `go2-p21` is the reference. Keep the best
as `go2-p22` (it may be `go2-p21` itself).

### 6.3 What to read
| Metric | Expect | Bad sign |
|---|---|---|
| `swing_h[cm]` | within ~1–2 cm of the target at moving bins (slightly above is normal: it is the peak) | > 3 cm above: marching (CoT up); far below: term too weak |
| `slip_v[cm/s]`, `slip_d[cm/step]` | low (no dragging); higher in the `-off` run | |
| `nonfoot[%time]` | ~0 | knees / shins touching |
| `power`, `CoT` | slightly above the `-off` run (lifting costs work) | strongly up: target or weight too high |
| `lin_err` | `~` | worse |
| feet plot | flat-ish top near the dashed line, speed peak inside the swing, little speed in the shaded stance, no low and fast segment outside it | |
TensorBoard: `Episode/rew_foot_clearance` rises toward the weight (1.0 = every moving foot on target).
Sim-to-sim: `swing_h` from leg kinematics in A and B. The `[S2R]` world gives Genesis unitree_mujoco's joint losses and
heavier lower legs, which both make lifting harder, so I expect the two to agree within about 1 cm; a larger gap
means the swing depends on something the two worlds still model differently.

### 6.4 Knobs
| Knob | How | Default | Effect |
|---|---|---|---|
| target | `--clearance 0.06` | 0.075 m | higher = more margin on sand / obstacles, more energy |
| weight | `--set reward.reward_scales.foot_clearance=0.5` | 1.0 | Spot: 0.5 next to tracking 5 + 5 (weaker relative to tracking) |
| kernel width | `--set reward.foot_clearance_std=0.05` | 0.02 m² | your comment: back to 0.05 if `swing_h` overshoots by > 3 cm |
| speed gain | `--set reward.foot_clearance_tanh_mult=4.0` | 2.0 s/m | higher = slower feet (side steps, turning) count fully |

### 6.5 Symptoms → fixes and an extra test
| Symptom | Fix |
|---|---|
| `swing_h` well below target | weight 1.5, or check P2.1 (very short swings leave no time to lift) |
| high stepping, CoT up a lot | std 0.05, weight 0.5 or a lower target |
| side steps (vy bins) still drag | `foot_clearance_tanh_mult` 4.0 |
| front feet clear, hind feet drag (feet plot) | check `nonfoot`, raise the weight; the hind legs carry more load |

Extra test in unitree_mujoco: its default `go2/scene.xml` has 8 cm bars across the path at x ≈ 1.2 m and 1.6 m (your
suite uses `scene_flat.xml` when it exists). A blind policy has no idea where they are, but with 7.5 cm of clearance
it should catch them less often than `go2-p22-off`:
```bash
python go2_mujoco_headless.py --um $UM --scene $UM/unitree_robots/go2/scene.xml --dt 0.002 --video bars.mp4 &
python go2_deploy_sdk2.py --policy logs/go2-p22/exported --sim --scenario walk --profile slow --yes
```

### 6.6 Done when
`swing_h` near the target, `slip_v` below the `-off` run, tracking unchanged, CoT not more than ~10 % above the `-off`
run, on 2 seeds; `swing_h` agrees between the bench and unitree_mujoco.

---

## 7. P2.3 Symmetry

### 7.1 The mirror
Reflection through the sagittal plane (y → −y), implemented in `go2_symmetry.py`:

| Quantity | Mirror |
|---|---|
| legs | FL ↔ FR, RL ↔ RR |
| hip (abduction, x axis) angle / velocity / action | sign flip |
| thigh, calf (y axis) | unchanged |
| base angular velocity (pseudovector) | (ωx, ωy, ωz) → (−ωx, ωy, −ωz) |
| projected gravity | (gx, gy, gz) → (gx, −gy, gz) |
| commands | (vx, vy, wz, dh) → (vx, −vy, −wz, dh) |
| P2.4 clock (sin, cos per foot) | FL ↔ FR, RL ↔ RR (for a trot: the same trot half a period later) |

Front-back is not a symmetry of the Go2 (all knees point backward, different front/rear thigh defaults and limits),
so only left-right is used. Every reward in `go2_env.py` is mirror-invariant, which augmentation needs: rsl_rl gives
mirrored samples the log-probabilities, values, advantages and returns of their originals. The `[S2R]` additions keep
the robot symmetric: the lower-leg CoM keeps the sign of each side, torsional and rolling friction are isotropic, and
single-axis commands are sampled with a uniform sign.

**Check it before training** (and after any change to the observation): `python go2_symmetry.py`. It drives two
Genesis robots, the second with the mirrored command and mirrored actions, and checks that the second observation
stays the mirror of the first. Measured on the `[S2R]` world (elliptic cone, torsional and rolling friction, go2.xml
lower legs), largest difference per block: ang_vel 4.6e-4, gravity 5.1e-5, dof_pos 6.0e-5, dof_vel 9.1e-5, clock
3e-6 (`--clock`). A wrong sign shows up at the size of the signal (flipping the angular-velocity rule gives about 0.8).

### 7.2 Modes (`--symmetry`)
| Mode | rsl_rl setting | What it does | When |
|---|---|---|---|
| `aug` (default) | `use_data_augmentation` | every mini-batch gets the mirrored samples too | first choice: Mittal et al. 2024 report the highest return and fastest convergence, with a low symmetry loss |
| `loss` | `use_mirror_loss`, coef 1.0 | MSE between π(mirror s) and mirror π(s) added to the loss | symmetry as a soft constraint; Mittal et al.: induces symmetry but costs return and convergence speed |
| `both` | both | | Mittal et al.: not necessarily better than `aug` |
| `log` | neither | only logs `Loss/symmetry` | free monitoring of any run (training unchanged) |

### 7.3 Commands
```bash
python go2_symmetry.py
python go2_train.py -e go2-p23 $BASE <P2.1/P2.2 flags> --symmetry            # aug
python go2_train.py -e go2-p23-loss $BASE <P2.1/P2.2 flags> --symmetry loss  # optional comparison
python go2_metrics.py -e go2-p23 --ckpt $CKPT
python go2_compare.py logs/go2-p22/metrics_ckpt1999_seed0.csv logs/go2-p23/metrics_ckpt1999_seed0.csv \
    --cols sym_err air_asym vy_meas wz_meas lin_err yaw_err air_t CoT
# sim-to-sim (3.8) with PROF=lateral and PROF=yaw: segments 1 and 3 are mirror images (+ then -)
```

### 7.4 What to read
| Metric | Expect | Notes |
|---|---|---|
| `sym_err[rad]` (Gait pattern & symmetry) | large drop vs `go2-p22` | RMS over joints of π(mirror s) − mirror π(s), in joint-target radians, on the visited states |
| `air_asym[%]` | lower | left vs right mean swing duration, front and hind pairs |
| Mirror-pair check (printed after the tables) | relative differences small | (0, ±0.3, 0) and (0, 0, ±0.5) should perform alike; differences use a floor so near-zero values do not show as 100 % |
| `vy_meas`, `wz_meas` at (0.5, 0, 0) and (1, 0, 0) | ~0 | sideways drift / turning on a straight command |
| `Loss/symmetry` (TensorBoard) | falls fast with `aug`, stays higher with `log` | `sym_err ≈ 0.25·√(Loss/symmetry)` roughly |
| `lin_err`, CoT | `~` or better | |
| `go2_analyze.py`, lateral / yaw seg 1 vs seg 3 | `vy_meas` (`wz_meas`) of opposite sign and equal size, `air_asym` low, in A and B | a gap that appears only in B = an asymmetry of the MuJoCo model or of the policy under MuJoCo contacts |

### 7.5 Knobs
| Knob | How | Default |
|---|---|---|
| mode | `--symmetry aug/loss/both/log` | aug |
| mirror-loss weight | `--symmetry loss --set train.algorithm.symmetry_cfg.mirror_loss_coeff=0.3` | 1.0 for `loss`/`both` (no published value; Mittal et al.: too high a weight hurts training; try 0.1–3 and watch `Loss/surrogate`, `Train/mean_reward`) |

### 7.6 Symptoms, caveats
| Symptom | Cause / fix |
|---|---|
| `Loss/symmetry` stays high with `aug` | mirror bug: run `python go2_symmetry.py` |
| `ValueError ... recurrent policies` | rsl_rl 5.5.1 does not support symmetry with `--rnn` (`go2_train.py` refuses the combination) |
| learning slower per iteration | `aug` doubles each mini-batch (`Perf/learning_time`); the simulation time is unchanged |
| symmetric in Genesis, asymmetric in unitree_mujoco or on hardware | the model or robot is not exactly symmetric (go2.urdf base inertia products ixy = 1.2e-4, iyz = −3e-5; real battery, payload, motor differences); Mittal et al. still climbed with ANYmal-D, which is not symmetric either |

### 7.7 Done when
`sym_err` and `air_asym` clearly down, mirror pairs alike, no drift at straight commands, tracking unchanged; lateral
and yaw segments symmetric in unitree_mujoco too.

---

## 8. P2.4 Gait pattern (only if needed)

### 8.1 Is it needed?
After P2.3, for the moving bins (0.25–1 m/s, lateral, turning), look at:
- `gait_ckpt1999_seed0.png`: are the diagonal pairs (FL–RR, FR–RL) aligned, and the pattern regular?
- `diag_sync[%]`: 100 = trot (or pronk, see the plot), ~50 = 4-beat walk or unrelated feet, ~0 = pace or bound.
- `air_cv[%]`: cycle-to-cycle spread of each foot's swing durations (mean over feet), ~5–15 % for a periodic gait,
  > 30 % = irregular.
- `air_asym[%]`: should already be low after P2.3.
- The same columns from `go2_analyze.py` in unitree_mujoco: a gait that is a clean trot in Genesis but irregular in
  MuJoCo is a world-gap problem, which a gait clock can also mask (option B forces the timing in both worlds).

My suggested threshold, not a literature value: if `diag_sync` ≳ 85–90 % and `air_cv` ≲ 20–25 % at walking speeds,
skip P2.4. A gait that is regular but not a trot (a clean walk at low speed) is not a reason either, unless you
specifically want a trot.

### 8.2 Option A: Spot GaitReward + air-time variance (`--gait spot`)
- `gait` (+1.5): product of 6 exp kernels on the current air and contact times. FL–RR and FR–RL must match each other,
  and the air time of one pair must equal the contact time of the other. It equals 1 for a perfect trot at 50 % duty.
  Active when a motion is commanded or the base moves faster than 0.5 m/s.
- `air_time_variance` (−1.0, ramped by the curriculum): variance over the 4 feet of the last swing and last stance
  durations, clipped at 0.5 s. Same weight as unitree_rl_lab's Go2 configuration (same tracking weights as yours).
  Gated like `gait`, because at zero command the timers are stale.
- Weight relative to tracking: Spot uses gait 10 next to 5 + 5 of tracking, i.e. 2.25 in your scale; 1.5 is two
  thirds of that.
- No observation change, nothing to do on the deploy side.

```bash
python go2_train.py -e go2-p24-spot $BASE <P2.1-P2.3 flags> --gait spot
python go2_metrics.py -e go2-p24-spot --ckpt $CKPT
python go2_compare.py logs/go2-p23/metrics_ckpt1999_seed0.csv logs/go2-p24-spot/metrics_ckpt1999_seed0.csv \
    --cols diag_sync air_cv air_asym duty air_t lin_err CoT
```
| Knob | How | Default |
|---|---|---|
| gait weight | `--set reward.reward_scales.gait=2.25` | 1.5 |
| variance weight | `--set reward.reward_scales.air_time_variance=-0.5` | −1.0 |
| kernel width | `--set reward.gait_std=0.05` | 0.1 s² (smaller = stricter timing) |
| max error | `--set reward.gait_max_err=0.1` | 0.2 s |
| speed gate | `--set reward.gait_velocity_threshold=0.3` | 0.5 m/s |
| variance clip | `--set reward.air_time_variance_clip=0.4` | 0.5 s |

If your swings or stances are longer than the variance clip (0.5 s), the clipped durations all look alike: raise the
clip above them (e.g. 0.8).

### 8.3 Option B: walk-these-ways contact schedule with clock (`--gait wtw`)
- A gait clock: phase += f·dt per policy step, per-foot offsets (trot: FL, RR at +0.5; FR, RL at 0), phase warped so
  that stance is [0, 0.5) and swing [0.5, 1) for any duty, and a smoothed desired contact (normal CDFs, κ = 0.07).
- Observation: 8 entries appended (sin and cos of each foot's phase). sin alone, as in WTW, is ambiguous for an MLP
  without history; WTW has an observation history. The first 45 (46) entries keep the P1 layout.
- Rewards, written as penalties: `contact_schedule_force` = contact force on a foot that should swing,
  `contact_schedule_vel` = speed of a foot that should stand (WTW `tracking_contacts_shaped_force/vel`, σ = 100 N²
  and 10 m²/s²). Weight −3 each (WTW: +4 next to tracking 1 + 0.5, so this is half of WTW's relative strength,
  because `only_positive_rewards` clips the total).
- At zero command the schedule asks all four feet to stand (WTW trots in place); the clock keeps running.
- Frequency: 2 Hz at 50 % duty gives 0.25 s swings. **`--gait wtw` removes `feet_air_time`**: P1's 0.3 s target is
  above the clock's swing, so every scheduled step would cost reward, and WTW has no such term. Put it back with
  `--set reward.reward_scales.feet_air_time=1.0 --air-time 0.2` if you want both (`go2_train.py` warns when the target
  exceeds the clock's swing).
- 2 Hz is well above what your policies stepped at so far (p17h on the slow profile: 1.22 Hz in plain Genesis,
  0.83 Hz in unitree_mujoco). For the smallest change of gait, set the clock near the baseline's `step_freq` at
  0.5 m/s (e.g. `--set env.gait_clock.frequency=1.5`); for WTW's regime, keep 2.
- Deploy: `go2_export.py` writes the clock parameters, `go2_fsm.Policy` runs the clock (phase 0 at each policy start,
  as Go2Env at reset), `go2_fsm_sim.py` checks the deploy observation, clock included, against Go2Env. For the clock,
  Go2Env's phase is recomputed from its own dt and frequency (k-th observation: k·dt·f), so the check covers the
  phase timing, not only the formula. `go2_deploy_sdk2.py` prints the clock at start.

```bash
python go2_train.py -e go2-p24-wtw $BASE <P2.2/P2.3 flags> --gait wtw
python go2_metrics.py -e go2-p24-wtw --ckpt $CKPT
python go2_export.py -e go2-p24-wtw --ckpt $CKPT && python go2_fsm_sim.py -e go2-p24-wtw   # obs check ~1e-6
python go2_symmetry.py --clock
# sim-to-sim (3.8): sched_match in A and B
```
| Knob | How | Default | Notes |
|---|---|---|---|
| frequency | `--set env.gait_clock.frequency=1.5` | 2.0 Hz | WTW samples 2–4 Hz (Go1); swing = (1 − duty)/f |
| duty | `--set env.gait_clock.duty=0.6` | 0.5 | stance share |
| gait | `--set "env.gait_clock.offsets={'FL':0.0,'FR':0.5,'RL':0.0,'RR':0.5}"` | trot | pace = this; bound `{'FL':0,'FR':0,'RL':0.5,'RR':0.5}`; pronk all 0 |
| smoothing | `--set env.gait_clock.kappa=0.05` | 0.07 | |
| weights | `--set reward.reward_scales.contact_schedule_force=-4.0` | −3 / −3 | |
| kernels | `--set reward.gait_force_sigma=50.0` | 100 N², 10 m²/s² | |

With a clock, `--symmetry` still works: the mirror swaps the per-foot clocks (FL ↔ FR, RL ↔ RR). That is a state the
clock produces only if the left-right offset difference is 0 or 0.5 and the same at the front and at the rear: trot
and pace (the same gait half a period later), bound and pronk (the same state). Other offsets would make augmentation
invent clock states the schedule never produces (nothing in the code checks this). `go2_symmetry.py --clock` checks
the trot.

For later (sand): WTW makes frequency and swing height *commands*, so one policy can change them on the fly. Here
they are fixed per run; making them commands means adding them to `commands` (as P1.6 did for dh), the export and the
FSM.

### 8.4 What to read
| Metric | Option A | Option B |
|---|---|---|
| `diag_sync` | up toward 100 % | ~100 % by design |
| `air_cv`, `air_asym` | down | low |
| `step_freq` | free | ≈ clock frequency (else the robot ignores the clock: raise the weights) |
| `air_t` | | ≈ (1 − duty)/f minus contact detection |
| `sched_match_true` (`go2_analyze.py`) | – | near the open-loop reference (90 %, 3.8) in A and B; lower in B only = MuJoCo contacts break the schedule |
| `lin_err`, CoT | `~` (if worse, weight too high) | `~` |
| TensorBoard | `Episode/rew_gait` rises | `Episode/rew_contact_schedule_*` toward 0 |

`sched_match` compares the stance with the clock's ideal square schedule, so even a perfect follower reads below
100 %: the open-loop trot of 3.8 reads 90 % with true contacts and 76 % with the torque estimate. Compare A with B,
and runs with each other.

If `Train/mean_reward` stays near 0 for long after enabling a gait term, the penalties exceed the positive terms and
`only_positive_rewards` clips the total to 0. Halve the weights.

### 8.5 Done when
`diag_sync`, `air_cv`, `air_asym` meet your criterion in Genesis and in unitree_mujoco, with tracking and CoT not
worse.

---

## 9. P2.5 Energy

### 9.1 What
`energy = −W · Σ_joints |τ·q̇|`, with τ the PD torque and q̇ the joint velocity of each 200 Hz physics step, averaged over
the policy step. Negative work is not credited back, which matches the CoT of `go2_metrics.py`: CoT = mean power /
(m·g·v). The joint losses (`[S2R]`) act outside the motor, so they enter only through the extra torque the PD needs
to overcome them, as on the robot (Σ|τ_est·q̇|, `go2_analyze.py` `power[W]`).
unitree_rl_lab's Go2 `energy` term is the same quantity with W = 2e-5 next to the same tracking weights, a
tie-breaker. Fu et al. 2021 make energy the main regularizer: α = 0.04 on the summed motor power τᵀq̇ at 100 Hz, next
to a forward-velocity term and an alive bonus. Walk, trot and bound then emerge on A1 at 0.375, 0.9 and 1.5 m/s.

### 9.2 Calibrating W
The term matters when it is a few percent of the tracking reward. Read `power[W]` of your latest run at (0.5, 0, 0),
then:

    W ≈ 0.05 × 2.25 / P_0.5        (5 % of the maximum tracking reward of 1.5 + 0.75 per second)

That gives 2.25e-3 at 50 W and 1.5e-3 at 75 W. The `--energy` default is 1.5e-3. The curriculum ramps it from 0.3 W
to W over the first 400 iterations.

### 9.3 Commands
```bash
for W in 5e-4 1.5e-3 4e-3; do python go2_train.py -e go2-p25-w$W $BASE <P2.1-P2.4 flags> --energy $W; done
for W in 5e-4 1.5e-3 4e-3; do python go2_metrics.py -e go2-p25-w$W --ckpt $CKPT; done
python go2_compare.py logs/go2-p23/metrics_ckpt1999_seed0.csv logs/go2-p25-w{5e-4,1.5e-3,4e-3}/metrics_ckpt1999_seed0.csv \
    --cols power CoT lin_err vx_meas step_freq duty air_t tau_rms "hf>10Hz" --rows "0.25 0 0" "0.5 0 0" "0.75 0 0" "1 0 0"
# sim-to-sim (3.8), PROF=fwd_back: power and CoT in A and B
```

### 9.4 What to read
| Metric | Expect | Bad sign |
|---|---|---|
| `power[W]`, `CoT[-]` (Energy) | down at 0.25–1 m/s | |
| `lin_err`, `vx_meas` at 1.0 m/s | `~` | undershoot (`vx_meas` < command): W too high, energy beats tracking |
| `step_freq`, `duty`, `air_t` | may shift with speed (Fu et al.: the gait changes with speed) | |
| `tau_rms`, `tau>10Hz`, `hf>10Hz` | down (power also penalizes chatter) | |
| standing bin (0, 0, 0) | power lower | |
| `go2_analyze.py` `power[W]`, `CoT[-]` | lower in A and B by a similar share | lower only in A: the saving relies on Genesis contact details |
TensorBoard: training-time power = `−Episode/rew_energy / W` (after iteration 400).

### 9.5 Knobs and an experiment
| Knob | How | Default |
|---|---|---|
| weight | `--energy 5e-4` | 1.5e-3 (flag) |
| existing torque penalty | `--set reward.reward_scales.torques=-2e-4` | −1e-5 (unitree_rl_lab Go2: −2e-4) |

Experiment (Fu et al.): energy instead of hand-made gait shaping. What gait emerges with tracking + energy but without
the air-time and clearance terms?
```bash
python go2_train.py -e go2-p25-fu $BASE --symmetry --energy 4e-3 \
    --set reward.reward_scales.feet_air_time=None reward.reward_scales.foot_clearance=None
```

### 9.6 Hardware CoT
- Mechanical, comparable with Genesis: Σ|`tau_est`·`dq`| from `LowState.motor_state[i]`: `go2_analyze.py`
  `power[W]` on a robot log.
- Electrical: `LowState.power_v × LowState.power_a` (bus voltage × current; `go2_probe.py` records both). It includes
  motor Joule losses and the electronics, so expect it several times above the mechanical one, and report which one
  you use.
- CoT = P / (m·g·v) with v from motion capture or a timed, measured distance (`go2_analyze.py` uses leg odometry on the
  robot, which overestimates when feet slip).

### 9.7 Done when
CoT down at 0.5–1 m/s with tracking within noise, on 2 seeds, and in unitree_mujoco.

---

## 10. Final P2 run, report and sim-to-sim gate

```bash
P2FLAGS="<P2.1 flags> <P2.2 flags> --symmetry --energy 1.5e-3"     # + --gait ... if needed
python go2_train.py -e go2-p2 $BASE $P2FLAGS
python go2_train.py -e go2-p2-s2 $BASE --seed 2 $P2FLAGS
python go2_metrics.py -e go2-p2 --ckpt $CKPT
python go2_metrics.py -e go2-p2 --ckpt $CKPT --keep_dr --tag dr
python go2_compare.py logs/go2-p20/metrics_ckpt1999_seed0.csv logs/go2-p2/metrics_ckpt1999_seed0.csv --preset gait
python go2_compare.py logs/go2-p20/metrics_ckpt1999_seed0_dr.csv logs/go2-p2/metrics_ckpt1999_seed0_dr.csv --preset gait
python go2_export.py -e go2-p2 --ckpt $CKPT
python test_go2_deploy.py --policy logs/go2-p2/exported
python go2_fsm_sim.py -e go2-p2 && python go2_fsm_sim.py -e go2-p2 -n 16 --dr && python go2_fsm_sim.py -e go2-p2 --scenario estop
# sim-to-sim (3.8) for slow, fwd_back, lateral, yaw; then the suites
python go2_sim_suite.py --policy logs/go2-p2/exported --um $UM --suite gate
python go2_sim_suite.py --policy logs/go2-p2/exported --um $UM --suite stress
```
The suite's `summary.md` now has a gait column (air_t / diag_sync of each moving segment): under noise, delays or a
payload the run may still pass while the trot falls apart, which this column shows.

Results log to fill in (at (0.5, 0, 0) unless stated; mean over seeds; G = `go2_metrics.py` in Genesis, M =
`go2_analyze.py` in unitree_mujoco):

| Run | Flags | Seeds | lin_err G | air_t G / M | stride G | swing_h G / M | slip_v G | sym_err G | diag_sync G / M | air_cv G | CoT 0.5 G / M | CoT 1.0 G | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| go2-p20 | (baseline) | 1, 2 | | | | | | | | | | | |
| go2-p21 | + P2.1 | | | | | | | | | | | | |
| go2-p22-off | − foot_clearance | | | | | | | | | | | | |
| go2-p23 | + --symmetry | | | | | | | | | | | | |
| go2-p24 | + --gait ... | | | | | | | | | | | | |
| go2-p25 | + --energy W | | | | | | | | | | | | |

---

## 11. Reference: configuration keys and constants

### env_cfg (P2 keys, and the P1 `[S2R]` keys P2 relies on)
| Key | Default | Unit | Part | Set by |
|---|---|---|---|---|
| `joint_friction`, `joint_damping` | 0.2, 0.1 | Nm, Nm·s/rad | P1 `[S2R]` | `--set env.joint_friction=` |
| `domain_rand.joint_friction_range`, `joint_damping_range` | absent | Nm, Nm·s/rad | P1 `[S2R]` | `--set env.domain_rand....=[lo,hi]` |
| `foot_torsional_friction`, `foot_rolling_friction` | 0.02, 0.01 | m | P1 `[S2R]` | `--set env.foot_torsional_friction=None` (off) |
| `friction_cone`, `impratio` | elliptic, 100 | | P1 `[S2R]` | `--set env.friction_cone=pyramidal` |
| `lower_leg_inertial` | go2.xml | kg, m | P1 `[S2R]` | `--set env.lower_leg_inertial=None` |
| `domain_rand.friction_range` | [0.4, 1.25] | μ | P1 `[S2R]` | `--set env.domain_rand.friction_range=` |
| `gait_clock` | absent | dict | P2.4 B | `--gait wtw` → `{frequency 2.0, duty 0.5, kappa 0.07, offsets trot}` |

### reward_cfg
| Key | Default | Unit | Part |
|---|---|---|---|
| `feet_air_time_target` | 0.3 | s | P1 `[fix]`, `--air-time` |
| `feet_air_time_max` | None | s | P2.1, `--air-time-max` |
| `contact_force_threshold` | 1.0 | N | P1.1 (used by P2.1, P2.4) |
| `feet_height_target` | 0.075 | m (clearance) | P1 `[S2R]`, `--clearance` |
| `foot_clearance_std` | 0.02 | m² | P1 `[S2R]` |
| `foot_clearance_tanh_mult` | 2.0 | s/m | P1 `[S2R]` |
| `gait_std` | 0.1 | s² | P2.4 A |
| `gait_max_err` | 0.2 | s | P2.4 A |
| `gait_velocity_threshold` | 0.5 | m/s | P2.4 A |
| `air_time_variance_clip` | 0.5 | s | P2.4 A |
| `gait_force_sigma` | 100 | N² | P2.4 B |
| `gait_vel_sigma` | 10 | m²/s² | P2.4 B |
| `penalty_scale_init`, `penalty_ramp_steps` | 0.3, 400 × `num_steps_per_env` (400 iterations) | env steps | P1 `[fix]` (applies to P2 penalties) |

### reward_scales (before the ×dt done by Go2Env)
| Term | Weight | Set by |
|---|---|---|
| `feet_air_time` | +1.0 | P1; removed by `--gait wtw` |
| `foot_clearance` | +1.0 | P1 `[S2R]` |
| `gait` | +1.5 | `--gait spot` |
| `air_time_variance` | −1.0 | `--gait spot` |
| `contact_schedule_force` | −3.0 | `--gait wtw` |
| `contact_schedule_vel` | −3.0 | `--gait wtw` |
| `energy` | −1.5e-3 | `--energy` |

### train_cfg
`algorithm.symmetry_cfg = {data_augmentation_func: "go2_symmetry:compute_symmetric_states", use_data_augmentation,
use_mirror_loss, mirror_loss_coeff}` (rsl_rl adds `env` itself).

### Constants
- `go2_env.py`: `GO2_FOOT_RADIUS` (0.023 m, P1), `TROT_PAIRS`, `WTW_TROT_OFFSETS`, `num_clock_obs()`, `gait_targets()`.
- `go2_train.py`: `MIRROR_LOSS_COEF` (1.0), `ENERGY_WEIGHT` (1.5e-3), `SPOT_GAIT`, `WTW_GAIT`, `WTW_CLOCK`.
- `go2_metrics.py`: `CONTACT_N` (3 N), `FOOT_R`, `MOVING_CMD` (0.1), `MIRROR_PAIR_METRICS`, `MIRROR_PAIR_FLOOR`.
- `go2_analyze.py`: `MOVING_CMD`, `DIAGONALS`, `MIRRORS`, `MIN_RUN` (40 ms debounce, P1).

---

## 12. Reference: metrics glossary (new and changed)

| Column | Where | Definition | Better | Part |
|---|---|---|---|---|
| `air_t[s]` | metrics Gait; analyze | mean duration of complete swings (metrics: 3 N contact; analyze: estimated contact, `_true`: simulator forces) | near the cap / your choice | P2.1 |
| `stride[cm]` | metrics Gait | mean horizontal foot travel per swing (last contact → first contact) | context: ≈ 100·v/step_freq | P2.1 |
| `swing_h[cm]` | metrics Gait; analyze | mean peak clearance per swing (analyze: from leg kinematics) | near the target | P2.2 |
| `diag_sync[%]` | metrics Gait pattern & symmetry; analyze | while ≥ 1 foot is in the air, share of time each diagonal pair has the same contact state; n/a at standing commands or with < 5 % air time | 100 for a trot | P2.4 |
| `air_cv[%]` | idem | std / mean of each foot's swing durations, mean over feet with ≥ 3 swings | low (regular) | P2.4 |
| `sym_err[rad]` | metrics | RMS over joints of π(mirror s) − mirror π(s) × action_scale; n/a for RNN | low | P2.3 |
| `air_asym[%]` | metrics; analyze | mean over the left/right pairs of the swing-duration difference / mean | low | P2.3 |
| mirror-pair check | metrics, printed | metrics of mirror-image bins side by side, \|A−B\| / max(mean, floor) | small % | P2.3 |
| `sched_match[%]` | analyze (`--gait wtw` exports) | share of time and feet whose stance (estimated; `_true`: simulator forces) equals the clock's (phase < duty), from the logged observations; moving phases only | high; compare worlds | P2.4 |
| `power[W]` | metrics Energy; analyze | metrics: mean Σ\|τ·q̇\| over the 200 Hz physics steps; analyze: Σ\|τ_est·q̇\| | low | P2.5 |
| `CoT[-]` | idem | power / (m·g·speed), n/a below 0.1 m/s command | low | P2.5 |

Typical relationships for a periodic gait: `air_t ≈ (1 − duty)/step_freq`, `stride ≈ 100·vx_meas/step_freq` (cm).

---

## 13. Troubleshooting

| Message / symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'go2_symmetry'` (eval, metrics, export, teleop) | the run was trained with `--symmetry`; rsl_rl imports the function when it builds PPO | copy `go2_symmetry.py` next to the scripts / run from that folder |
| `--symmetry is not supported by rsl_rl for recurrent policies` | rsl_rl 5.5.1 limitation | drop `--rnn` or `--symmetry` |
| `go2_symmetry: observation of size X, the mirror knows ...` | a new observation term | add its mirror in `go2_symmetry._obs_layout()`, then `python go2_symmetry.py` |
| `the contact_schedule_* rewards need env_cfg['gait_clock']` | WTW terms without the clock | use `--gait wtw` |
| `deploy_cfg.json and the env disagree on gait_clock` (fsm_sim) | stale export | re-run `go2_export.py` |
| `--cols air_t: no such metric` (compare) | CSV from the old `go2_metrics.py` | re-run the evaluation |
| `WARNING: the clock asks for 0.25 s swings ...` | you put `feet_air_time` back with a target above the clock's swing | `--air-time 0.2` or leave the term out |
| `Train/mean_reward` ≈ 0 for a long time | penalties exceed positive terms; `only_positive_rewards` clips to 0 | halve the newest penalty weight |
| a new penalty seems weak for 400 iterations | penalty curriculum (×0.3 → 1) | expected; `--set reward.penalty_scale_init=1.0` to compare without it |
| `--air-time: expected one argument` | the flag now needs a value (P1 sets the default) | `--air-time 0.25` |
| `sched_match` n/a | not a clock policy, standing phase, or no `pol` stream in the log | run a `walk` scenario with a profile |
| gait columns differ a lot between `go2_metrics.py` and `go2_analyze.py` | different contact detection (3 N force vs torque estimate with debounce) | compare each tool with itself, world against world |
| slower simulation | joint-loss DR (batched dof info), torsional / rolling friction, elliptic cone | P1 `[S2R]` cost, not P2 |

---

## 14. Sources

- unitree_mujoco `unitree_robots/go2/go2.xml` (joint damping 0.1, frictionloss 0.2, armature 0.01; foot friction
  "0.4 0.02 0.01", condim 6, priority 1; `<option cone="elliptic" impratio="100"/>`) and `go2/scene.xml` (bars and
  stairs).
- Genesis 1.4.3: `RigidEntity.set_dofs_frictionloss / set_dofs_damping`, rigid constraint solver; bundled
  `urdf/go2/urdf/go2.urdf` (no `<dynamics>` tags).
- legged_gym `legged_robot.py` `_reward_feet_air_time` (0.5 s, \|cmd_xy\| > 0.1 gate); `anymal_c_rough_config.py`
  (`base_height_target = 0.5`).
- Isaac Lab `locomotion/velocity/config/spot/mdp/rewards.py` (`air_time_reward` with `mode_time`, `GaitReward`,
  `foot_clearance_reward`, `air_time_variance_penalty`) and `spot/flat_env_cfg.py` (weights); `config/go2/flat_env_cfg.py`
  and `rough_env_cfg.py`; `velocity_env_cfg.py` (`feet_air_time` threshold 0.5).
- unitree_rl_lab `tasks/locomotion/robots/go2/velocity_env_cfg.py` (energy −2e-5, air_time_variance −1,
  feet_air_time 0.1, tracking 1.5 / 0.75) and `tasks/locomotion/mdp/rewards.py` (`energy`).
- unitree_guide `main.cpp` (`WaveGenerator(0.45, 0.5, ...)` trot) and `State_Trotting.cpp` (`_gaitHeight = 0.08`).
- Walk-these-ways (Margolis & Agrawal 2022): `go1_gym/envs/base/legged_robot.py` `_step_contact_targets`,
  `go1_gym/envs/rewards/corl_rewards.py`, `scripts/train.py` (weights, σ, κ, 2–4 Hz), `scripts/play.py` (3 Hz, 0.08 m).
- rsl-rl-lib 5.5.1: `rsl_rl/extensions/symmetry.py`, `rsl_rl/algorithms/ppo.py`, `rsl_rl/utils/logger.py`.
- Mittal et al., "Symmetry Considerations for Learning Task Symmetric Robot Policies", ICRA 2024 (arXiv:2403.04359).
- Fu, Kumar, Malik, Pathak, "Minimizing Energy Consumption Leads to the Emergence of Gaits in Legged Robots",
  CoRL 2021 (arXiv:2111.01674).
- unitree_sdk2_python `unitree_go/msg/dds_/_LowState_.py` (`power_v`, `power_a`, `foot_force`) and `_MotorState_.py`
  (`tau_est`).
