# Go2 locomotion: P1.1 to P1.6, P3.1 to P3.4 and P3 v2 changes

**P3.1** (curriculum training on terrain), **P3.2** (foot slip from ground truth, and the foot-slip penalty decision), **P3.3** (privileged observations for the critic and a teacher) and **P3.4** (Baseline v1.0: provenance, checked ONNX export, harness report, config card, git tag) are the sections after P1.6; **P3 v2** applies your P2 results to them. They build on the P2 code (P2 is documented in `TEST_AND_TINKER_P2.md`), and their companion guide is `TEST_AND_TINKER_P3.md`.

Base: your `go2_env.py`, `go2_train.py`, `go2_eval.py` and `go2_metrics.py` (Genesis 1.4.2, rsl-rl-lib 5.5.1).

## How to use this bundle

- `p1_1/` to `p1_6/` are complete, runnable snapshots. Each one builds on the previous one.
- `patches/` holds the same changes as one git patch per phase. Apply them in order, in the folder that holds the four files:
  `git apply p1_1_stand_still.patch`, then `p1_2_...`, `p1_3_...`, `p1_4_...`, `p1_5_...`, `p1_6_...`.
  - Line endings are CRLF, like your files. P1.4, P1.5 and P1.6 also add new files.
  - `git apply` may warn about "trailing whitespace" in new files: that is the CR of each CRLF line. Harmless.
- In the code, every change is tagged `[P1.1]` to `[P1.6]`, with a comment saying why.
- **`TEST_AND_TINKER.md`** is the companion to this file: the commands of each phase, the knobs and switches, how to read TensorBoard and `go2_metrics.py`, experiments to run, troubleshooting.
- `go2_compare.py` (P1.6) puts the metrics CSVs of several runs side by side.
- Train each phase under its own name so `go2_metrics.py` can compare them.
- Checkpoints are saved every 100 iterations, plus the last one. **The last checkpoint of a 1000-iteration run is `model_999.pt`**: rsl_rl counts iterations from 0, so there is no `model_1000.pt`. The commands in the previous version of this file used `--ckpt 1000`; that was wrong.

```
python go2_train.py -e go2-p11                         # default 1000 iterations
python go2_eval.py  -e go2-p11 --ckpt 999 --cmd 0 0 0  # viewer, standing still
python go2_metrics.py -e go2-p11 --ckpt 999            # table + gait plot (+ spectra plot from P1.4)
python go2_metrics.py -e go2-p12 --ckpt 999 --keep_dr --measure_s 20   # P1.2+: under training randomization
python go2_export.py  -e go2-p15 --ckpt 999            # P1.5: policy.pt + deploy_cfg.json for the robot
python go2_fsm_sim.py -e go2-p15                       # P1.5: stand up -> policy -> sit down, in Genesis
python go2_train.py -e go2-p16h --height-cmd           # P1.6: + body-height command
python go2_train.py -e go2-p16r --rnn lstm             # P1.6: LSTM actor and critic
python go2_teleop.py -e go2-p16h --ckpt 999            # P1.6: drive it with the keyboard
```

**Checked here** (Genesis 1.4.2 and rsl-rl-lib 5.5.1 installed, CPU backend):
- Each phase builds, runs a few PPO iterations, and runs `go2_metrics.py` in the nominal and the randomized world. Two runs give the same checksum.
- One P1.3 checkpoint measured with the P1.3, P1.4 and P1.5 code (recovery off) gives the same checksum, and one P1.5 checkpoint with the P1.5 and P1.6 code too. So those additions leave the dynamics unchanged.
- With its default flags, P1.6 trains the same weights as P1.5, bit for bit.
- `go2_eval.py` options work (run headless; P1.6 also with the viewer, under a virtual display).
- The patches apply to your files in sequence and reproduce the snapshots.

**Not checked:** training to convergence (that needs your GPU), and anything on the real robot.

## 0. Genesis 1.4.2 facts behind the P1.2 code

1. **`set_mass_shift` and `set_COM_shift` do not exist in 1.4.2.** The online DR guide still shows them.
   - Use `robot.set_links_mass()` and `robot.set_links_COM()` instead. They take absolute values.
   - Per-env values need `RigidOptions(batch_links_info=True)` (`rigid_entity.py`).
2. **Contact friction is `max(mu_a * ratio_a, mu_b * ratio_b)`, and every URDF geom has mu = 1** (`collider/contact.py`).
   - So `robot.set_friction_ratio(0.2)` alone changes nothing; the ground's ratio has to be scaled too.
   - Test here, 0.5 m/s side push on a standing robot, foot slide after 1 s:
     - mu = 1: 0.9 cm
     - robot ratio 0.2 only: **0.9 cm**
     - robot and ground ratio 0.2: **3.4 cm**
3. **The URDF morph adds `default_armature=0.1` kg m² to every revolute joint** (`options/morphs.py`). MuJoCo Menagerie's Go2 model uses 0.01.
4. **The `go2.urdf` bundled with Genesis has Go1-like calf limits.**
   - Bundled file: 35.55 Nm and 20.07 rad/s. Unitree's `go2_description` URDF: 45.43 Nm and 15.70 rad/s.
   - Genesis loads the URDF effort limit into the force range and clamps every applied force to it.
   - It does not read velocity limits at all.
5. **Setters given a boolean `envs_idx` want values for the selected envs only** (`set_friction_ratio`, `set_links_mass`, `set_links_COM`).
   - That is a data-dependent shape, so a GPU sync on every call.
   - So friction, mass and CoM are drawn once per env, after `build()`. The Genesis DR guide and legged_gym do the same.
   - Pushes hit all envs at once, as in legged_gym.
6. **Genesis' internal PD (`control_dofs_position`) treats its Kd term implicitly** under the default `approximate_implicitfast` integrator (`abd/forward_dynamics.py`).
   - The explicit PD used from P1.2 on loses that.
   - At 200 Hz with armature 0.01 it stayed stable in every check here.

## P1.1: stand still at zero command

**go2_train.py**
- Command ranges change from a single command (`vx 0.5, vy 0, wz 0`) to `vx [-1, 1]`, `vy [-0.5, 0.5]`, `wz [-1, 1]`.
  - The old policy only ever saw one command.
  - The new ranges cover the `go2_metrics.py` grid, whose `ood` column is now empty.
- `lin_deadzone` and `ang_deadzone` are 0.2, and `rel_standing_envs` is 0.1. About 12% of sampled commands are then an exact (0, 0, 0) (measured).
- `feet_air_time: 1.0`, with a target air time of 0.5 s and a contact threshold of 1 N (legged_gym).
- `stand_still_scale: 5.0` and `stand_still_vel_threshold: 0.5` (Isaac Lab Spot).
- `tracking_ang_vel` goes from 0.2 to 0.5 (legged_gym), because yaw commands are no longer always zero.
- Other changes:
  - `foot_links` added to the config.
  - `--max-iterations` default goes from 101 to 1000.
  - `--cpu` added for smoke tests.
  - `run_manifest()` is now called. It was defined but never used. Outside git it only prints a warning.

**go2_env.py**
- `_resample_commands` applies the deadzone, then gives the exact-zero command to the standing envs.
- The foot links are always kept (`links_to_keep`).
- `_update_feet_contacts()` runs each step, as in legged_gym:
  - contact when |F| > 1 N, OR-ed with the previous step;
  - air time counted in policy steps.
- `_reward_feet_air_time` pays (air time − 0.5 s) at each touchdown, and **nothing when the command is zero**.
  - The gate uses the full command, where legged_gym uses |v_xy| only.
  - That way, turning in place still gets proper steps.
- `_reward_similar_to_default` is multiplied by 5 when the command is zero **and** the base moves slower than 0.5 m/s (Isaac Lab Spot `joint_position_penalty`).
  - I chose this over legged_gym's `_reward_stand_still` for its speed gate.
  - After a push (P1.2), the robot can step to recover without paying the ×5 penalty.

**go2_commands.py** (new)
- `apply_command_deadzone()` is used in training.
- `sticks_to_command()` is for the robot side:
  1. stick deadzone of 0.1, rescaled so the command stays continuous;
  2. scaling to the training ranges;
  3. the same 0.2 zeroing as in training.
- So the robot never receives a command the policy never saw.
- No Genesis import: copy it to the deployment PC as is.

**go2_eval.py**
- `--cmd VX VY WZ` pins the command (`--cmd 0 0 0` to check standing). The training deadzone still applies.

**go2_metrics.py**
- Unchanged.
- P1.1 passes when the `(0, 0, 0)` row shows `step_freq` near 0 and `duty` near 1.

## P1.2: domain randomization and actuation fidelity

**go2_env.py, actuation**
- Physics goes from 100 Hz to 200 Hz: `SimOptions(dt=0.005, substeps=1)` plus a decimation loop of 4 in `step()`.
  - `self.dt` stays the policy period (0.02 s), so rewards, episode length and metrics keep their meaning.
- The PD moves out of Genesis and into `step()`, as legged_gym `_compute_torques`:
  - `tau = strength * (Kp * (q* - q) - Kd * qd)`, computed every physics step;
  - clipped at the Go2 effort limits;
  - applied with `control_dofs_force()`.
- What the explicit PD makes possible:
  - per-env gains, motor strength and clipping, without `batch_dofs_info`;
  - exact applied torques (`env.torques`) for P1.3.
- The Genesis force range is set to the same limits, so `go2_metrics.py` reads 45.4 Nm at the calf.
- Latency: the previous target is held for the first k physics steps of each policy step.
  - k is drawn from {0..4} per env and episode, i.e. 0–20 ms in 5 ms steps.
  - k = 4 is exactly the old fixed one-step delay.
- `default_armature=0.01` on the URDF morph (Menagerie value).

**go2_env.py, randomization** (`domain_rand` in env_cfg; empty means off)
- Drawn once per env after `build()`:
  - friction 0.5–1.25, on the robot **and the ground** (see §0.2);
  - base mass −1 to +2 kg;
  - base CoM ±3 cm on each axis.
  - `batch_links_info=True` is switched on only when mass or CoM are randomized.
- Redrawn at every reset:
  - Kp and Kd ×0.85–1.15, per env;
  - motor strength ×0.9–1.1, per env and joint;
  - latency.
- Observation noise at legged_gym's scales: ang_vel 0.2, gravity 0.05, dof_pos 0.01, dof_vel 1.5 (physical units, uniform).
- Pushes every 15 s on all envs: U(−1, 1) m/s added to the base x and y velocity.
  - Added, as Isaac Lab does, rather than overwritten, as legged_gym does.
  - That makes the push an impulse instead of a velocity reset.
- New `_reward_orientation` (legged_gym), and `nominal_world()` for evaluation.

**go2_train.py**
- The values above. `latency_s: 0.02` replaces `simulate_action_latency`, which the env never read.
- The tilt termination goes from 10° to 30°, with `orientation: -2.5` (Isaac Lab Go2 flat).
  - Measured here with the PD stance at zero action:
    - a 0.5 m/s side push already tilts the robot 9.6°;
    - a 1 m/s push tips it over;
    - the drop at reset reaches 10.4° in the new physics (6.5° in the old).
  - At 10°, push episodes would end before any recovery step could be learned.
  - The orientation penalty takes over the "keep the base level" role.

**go2_eval.py and go2_metrics.py**
- Both now run a **nominal world** by default: no randomization, no noise, no pushes, and a fixed latency of 20 ms (`--latency_ms` in metrics, `--latency-ms` in the viewer).
  - Without this, `go2_metrics.py` would load each run's `domain_rand` from `cfgs.pkl`, and every run would be measured under different conditions.
- The physics (200 Hz, explicit PD, armature 0.01, Go2 limits) comes from the code, not the run's config.
  - Older runs (baseline, P1.1) are therefore measured in the same world.
  - Re-run your baseline once for a fresh reference.
- `--keep_dr` (metrics) and `--dr` (viewer) keep the training randomization. Use `--measure_s 20` so a push happens during the measurement.
- `--fall_deg 10`: a fall is a tilt over 10°, whatever the run's training termination, so falls stay comparable with your existing tables.
- When mass is randomized, `get_mass()` returns one value per env, which made the old `float(...)` call crash. CoT now uses the per-env mass.

**What to check**
- Compare the training curves with P1.1: `Mean reward`, `Mean episode length`, `rew_tracking_lin_vel`.
- Expect slower iterations: 4 solver steps per policy step instead of 1.
- If training doesn't converge, start with narrower ranges (friction, latency, pushes) and widen them.

## P1.3: safety and limits

**go2_env.py**
- Contact forces are read once per step (`link_contact_force`) and shared by feet, collisions and termination.
- Link groups are matched by name substring, as in legged_gym:
  - `penalize_contacts_on = [thigh, calf, base]`
  - `terminate_after_contacts_on = [base]`
- Soft limits:
  - 90% of the URDF position range, centered;
  - 90% of the Go2 velocity limits (14.1 rad/s at the calf).
- New rewards, using legged_gym's functions:
  - `ang_vel_xy`, `torques`, `dof_acc`, `dof_pos_limits`;
  - `dof_vel_limits`, with the excess clipped at 1 rad/s per joint;
  - `collision`, counting links with |F| > 0.1 N.
  - `dof_vel` (L2) is defined but not enabled (legged_gym default 0).
- Termination on base contact, |F| > 1 N (legged_gym `check_termination`). The 30° tilt limit stays as a backstop.
- `only_positive_rewards` (legged_gym) clips the step reward at 0.
  - With many penalties and contact termination, a negative total would make an early end of episode attractive.

**go2_train.py**
- Weights are in the table below.
- `dof_vel_limits: -1.0` is my choice; legged_gym has no default for it.
- If torques or CoT stay high, note that unitree_rl_gym's Go2 uses `torques: -2e-4`, 20× legged_gym.

**go2_metrics.py**
- Unchanged. Base contact now also counts as a fall.

## P1.4: smooth actions

**go2_env.py**
- New reward `action_smoothness`: the squared second difference of the joint targets, (a_t − 2a_{t−1} + a_{t−2}) × action_scale.
  - It is walk-these-ways' `action_smoothness_2`, including its masking of the first two steps after a reset (the zeroed action history is not a real change).
  - Why a second difference next to `action_rate`: for a sinusoid of frequency f, its square is 4 sin²(π f dt) times the first difference's.
  - That is 4× at 25 Hz (an action flipping every step) and about 1/16 at a 2 Hz gait. So it targets jitter, not the gait.
- `last_last_actions` buffer: shifted at the end of `step()`, zeroed at reset.
- `torques_substeps`: the torques of the 4 physics steps of each policy step, for the 200 Hz analysis in the harness. It does not change the simulation.
- `obs_noise_scales()` moved out of the class (same values), so `go2_train.py` can reuse it for CAPS.

**go2_train.py**
- `action_smoothness: -0.1`: walk-these-ways' weight, on joint targets as there (their action_scale is also 0.25).
- Flags `--lcp`, `--caps-t` and `--caps-s` (all off by default) switch PPO to `go2_smooth_ppo.SmoothPPO`. The choice is saved in `cfgs.pkl`, so eval and metrics rebuild the same class.

**go2_smooth_ppo.py** (new, used only with those flags)
- LCP gradient penalty: E[‖∇ₛ log π(a|s)‖²] on the PPO mini-batch, as in their code (`ppo_rma.py`, `_calc_grad_penalty`). They use 0.002 for every robot.
- CAPS:
  - temporal L_T = E‖μ(s_t) − μ(s_{t+1})‖, on random consecutive steps of the rollout from the same episode;
  - spatial L_S = E‖μ(s) − μ(s + ε)‖, with ε ~ N(0, σ) and σ = the P1.2 observation-noise scales (CAPS: "based on expected measurement noise").
  - Their PPO setting for Ant: λ_T = 0.1, λ_S = 0.5.
- `update()` is a copy of rsl_rl 5.5.1 `PPO.update()` with three marked insertions; I diffed it against the installed version. It warns if your rsl_rl version differs.
- Cost of the update step on a CPU smoke run: 1.4× (LCP) and 2.1× (CAPS). On your GPU, data collection dominates the iteration time.

**go2_metrics.py**
- The table "Smoothness & actuation" is split into "Smoothness" and "Actuation limits". The CSV keeps its column names and gains new ones.
- New columns:
  - `act_acc[rad/s2]`: RMS second difference of the joint targets / dt², i.e. what the new reward penalizes.
  - `tau_rms[Nm]`: RMS torque over joints and time. Motor Joule losses go with current², i.e. torque².
  - `tau>10Hz[Nm]`: RMS of the torque content above 10 Hz, from the 200 Hz signal. A torque sampled once per policy step would fold PD chatter above 25 Hz back into the band below it.
- New plot `spectra_ckpt*_seed*.png`: power spectral density of the joint targets (0–25 Hz) and of the torques (0–100 Hz). It covers the commands (0, 0, 0), (0.5, 0, 0), (1.0, 0, 0) and (0, 0, 0.5). The gait sits left of the 10 Hz line; jitter shows up right of it.

**How to decide on LCP / CAPS**
- Compare a P1.3 and a P1.4 run (same seed, same checkpoint iteration) in the "Smoothness" table and the spectra plot. The content above 10 Hz should drop, with no loss in tracking.
- If it does not, or if the robot buzzes or its motors heat up, retrain with `--lcp 0.002` first (one coefficient, nothing else to tune). Then try CAPS: `--caps-t 0.1 --caps-s 0.5`.

## P1.5: stand-up and sit-down

New files:
- `go2_fsm.py`: the state machine, robot side (numpy and torch only).
- `go2_export.py`: exports the policy for the robot.
- `go2_fsm_sim.py`: Genesis test bench for the state machine.
- `go2_deploy_sdk2.py`: adapter to the robot (unitree_sdk2_python).

`go2_env.py` and `go2_train.py` only gain the optional learned recovery (last part of this section).

**State machine (`go2_fsm.py`)**

| mode | motor command | duration | source |
|---|---|---|---|
| passive (start, e-stop) | kp 0, kd 3 | – | unitree_rl_lab Go2 `Passive` |
| stand_up, from passive | current → lying pose (0, 1.36, −2.65) → default pose; kp 60/80/80, kd 5/4/4 (hip/thigh/calf) | 1 s + 1 s, then hold | unitree_rl_lab Go2 `FixStand` |
| stand_up, from policy | current → default pose, same gains | 1 s | |
| policy | gain ramp from the stand gains to kp 20, kd 0.5 at the default pose, then the policy at 50 Hz | ramp 0.5 s | ramp: my choice |
| sit_down | current → lying pose, stand gains; hold, then passive | 2 s + 0.5 s | lying pose: go2_stand_example |

- Interpolation is minimum-jerk instead of unitree_rl_lab's linear one: zero velocity and acceleration at every waypoint, so no velocity step at segment ends with 60–80 Nm/rad gains. The path velocity is sent as `dq`, so kd damps around the motion instead of against it.
- Gamepad, go2_deploy's mapping: L1+R2 stand up, L1+A policy (only once standing), L1+R1 sit down, L1+Y damping.
- Sticks, as unitree_rl_gym `deploy_real`: vx = ly, vy = −lx, wz = −rx. Then `go2_commands.sticks_to_command()`: stick deadzone 0.1, then the 0.2 training deadzone.
- In policy mode, damping if the tilt exceeds 1 rad (unitree_rl_lab `bad_orientation(1.0)`) or a joint reads 0.2 rad outside its URDF range.
- The policy restarts with last action = 0, as at the start of a training episode.

**`go2_export.py`**
- Writes the TorchScript actor (rsl_rl `export_policy_to_jit`) and `deploy_cfg.json`: joint order, default pose, gains, scales, command ranges and deadzones, torque and joint limits.
- Checks the exported network against the training policy on random observations (identical here).

**`go2_fsm_sim.py`: measured here**

Setup: zero-action policy (the "policy" holds the default pose with kp 20), nominal world, 1 robot unless stated.

| check | result |
|---|---|
| stand-up | lying z 0.076 m → 0.318 m; peak torque 25 % of the limit; tilt ≤ 6° |
| hand-over, 0.5 s ramp | base drops 4.9 cm; tilt ≤ 2.7°; torque ≤ 17 % |
| hand-over, direct switch (`--handover-s 0`) | base drops 5.1 cm; tilt ≤ 4.2°; torque ≤ 20 % |
| sit-down | sinks at ≤ 17 cm/s; peak torque 36 % |
| e-stop from standing (damping, kd 3) | sinks at ≤ 54 cm/s; on the ground within 1 s |
| 16 robots, training randomization (`-n 16 --dr`) | 16/16 standing after stand-up; peak torque 33 % |
| deploy observation vs Go2Env observation | max difference 1.2e-7 (float32 rounding) |

- The observation check confirms that joint order, scales and the gravity vector match training.
- The hand-over drop is pessimistic: a trained policy compensates for the lower gains, the zero-action policy cannot.
- Rerun the bench with your own policy: `go2_export.py`, then `go2_fsm_sim.py -e <exp>`.

**`go2_deploy_sdk2.py` (adapter to the robot, NOT tested on the robot)**
- It mirrors Unitree's `go2_stand_example.py`:
  - LowCmd initialisation and CRC;
  - `rt/lowcmd` and `rt/lowstate`;
  - a 2 ms `RecurrentThread`;
  - sport-mode release through `MotionSwitcherClient`.
- Safety: damping if no state arrives for 50 ms, 1 s of damping on Ctrl+C, and damping on any error in the state machine.
- Tested here only over DDS loopback, against a fake robot publishing `rt/lowstate` with scripted gamepad presses.
  - Checked: CRC, gains in each mode, the gain ramp, the watchdog, stand-up and sit-down, Ctrl+C, unused motor slots left untouched.
  - Not checked: real motors, IMU, gamepad, sport-mode release.
  - The test used Python 3.11 and cyclonedds 11.0.1; Unitree's install pins cyclonedds 0.10.2.
- First runs, in this order:
  1. unitree_mujoco (`--sim`: domain 1, interface `lo`);
  2. the robot hung on a stand;
  3. the robot on the ground, finger on L1+Y.
- The robot computer needs unitree_sdk2_python, numpy, torch, `go2_fsm.py`, `go2_commands.py` and the exported folder.

**Learned recovery (optional, W12 in your plan): `python go2_train.py --recovery`**
- 15 % of the resets, drawn per env, start from a random fallen state:
  - base dropped from 0.5 m with a uniformly random orientation;
  - joint angles uniform within the soft limits.
  - Basis: MetalHead (`recovery_init_prob` 0.15, random Euler angles, `terminate_after_contacts_on` empty).
- In those episodes:
  - the command is 0;
  - there is no tilt or base-contact termination;
  - the step reward is not clipped at 0. These episodes cannot end early, so the penalties stay informative.
- Rewards, paid in recovery episodes only (weights are my starting point):
  - `rec_upright` 1.0: (1 − g_z) / 2;
  - `rec_height` 1.0: base height / target, × upright;
  - `rec_pose` 0.5: exp(−‖q − q_default‖²), × upright.
- Logged as `Episode/recovery_success`: the share of recovery episodes that end upright with the base above 0.2 m.
- Off by default; evaluation (`nominal_world`) and the bench always turn it off.
- Checked here: fallen starts, gating, logging, 3 training iterations. Not trained, so expect to tune the weights.

## P1.6: body-height command, keyboard control, recurrent policy

All three could be done properly, as opt-in additions. Without `--height-cmd` and `--rnn`, a P1.6 run is a P1.5 run: same weights, same checksum. One combination is refused rather than half-supported: the P1.4 regularizers (`--lcp`, `--caps-*`) with an RNN. New files: `go2_teleop.py` and `go2_compare.py`.

```
python go2_train.py -e go2-p16h --height-cmd                  # 4th command: body-height offset dh
python go2_train.py -e go2-p16r --rnn lstm                    # LSTM actor and critic (or --rnn gru)
python go2_train.py -e go2-p16hr --height-cmd --rnn lstm      # both
python go2_eval.py  -e go2-p16h --ckpt 999 --cmd 0.5 0 0 -0.05   # walk 5 cm lower than nominal
python go2_teleop.py -e go2-p16h --ckpt 999                   # keyboard control in the viewer
python go2_teleop.py -e go2-p16h --ckpt 999 --headless --cmd 0.5 0 0 -0.04 --duration 10 --record crouch.mp4
```

### Your pre-summer version: what went wrong, and what P1.6 does instead

These are the reasons the obsolete files could not be ported as they were. Each item names the problem and the fix.

1. **The height command was an absolute base height, 0.25 to 0.45 m.**
   - The Go2 cannot stand at 0.45 m. With the calves at their URDF limit (−0.84 rad) and the feet straight under the hips, the base is at most 0.41 m above the ground.
   - With the feet where the default pose puts them, the front calves reach their soft limit (where `dof_pos_limits` starts to penalize) 6.9 cm above the default stance, at about 0.40 m, and their URDF limit at +7.8 cm (rigid legs).
   - So about a quarter of the range (0.40 to 0.45 m) asked for heights the legs cannot give, and the height reward pushed the legs into their joint limits.
   - Now: an offset `dh` from `base_height_target` (0.30 m), as walk-these-ways' `body_height_cmd`, sampled in [−8, +4] cm. +4 cm keeps a margin to the soft limit.
2. **`base_height` pulled toward the command while `similar_to_default` pulled toward the fixed default pose.**
   - The two terms disagree at every height except the default one, so the policy settles in between and the height tracking is biased.
   - Now: `similar_to_default` measures the distance to the leg pose of the commanded height (IK, below). At dh = 0 that is exactly the default pose.
3. **The reward was computed after the reset, and the command resampled before the reward** (step order of the old Genesis example).
   - The last step of each episode was scored in the reset state, and a step where the command changed was scored against the new command.
   - Your current base already fixes this (reward, then resampling, then termination and reset, then observation). Kept.
4. **The last action was removed from the observation** (`num_obs` 34).
   - With one step of latency, the robot executes the previous action, so the policy no longer saw the target its motors were tracking.
   - `action_rate` then penalizes a change the policy cannot see. An RNN can partly infer it; an MLP cannot.
   - Now: kept in the observation, on both sides (`go2_fsm.Policy` builds it on the robot).
5. **A 1280×960 camera was added in `Go2Env.__init__`.**
   - Every scene got it, training included. On a headless GPU server, it needs an OpenGL/EGL context for nothing.
   - Now: a hook, `_add_extra_entities()`. Only `go2_teleop.py --record` adds a camera.
6. **The teleop ran a tkinter window in a thread and wrote `env.commands` from the main loop.**
   - Keys only reached the Tk window, a second window to keep in focus. Tk is not thread-safe, and macOS requires it in the main thread.
   - `_resample_commands` still ran inside `env.step()`, so every 4 s and after each reset the policy saw one random command. Your eval script re-applied its command after `env.step()`, which had already built the observation: same problem.
   - The command was written after `policy(obs)`, so every change reached the policy one step late.
   - Now: Genesis' own keybindings in the viewer (no thread, no second window). The command is pinned inside the env and never resampled. The observation is recomputed as soon as the command changes.
7. **The recording used the pre-1.0 camera API** (`start_recording()`, `render()` every step, `stop_recording(save_to_filename=..., fps=...)`).
   - In 1.4.2, `stop_recording()` takes no argument (TypeError). The file name and fps go to `start_recording()`.
   - The camera now renders itself as the scene steps and streams the frames to the file, so the extra `render()` only costs time.
   - Now: `start_recording(save_to_filename, fps=50)` gives one frame per policy step. `stop_recording()` runs in a `finally`, also on Ctrl+C.
8. **The RNN config was written for rsl_rl 2.x** (`policy_class_name: ActorCriticRecurrent`, `rnn_hidden_size`), and the eval loops never reset the memory.
   - rsl-rl-lib 5.x configures one model per `actor` and `critic` (`RNNModel`), so that config does not load anymore.
   - Without `reset(dones)`, a robot that fell restarts with the memory of its fall.
   - There was no way to carry the memory to the robot.
   - The config comment was also wrong: the RNN size does not have to match the first MLP layer. The MLP takes the RNN output, whatever its size.
   - Now: `--rnn lstm|gru` builds `RNNModel`. `policy.reset(dones)` runs in eval, metrics and teleop. The TorchScript export keeps the memory and has `reset()`, and the robot-side state machine calls it at each policy start.
9. **`num_mini_batches: 2  # reduce to 2 if hitting OOM`**
   - It works the other way round: fewer mini-batches means bigger ones, so more memory in the update. To save memory, raise `num_mini_batches` or lower `-b`.
   - Unchanged (4).
10. **`.nonzero()` in `step()` and `.item()` in `reset_idx()`** (old example).
    - Each call forces a GPU-to-CPU sync, every step.
    - Already gone in your current base (boolean masks and `torch.where`).

Reviewed, not carried over:
- **kp 40, kd 1** (instead of 20, 0.5). That is a different actuator, not a fix: the whole policy changes with it. It stays a knob (`env_cfg["kp"]`, `["kd"]`), and it is exported in `deploy_cfg.json`, so the robot side follows.
- **The "memory leakage fix"** (preallocated buffers, `[:]` copies). Reassigning a tensor frees the old one; nothing leaks unless something keeps a reference. If memory grows in a long run, log `torch.cuda.memory_allocated()` per iteration first.
- **vx ±2 m/s, vy ±1 m/s.** Ranges that wide need a command curriculum (legged_gym `update_command_curriculum`). That is P2 material.
- **64 steps per env with the RNN.** Now a flag: `--steps-per-env 64`. The default is 48 with an RNN.

In the files you sent, nothing else differs from base rsl_rl: the differences are configuration values (items 8 and 9, the RNN settings, `num_steps_per_env`). If you also edited rsl_rl's own source (e.g. `ppo.py`), that file was not among them.

### What changed, per file

**go2_env.py**
- Optional 4th command `dh` [m]: an offset from `base_height_target`. On when `command_cfg["num_commands"]` is 4.
  - The command scales and limits are built from lists, so 3 and 4 commands share one code path.
  - The P1.1 logic only looks at (vx, vy, wz): `zero_cmd`, the deadzone, and `rel_standing_envs`. A standing env keeps its height offset, since standing still while crouched is a valid command.
  - `base_height` targets `base_height_target + dh` (walk-these-ways `_reward_jump`).
- `nominal_dof_pos(dh)`: the joint pose that moves the base by `dh`, with each foot where the default pose puts it.
  - Planar 2-link IK on thigh and calf, hip at its default. Segment lengths are 0.213 m (URDF), checked against Genesis' link positions.
  - `similar_to_default` uses it as its reference, instead of the fixed default pose.
  - Out of reach, the calf angle saturates instead of returning NaN.

  | dh | front thigh / calf [rad] | rear thigh / calf [rad] |
  |---|---|---|
  | −12 cm | 1.18 / −2.21 | 1.49 / −2.18 |
  | −8 cm | 1.06 / −1.99 | 1.32 / −1.97 |
  | −4 cm | 0.94 / −1.76 | 1.16 / −1.75 |
  | 0 | 0.80 / −1.50 (default) | 1.00 / −1.50 (default) |
  | +4 cm | 0.64 / −1.20 | 0.83 / −1.21 |
  | +6.9 cm | front calf at the soft limit (−0.93): `dof_pos_limits` starts to penalize | |
  | +7.8 cm | front calf at the URDF limit (−0.84) | |

- `obs_noise_scales(obs_cfg, num_actions, num_commands)`: the noise vector follows the observation size (46 with `dh`). Without this, training with `dh` crashed on a 46 vs 45 size mismatch.
- `_add_extra_entities()`: an empty hook, called just before `scene.build()`.
- The viewer camera pose comes from `env_cfg["viewer_camera_pos"]` and `["viewer_camera_lookat"]`, with the old values as defaults.
- Fix of the P1.5 log `Episode/recovery_success` (see "Fixes" below).

**go2_train.py**
- `--height-cmd`: `num_commands` 4, `body_height_range` [−0.08, 0.04] m, `obs_scales["body_height"]` 5.0.
  - The scale maps the range to ±0.4 in the observation, within the span of the other command entries (±0.25 to ±2).
  - Walk-these-ways uses a scale of 2 for a range about three times wider.
- `--rnn lstm|gru` and `--rnn-hidden` (default 256, rsl_rl's default):
  - actor and critic become rsl_rl `RNNModel`: a one-layer RNN on the observation, then the same [512, 256, 128] MLP;
  - each network has its own memory.
- `--steps-per-env`: 24 for an MLP (unchanged), 48 by default with an RNN.
  - rsl_rl trains the RNN by backpropagation through time, cut at the rollout: each mini-batch trajectory starts from the memory saved during the rollout, and ends at a reset or at the end of the rollout.
  - So the rollout length bounds how far back a gradient reaches: 48 steps = 0.96 s.
- `--rnn` with `--lcp` or `--caps-*` is refused.
  - `go2_smooth_ppo.py` evaluates the policy on flat mini-batches of single observations. A recurrent version would need rsl_rl's padded-sequence path with hidden states.
  - Refusing is better than training something silently wrong.

**go2_eval.py**
- `--cmd VX VY WZ [DH]`. Giving DH to a run without the height command is an error, with a message saying why. Three values on a 4-command run mean dh = 0.
- `--fallen`: every episode starts from a random fallen pose, to watch a `--recovery` run (P1.5).
- `policy.reset(dones)` after each step: the memory of a reset env is cleared (no-op for an MLP).

**go2_metrics.py**
- 4-command runs get 4 more bins: (0, 0, 0) and (0.5, 0, 0) at dh = −6 and +3 cm. The other bins run at dh = 0.
- A `dh` column in the tables, the CSV and the gait plot.
- "Tracking & stability" gains `z[m]` (mean base height) and `z_err[cm]` (mean |z − (0.30 + dh)|), for every run.
- `policy.reset(dones)` in the rollout.
- `--tag NAME` adds `_NAME` to the output file names, so a second evaluation of the same checkpoint (e.g. `--keep_dr`) does not overwrite the first.

**go2_export.py**
- `deploy_cfg.json` gains `num_commands`, `command_ranges.body_height`, `base_height_target` and `recurrent`.
- For a recurrent policy, rsl_rl exports a batch-1 TorchScript module that keeps the LSTM/GRU state between calls and has `reset()`.
  - The check resets both networks and compares 100 consecutive steps.
  - The MLP export has an empty `reset()`, so the robot side can always call it.

**go2_fsm.py, go2_deploy_sdk2.py, go2_fsm_sim.py**
- `Policy.reset()` also clears the network memory, at each policy start.
- D-pad up and down: `dh` ±1 cm in policy mode, clamped to the exported range, back to 0 at each policy start. The status line of `go2_deploy_sdk2.py` shows it.
- Older exports (no `num_commands` in `deploy_cfg.json`) load as 3-command policies.
- The bench's `full` scenario presses D-pad down three times (8.6 to 9.0 s). 3-command runs ignore it.

**go2_commands.py**
- Docstring only: `apply_command_deadzone()` also takes 4 commands and leaves dh untouched.

**go2_teleop.py** (new)
- Keyboard control in the Genesis viewer, through Genesis 1.4.2's keybinding API (`Keybind`, `KeyAction.PRESS`, `viewer.register_keybinds`). Genesis' own keys stay as they are.

  | keys | command |
  |---|---|
  | Up / Down | vx ±0.1 m/s |
  | Left / Right | vy ±0.1 m/s (left = +) |
  | Q / E | wz ±0.1 rad/s (Q = turn left) |
  | T / G | dh ±1 cm (runs trained with `--height-cmd`) |
  | Space | vx = vy = wz = 0 (keeps dh) |
  | Backspace | reset the robot and the policy's memory |

- The command is clamped to the run's training ranges, then the training deadzone applies, as on the robot. The terminal shows both values, and says when a key hits a training limit.
- `TeleopGo2Env` pins the command (also after a fall) and adds the recording camera through the hook.
- The viewer follows the robot from a chase position. `--record` adds a camera that follows it too, and streams 1280×720 at 50 fps, i.e. one frame per policy step.
- `--headless --duration --cmd` records without a display. On Linux without a display, prefix the command with `PYOPENGL_PLATFORM=egl`.
- Nominal world by default (`--dr` for the training randomization), never a fallen start, no time-out resets. A fall resets the robot and the policy's memory; the fall count is printed at the end.

**go2_compare.py** (new)
- Reads the metrics CSVs of several runs and prints one table per metric, with the change from the first run in % (or as a difference), and a `~` when the change is within 2 standard errors.
- Rows are matched on the command, so 3-command and 4-command runs compare on their dh = 0 rows. No Genesis needed.

### Fixes to earlier phases, found while testing P1.6

1. **`Episode/recovery_success` read far too low (P1.5).** rsl_rl averages a logged value over every step of the iteration, and P1.5 logged 0 at the steps where no recovery episode ended. Now the env keeps decayed counts (×0.99 per step, a window of about 100 steps) and logs their ratio at every step.
2. **`go2_metrics.py --keep_dr` on a `--recovery` run (P1.5).** About 15 % of the envs were flagged as recovery episodes, which a fall never ends: their falls went uncounted, and their data kept counting after a fall. (The fallen pose itself was undone by the harness' joint-noise reset, so it was not visible.) On a smoke run: 10 falls instead of 14, out of 22 envs. Now the harness never makes fallen starts.
3. **`go2_eval.py --dr` on a `--recovery` run** started 15 % of the episodes fallen. Now only `--fallen` does.
4. **`--ckpt 1000`** in the previous version of this file: there is no such checkpoint (see "How to use this bundle").

### Checked here

- Default flags: P1.5 and P1.6 train bit-identical networks and optimizer states (68 tensors; 3 iterations, CPU, 32 envs, seed 1). One P1.5 checkpoint measures the same checksum with the P1.5 and the P1.6 code (−5.0413971241e+01).
- IK:
  - `nominal_dof_pos(0)` equals the default pose (1.2e-7 rad), and forward kinematics moves the feet by exactly −dh relative to the base (5e-8 m).
  - Holding the IK poses with stiff gains (kp 70, kd 4) moves the base by −8.4, −4.2, +2.1 and +4.3 cm for dh = −8, −4, +2 and +4 cm.
  - With the policy gains (kp 20), the default pose alone sags to 0.27 m. The policy already has to add the difference to reach 0.30 m, at every height.
- Training (3 iterations, CPU, 64 envs): `--height-cmd`, `--height-cmd --rnn lstm`, `--rnn gru`, `--recovery`. `--rnn lstm --lcp 0.002` is refused.
- Metrics: the height bins and the `dh` column; the LSTM run gives the same checksum twice; `--tag` writes separate files; the fix of `--keep_dr` on a `--recovery` run (above).
- Export:
  - LSTM and GRU: zero difference over 100 steps; the output depends on the memory, and `reset()` restores the first output exactly;
  - MLP with the height command: no memory, empty `reset()`.
- FSM bench: LSTM + height and MLP + height: observation match 1.79e-7 and the three D-pad log lines; GRU without height: observation match 1.79e-7, D-pad ignored.
- `go2_eval.py` with the viewer, under a virtual X display (Xvfb): LSTM + height with `--cmd 0.5 0 0 -0.03`, GRU with `--fallen`.
- Teleop:
  - headless recording (EGL): 1280×720, 50 fps, 150 frames for 3 s;
  - in the viewer under Xvfb, with real key events (XTest): every key changed the command as documented, including the deadzone display and the clamp at +4 cm; Space and Backspace too;
  - viewer and recording together: 100 frames for 2 s;
  - Ctrl+C: the video is finished (279 frames for 5.6 s);
  - closing the viewer window (the close event a window manager sends): clean exit, video finished (86 frames for 1.7 s);
  - the keys register next to Genesis' default keys without conflict;
  - a 3-command MLP run, with `--dr`;
  - a unit test of the key handling: clamps, deadzone, messages.
- `go2_compare.py` on 3- and 4-command CSVs.
- The fine-tune recipe of `TEST_AND_TINKER.md` (section 13): loading a checkpoint carries on its iteration count.

**Not checked:**
- Training to convergence. How well the height is tracked, and whether the LSTM beats the MLP, is for your GPU runs.
- The viewer on your own machine (tested under Xvfb only).
- The robot.

## P3.1: curriculum training (terrain tiles, terrain curriculum, command curriculum, slopes up to 25°)

P3.1 builds on the P2 code (`[P2.x]` tags; P2's notes are `TEST_AND_TINKER_P2.md`). Everything is opt-in. Without `--terrain` and `--cmd-curriculum`, a P3.1 run is a P2 run, with the same weights bit for bit, and every evaluation script still measures on flat ground by default. The policy stays blind: the observation is unchanged (45, 46 or 54 entries), so nothing changes on the robot side. New file: `go2_terrain.py`. Every change is tagged `[P3.1]`; `grep -n "\[P3" *.py` lists them. The companion guide is **`TEST_AND_TINKER_P3.md`**.

In the bundle, `p3_1/` is the complete, runnable snapshot (P2 files + P3.1). `patches/p3_1_curriculum.patch` holds the same change against your P2 files: in the folder that holds them, run `git apply p3_1_curriculum.patch` (or `patch -p1 < p3_1_curriculum.patch`). Line endings are LF; for CRLF files, use `git apply --ignore-whitespace`.

```
python go2_terrain.py                                        # default grid: tile table + terrain_grid.png (numpy only)
python go2_terrain.py --check --genesis                      # self-checks, and the heights against Genesis
python go2_train.py -e go2-p31 --terrain --cmd-curriculum    # tiles + terrain curriculum + command curriculum
python go2_train.py -e go2-p31i --terrain --cmd-curriculum --init-from logs/go2-p2/model_1999.pt   # from a P2 policy
python go2_metrics.py -e go2-p31 --ckpt 1999 --terrain grid  # one bin per (type, level) + success heat map
python go2_metrics.py -e go2-p31 --ckpt 1999 --slope 15      # the usual command grid on a 15° plane
python go2_compare.py logs/go2-p2/metrics_ckpt1999_seed0_grid.csv logs/go2-p31/metrics_ckpt1999_seed0_grid.csv --preset terrain
python go2_teleop.py -e go2-p31 --ckpt 1999 --terrain --tile hill 9    # J / K: level, U: type
python go2_fsm_sim.py -e go2-p31 --slope 15                  # the deploy state machine on a slope
```

### What P3.1 does, and why

1. **The terrain is one heightfield built by `go2_terrain.py`, not Genesis' own subterrains.**
   - In Genesis 1.4.2, `gs.morphs.Terrain(subterrain_types=..., subterrain_parameters=...)` takes one parameter set per terrain *type*, not per tile. So it cannot grade difficulty row by row, which a curriculum needs (legged_gym's `Terrain.curiculum()` builds row i at difficulty i / num_rows).
   - `go2_terrain.py` builds the whole grid in numpy (rows = levels, columns = types) and passes the heights as `Terrain(height_field=..., horizontal_scale=0.1, vertical_scale=0.005)`.
   - About `name=` caching: Genesis caches only terrains it generates itself (`height_field=None`), so it adds nothing for a given heightfield. `go2_terrain.py` has its own cache instead (`~/.cache/go2_terrain/go2_terrain_<key>.npz`; the key is a hash of the generation keys and `GENERATOR_VERSION`). Every run also saves the exact grid it trained on as `logs/<exp>/terrain.npz`. The Genesis entity is still named `go2_terrain_<key>`.
2. **Tile types** (default shares in brackets), from difficulty d = level / 9:

   | type | what | level 0 → level 9 | source |
   |---|---|---|---|
   | flat (10 %) | no relief at any level: keeps the flat gait in the mix | – | |
   | rough (20 %) | random heights on a 0.2 m grid, linear in between | ±0.5 cm → ±3 cm | legged_gym `random_uniform_terrain` |
   | hill (17.5 %) | pyramid, 4 planar faces, 2 m flat top; spawn on top, so walking away goes **downhill** | 0° → 25°, 1.40 m tall | legged_gym / Isaac Lab pyramid slopes |
   | pit (17.5 %) | the same pyramid upside down; walking away goes **uphill** | 0° → 25°, 1.40 m deep | idem, inverted |
   | wave (15 %) | A·sin(πkx/T)·sin(πky/T): whole half-waves per tile (4 m wavelength), so 0 on the edges | 0 → 15 cm, steepest 0° → 13.3° | Isaac Gym `wave_terrain` (bundled in Genesis, `ext/isaacgym/terrain_utils.py`) sums a sine and a cosine instead, which does not meet 0 at the edges |
   | obstacles (20 %) | 50 rectangles of 0.3–1 m, raised or sunk by h or h/2, flat 2 m centre | h 2 cm → 10 cm | legged_gym `discrete_obstacles_terrain` |

   - The pyramids have planar faces (height ∝ distance to the nearest edge), so a level-l hill is l × 2.8° everywhere on its faces (`go2_terrain.py --check` measures exactly that). legged_gym and Isaac Lab use bilinear pyramids, whose slope changes over a face.
   - Single inclines (Genesis `sloped_terrain`) are not used: a 25° incline across an 8 m tile rises 3.7 m and leaves a cliff to the next tile.
   - Every tile is exactly 0 on its outer ring of vertices, so any two neighbours join without a step.
3. **Slopes up to 25°.** That is the top level of hill and pit (`max_slope_deg`). Note: tan 25° = 0.47 is above the lowest friction of the P1 randomization (0.4). An env drawn with μ < 0.47 cannot stand still on a top-level face, whatever the policy. The terrain curriculum adapts per env, so these envs simply stay lower.
4. **Terrain curriculum** (Rudin et al. 2022, the game-inspired curriculum; legged_gym `_update_terrain_curriculum`). Judged per env when its episode ends:
   - Up one level if it walked more than half a tile (4 m) from its spawn point without falling.
   - Down one level if it fell, or if it covered less than half the distance its commands asked for (only when they asked for at least 1 m).
   - From the top level, up means a random level, which keeps every level in the mix.
   - Starting levels are uniform in 0–4.
   - One difference from legged_gym: legged_gym compares the distance walked with the *last* command × episode length. With commands resampled every 4 s, that judges a robot on a command it had for a few seconds. Here the commanded distance is integrated over the episode, and the progress is measured along the command.
5. **Command curriculum** (legged_gym `update_command_curriculum`, extended from vx alone to all three axes).
   - The ranges start at vx ±0.5, vy ±0.25, wz ±0.5.
   - Every 20 s of simulated time they grow by 0.1 / 0.05 / 0.1 per side, whenever the mean tracking reward of the episodes that ended meanwhile is above 0.8 × its maximum. The linear score drives vx and vy; the yaw score drives wz.
   - They stop at the configured ranges (±1, ±0.5, ±1).
   - The state reached is saved in every checkpoint (`infos["curriculum"]`). `go2_export.py` deploys the reached ranges, not the configured ones (see below).
6. **Terminations on terrain.**
   - With `--terrain`, the tilt limit becomes `max_slope_deg` + 30° = 55°: a base held parallel to a 25° face already pitches 25°, so the P1.2 limit of 30° would end good climbs. Base contact stays the real fall criterion; legged_gym's and Isaac Lab's rough-terrain tasks have no tilt termination at all.
   - A new termination ends an episode that walks off the border: base below the lowest ground − 1 m.
7. **Ground-relative quantities.** On terrain, `base_height` targets the base height above the mean ground under the feet (legged_gym: above the measured heights). `foot_clearance` measures each foot above the ground under it. The recovery terms and fallen starts are also measured from the ground. On flat ground all of these are exactly what they were.
8. **Spawning.**
   - Each robot spawns at its tile centre ±0.5 m, which keeps it inside the 2 m platforms, with a random heading.
   - Its base is 0.42 m above the highest ground under a 0.8 × 0.8 m footprint.
   - Env columns follow the type shares deterministically: env i takes the column at quantile (i + 0.5) / N.
   - On a `--slope` plane (evaluation, bench), robots spawn standing on it: base parallel to the plane, 0.42 m above it along its normal.
9. **A Genesis 1.4.2 build fix** (`go2_env.skip_terrain_support_field()`, `TERRAIN_FAST_BUILD = True`).
   - At `scene.build()`, Genesis 1.4.2 precomputes, for every geom, the furthest vertex in 33,480 directions. For the terrain that is 33,480 × every heightfield vertex, in 4 GB chunks: 35 s of a 45 s build for a 245k-vertex grid.
   - Terrain contacts never read that table; their support comes from the prism of each heightfield cell. Genesis 1.4.3 made the same change upstream.
   - So the fix gives the terrain an empty block and computes every other geom exactly as before. It is installed only when Genesis' function has the 1.4.2 source hash, and does nothing on 1.4.3+.
   - Checked: identical trajectories with and without it.
10. **5 m border instead of legged_gym's 25 m.** Genesis also processes the terrain mesh at build time (adjacency, normals), and that is what bounds the grid size. Measured on 2 CPU cores / 7 GB, kernels already compiled:
    - flat plane: 15 s to build the env, 1.5 GB peak;
    - 523k vertices (the default grid, 10 levels × 6 columns at 0.1 m, 5 m border): 35 s, 4.3 GB;
    - 1.06M vertices (20 m border): above 6.4 GB, killed.

    The off-grid termination (6) ends episodes that walk off the border.
11. **Resets with forward kinematics on terrain** (`env_cfg["reset_forward"]`; on for terrain, off on flat ground).
    - The reset writes the new state with `set_qpos(..., skip_forward=True)`, as in the Genesis example. That leaves `get_dofs_position()` and the link poses at their pre-reset values until the next physics step.
    - So the explicit PD of P1.2 computes its first substep from the old joint angles: up to the torque limit for 5 ms after every reset (see "Fixes" below).
    - New terrain runs reset with forward kinematics. Flat runs keep the old behaviour, so they stay identical to P2; `--set env.reset_forward=True` fixes it there too.

### What changed, per file

**go2_terrain.py** (new; numpy and torch, no Genesis)
- `TERRAIN_DEFAULTS`: every key of the grid (layout, tile parameters, spawning, curriculum). `terrain_config()` validates a config (unknown keys are an error) and fills in the defaults.
- `make_tile()` and the tile generators.
- Grounds with one interface (`height`, `gradient`, `spawn_height`, `describe`):
  - `FlatGround`;
  - `PlaneGround` (mode `"slope"`, a plane through the origin rising along +x);
  - `TerrainGrid` (mode `"grid"`), with `tile_origins` [L, C, 3], `tile_slope` / `tile_tilt` per tile, `column_of_type()`, `tile_stats()`, `save()` / `load()` / `cached()`.
- The height lookup follows Genesis' heightfield triangulation (vertex [i, j] at x0 + i·hs, y0 + j·hs; cells split along their (i+1, j)–(i, j+1) diagonal), in numpy and in torch. So it returns the surface the feet stand on.
- `make_ground(cfg, run_dir)`: flat, slope or grid. For a grid it reads `cfg["file"]`, else `run_dir/terrain.npz`, else the cache.
- CLI: tile table, `terrain_grid.png` (heights on a diverging blue–red scale), `--set KEY=VALUE`, `--run logs/<exp>`, `--check [--genesis]`.

**go2_env.py**
- The ground entity is the plane URDF (flat), `gs.morphs.Plane(normal=...)` (slope) or `gs.morphs.Terrain(height_field=...)` (grid). `self.plane` stays the ground entity whatever its shape, so friction randomization and `go2_fsm_sim.py --friction` work as before.
- New buffers:
  - `feet_ground` [N, 4] (ground height under each foot) and `base_height` [N] (base above the mean of those);
  - `max_pitch_deg` / `max_roll_deg` [N] (per-env tilt limits);
  - `fell` [N] (ended, not timed out).
- Terrain state: `spawn_pos` / `spawn_quat`, `terrain_levels` / `terrain_cols` / `terrain_types`, `cmd_dist` (commanded distance) and `progress` (distance covered along the command), both accumulated per episode.
- Methods:
  - `_terrain_assignment()`: column and initial level per env; overridden by `go2_metrics.py`;
  - `_spawn_poses()`;
  - `_update_terrain_curriculum()`;
  - `_init_command_curriculum()`, `_score_command_curriculum()`, `_update_command_curriculum()`: tensor operations only, no GPU–CPU sync;
  - `_curriculum_extras()`: the TensorBoard tags `Curriculum/*`;
  - `curriculum_state()` / `load_curriculum_state()`.
- The command curriculum switches itself off when the tracking rewards are absent. Evaluation scripts clear `reward_scales`, so they never run it.
- Resampling uses `cmd_limits`, the current curriculum ranges. They equal the configured ranges when the curriculum is off.
- On a slope, robots spawn standing on the plane: base parallel to it, at the initial height along its normal.
- `nominal_world()` also sets flat ground. The terrain modes of the evaluation scripts put their ground back after the call.
- `skip_terrain_support_field()`, `TERRAIN_FAST_BUILD` (item 9).
- `reset_forward` (item 11): `skip_forward=not reset_forward` in `_reset_idx()` and `_fallen_starts()`.
- Rewards on terrain: `base_height`, `foot_clearance`, `rec_height` and the upright check measure from the ground (item 7).

**go2_train.py**
- `--terrain` sets `env_cfg["terrain"] = {"mode": "grid"}`. Change any key with `--set env.terrain.KEY=VALUE`; the config is validated before Genesis starts.
- `--cmd-curriculum` sets `command_cfg["curriculum"]` to `CMD_CURRICULUM`.
- `--init-from MODEL_PT` loads actor and critic weights only (fresh optimizer, iteration 0), plus the checkpoint's curriculum state.
- `TERRAIN_TILT_MARGIN_DEG` (30) raises the tilt termination on terrain, unless `--set env.termination_if_*` is given.
- A `--- terrain and curricula (P3) ---` block is printed at start, under the P2 block.
- Saves `logs/<exp>/terrain.npz`. Every checkpoint carries `infos["curriculum"]`, through a wrapper around `runner.save`.

**go2_metrics.py**
- `--terrain grid`: one bin per (type, level), all with `--grid_cmd` (default 0.5 0 0, facing +x).
  - The tiles are the run's own `terrain.npz`, or the default grid for runs trained on flat ground. `--terrain_set` changes generation keys, e.g. `seed=1` for unseen tiles; `--types` and `--levels` select tiles.
  - The window lasts as long as a perfect tracker needs to walk half a tile (the curriculum's "up").
  - Each env's tilt limit is `--fall_deg` + its tile's tilt.
- `--slope DEG`: the usual command grid on a tilted plane; the fall limit grows by |DEG|.
- Both terrain modes measure heights from the ground (`z`, `z_err`, `swing_h`) and add a **Progress** table:
  - `dist[m]`: distance from the start point;
  - `through[%]`: covered ≥ `--through` (0.75) of the commanded distance without falling.
- Grid outputs: `terrain_*.png`, a success heat map. The `.npz` gets `foot_ground`, `base_height`, `spawn_pos`, `bin_terrain` and `bin_level`.
- The checkpoint's curriculum state is printed, and its command ranges set the `ood` column.
- Fix: on terrain the harness writes the whole `qpos` (spawn pose + noisy joints) in one call, with forward kinematics.
  - Genesis' `set_dofs_position()` after `set_qpos(..., skip_forward=True)` reverts the free base (fact 2 below), so robots used to start at the origin instead of their tile.
  - Without forward kinematics, the first PD substep pulls from stale joint angles (fact 3 below). On flat tiles that made robots fall 4× more often at their first landing than on the plane, from identical states.
- Flat-ground output is unchanged (same CSV, same checksum).

**go2_compare.py**
- Grid CSVs are matched on (terrain, level, command). `--rows "hill 9" "pit 3"` selects tiles; `--rows "hill 9 0.5 0 0"` selects a tile and a command.
- `--preset terrain`: through, dist, lin_err, vx_meas, z_err, swing_h, slip_v, nonfoot, tau_peak, CoT.

**go2_feet_plot.py**
- On terrain recordings, clearance is measured from the ground under each foot.
- `--tile TYPE LEVEL` picks a tile bin of a `--terrain grid` recording.

**go2_export.py**
- With the command curriculum, `command_ranges` in `deploy_cfg.json` are the ranges the curriculum had reached in that checkpoint. The policy never trained beyond them, and `go2_fsm.py` scales the sticks to `command_ranges`.
- The configured ranges stay in `command_ranges_configured`.
- Terrain runs get a `training_terrain` record: types, levels, mean level reached per type, share at the top level. It is only a record; nothing reads it on the robot.

**go2_teleop.py**
- `--terrain`: the run's grid, or the default grid for a flat-trained run. `--tile TYPE LEVEL` gives the start tile; the default is a hill at the middle level.
- J / K change the level, U the type; each respawns the robot at the tile centre.
- `--slope DEG`.
- For a flat-trained run, the tilt termination is raised by the ground's steepest tilt.
- With a command-curriculum run, the keys stop at the reached ranges; `--configured_ranges` goes up to the configured maximum.

**go2_fsm_sim.py**
- Flat ground by default, also with `--dr` and for runs trained on terrain (the robot side has no map).
- `--slope DEG`: the robot lies aligned with the slope, facing +x. The base height is logged above the ground and the tilt from the ground normal, so the checks keep their meaning. The log meta records `slope_deg`.

Unchanged: `go2_fsm.py`, `go2_deploy_sdk2.py`, `go2_inputs.py`, `go2_commands.py`, `go2_symmetry.py`, `go2_analyze.py`, `go2_sim_suite.py`, `go2_eval.py`, `go2_smooth_ppo.py`, `go2_kinematics.py`, `go2_log.py`, `go2_mujoco_headless.py`, `go2_probe.py`, `test_go2_deploy.py`.

### Genesis 1.4.2 facts behind the P3.1 code

1. **Heightfield layout.** `height_field[i, j]` sits at x = x0 + i·hs, y = y0 + j·hs. Each cell is split along its (i+1, j)–(i, j+1) diagonal (`utils/terrain.py`, collider narrow phase). The lookup in `go2_terrain.py` uses the same triangles: it differs from `TerrainEntity.get_terrain_height` by 0.002 mm at most over 500 points.
2. **`set_dofs_position()` after `set_qpos(..., skip_forward=True)` reverts the free base** to its previous pose. Write the whole `qpos` (base + joints) in one call instead.
3. **`set_qpos(..., skip_forward=True)` leaves `get_dofs_position()` and every link pose at their old values** until the next physics step; only the velocities read 0. Measured after 10 random steps: the calf still reads −1.957 rad after a reset to −1.5. With `skip_forward=False` all reads are the new state.
4. **`SupportField.activate()` precomputes a table that terrains never use** (item 9 above). The 1.4.2 source hash is `eb0321f73f3db143`; 1.4.3 skips terrains itself.
5. **`gs.morphs.Plane(normal=...)`** gives an infinite tilted plane through the origin. It is used for `--slope`.
6. **The terrain geom has its own torsional and rolling friction** (0.005 and 1e-4; the URDF plane has 0 and 0). The feet's values (0.02, 0.01) dominate, so the contact matches the plane: same robot, same states, flat tile vs plane, zero then random actions for 3 s: pitch within 0.002°, foot forces within 0.04 N.
7. **Quaternions:** `transform_quat_by_quat(v, u)` returns R_u·R_v (u applied after v). `quat_to_xyz(q, rpy=True)` gives roll and pitch independent of the heading, so the tilt limits work at any yaw.

### Fixes to earlier phases, found while testing P3.1

1. **A torque kick after every reset (P1.2 on).** The explicit PD reads `get_dofs_position()` at each substep. After the reset's `set_qpos(skip_forward=True)` that read is the pre-reset joint state (fact 3), e.g. a fallen robot's folded calves. So for 5 ms the PD pulls the freshly reset joints toward their targets from the old angles, up to the torque limit.
   - The robot spawns 10 cm in the air, so it lands normally. But the first 50–100 ms of each episode are not the clean start they look like.
   - Kept on flat ground (P2 bit-identity, and every P1/P2 run trained with it); fixed on terrain (`reset_forward`, item 11); fixed in the evaluation harness for terrain (the flat harness never had it: its `set_dofs_position()` runs forward kinematics).
   - Suggestion for the v1.0 freeze (P3.4): train the baseline with `--set env.reset_forward=True`.

### Checked here

- Default flags, against the P2 code:
  - P3.1 trains bit-identical networks and optimizer states (68 tensors; 3 iterations, CPU, 32 envs, seed 1);
  - the env gives identical observations, rewards and resets over 60 steps with pushes;
  - the FSM bench gives identical logs (39 arrays; only wall-clock timings differ).
- `go2_terrain.py --check --genesis`:
  - lookup = vertex heights; torch = numpy; gradient = finite differences;
  - hill/pit faces measure their nominal slope; every tile edge is 0; cache and save/load round trip;
  - heights match Genesis' within 0.002 mm; spheres dropped on the terrain rest 0.1 mm into it.
- The support-field fix: same trajectories with and without it (4 envs, 40 steps, rough + pit grid).
- Terrain curriculum, on a scripted set of episode ends: levels moved as specified (up, down, fell, short, from the top to random).
- Command curriculum: the ranges grew to ±0.75 / ±0.3 with a linear score of 0.9 while wz stayed (yaw score 0.5), and stopped at the configured maximum.
- Default grid env (32 envs, CPU):
  - spawn base 0.42–0.46 m above the ground; standing foot gap −1.0 to +0.6 mm; `base_height` 0.24–0.29 m;
  - 150 steps in 11.8 s.
- Slopes ±15°: spawns lie parallel to the plane (pitch −15.00° facing uphill, +15.00° facing downhill), all four feet within 7 mm of the same height above it; after 1 s of zero actions the feet rest within 1 mm of the plane.
- Training smoke runs (CPU, 32 envs): `--terrain --cmd-curriculum`, `--init-from` (weights and curriculum state restored).
- `go2_metrics.py`:
  - flat tile vs plane, same policy and noise: before the reset fix 3/20 vs 12/20 falls at the first landing; after it, identical pitch for the 6 steps in the air and within 0.03° after landing (3/20 vs 5/20 falls, a policy at the edge of the 10° limit);
  - `--terrain grid` (auto window) and `--slope ±15`;
  - a standing policy reads through 0 % (it used to read 100 %, before the reset fix above);
  - grid subsets of the default grid for a flat-trained run.
- `go2_compare.py` (grid rows, `--preset terrain`, flat output unchanged) and `go2_feet_plot.py --tile`.
- `go2_export.py`: reached ranges and terrain record in `deploy_cfg.json`, the JIT check (difference 0), `test_go2_deploy.py --policy` on that export.
- `go2_teleop.py` headless: `--slope 15`, `--terrain --tile pit 2`, and the J / K / U path with simulated key presses (respawn at each tile centre, level clamped at 0).
- `go2_fsm_sim.py --slope ±15`.
- `go2_symmetry.py`: same mirror errors as P2 (the mirror does not depend on the ground).

**Not checked:**
- Training to convergence on terrain (your GPU). How far the curriculum climbs, and how long it takes.
- The build time with 4096 envs on your laptop. The terrain is one shared entity, so it should not depend on the number of envs, but GPU memory was not measured.
- The J / K / U keys in a real viewer (headless only).
- The robot.

## P3.2: foot slip from ground truth, and the foot-slip penalty decision

P3.2 builds on P3.1. Everything is opt-in. Without `--foot-slip`, a run trains exactly as in P3.1 (and as in P2 on flat ground), bit for bit. On flat ground, the harness's P1/P2 columns keep their exact values; P3.2 only adds columns. Every change is tagged `[P3.2]`. New file: `go2_decide.py` (shared with P3.3). The companion guide is section 5 of **`TEST_AND_TINKER_P3.md`**. The checks of P3.2 and P3.3 are listed together at the end of the P3.3 section.

In the bundle, `p3_3/` is the complete, runnable snapshot (P3.1 + P3.2 + P3.3). `patches/p3_2_3_slip_privileged.patch` holds P3.2 and P3.3 together against the P3.1 files (they touch the same functions): apply `p3_1_curriculum.patch` first, then this one.

```
python go2_terrain.py --check --genesis                       # + the contact normal check (P3.2)
python go2_train.py -e go2-p32-slip --terrain --cmd-curriculum --foot-slip    # + Spot foot_slip_penalty, W = 0.1
python go2_metrics.py -e go2-p32-slip --ckpt 1999 --slope 20  # "Foot slip" + "Foot slip per foot" tables
python go2_compare.py logs/go2-p32/metrics_ckpt1999_seed0_slope+20.csv logs/go2-p32-slip/metrics_ckpt1999_seed0_slope+20.csv --preset slip
python go2_decide.py slip --off go2-p32 go2-p32-s2 --on go2-p32-slip go2-p32-slip-s2 --ckpt 1999   # -> decisions/slip.md
python go2_feet_plot.py logs/go2-p32-slip/metrics_ckpt1999_seed0_slope+20.npz --cmd 0.5 0 0
```

### What P3.2 does, and why

1. **One definition of slip, from the simulator's state.** `go2_terrain.contact_slip_speed(v, ω, n, R)` gives the speed of the foot sphere's contact point along the ground's tangent plane:
   - contact point = centre − R·n, so its velocity is v_cp = v + ω × (−R·n);
   - slip = |v_cp − (v_cp·n)·n|: the normal part is removed;
   - a foot rolling without slipping gives 0, and so does a foot sinking into the soft contact.

   n is the normal of the heightfield triangle under the foot (`normal_at()` / `normal_np()` on every ground), i.e. the face Genesis collides with. The env (penalty, privileged observation), the harness and the foot plot all call this one function.
2. **Why the P1 metric had to change on terrain.** `m_slip()` took the horizontal part of v + ω × (0, 0, −R), averaged over the four feet. That is exact on flat ground, but on a slope:
   - the horizontal speed of a foot sliding along the face is the slip × cos θ (−6 % at 20°, −9 % at 25°);
   - the contact point is along −n, not straight under the centre;
   - a foot sinking into the contact moves along n, and part of that reads as horizontal speed;
   - the mean over feet hides which leg slips.

   On flat ground the old formula is kept character for character, so flat CSVs keep their values exactly.
3. **New harness outputs** (`go2_metrics.py`):
   - "Foot slip" table: `slip_share[%]` (share of stance samples sliding faster than 2 cm/s), `pen_xy` and `pen_t` [cm/s]: what the penalty sums in each mode (below), per policy step, mean over the window. The penalty costs W × pen per second of reward.
   - "Foot slip per foot" table: `slip_v / slip_d / slip_share` of FL, FR, RL, RR (one column per foot; separate `slip_v_FL[cm/s]` ... columns in the CSV).
   - Stance is still |F| > 3 N with the first sample of each stance dropped. The penalty's contact is the reward's 1 N.
4. **The penalty: Isaac Lab Spot `foot_slip_penalty`.** Σ over feet with |F| > 1 N of the foot body's planar world speed [m/s]. Isaac Lab's generic `feet_slide` computes the same.
   - Weight: Spot uses −0.5 next to tracking 5 + 5. The same share of our 1.5 + 0.75 is 0.11, so `FOOT_SLIP_WEIGHT = 0.1`.
   - It is a gentle term: two stance feet at 5 cm/s cost 0.01 per second, against 2.25 for perfect tracking (about Spot's ratio).
   - Contact: unfiltered, this step. The filtered contact of `_update_feet_contacts()` also counts the first step of each swing, the fastest one.
   - Negative, so the P1 penalty curriculum ramps it (×0.3 → 1 over 400 iterations).
5. **Mode `xy` (default) vs `tangent`.** The Go2's foot body is the sphere *centre*. In stance it moves at R·ω of the calf even when the foot does not slip (R = 2.3 cm: 4.6 cm/s at 2 rad/s). So mode `xy` also prices the leg's rotation over a planted foot; on a slope it sees the slip × cos θ. `--set reward.foot_slip_mode=tangent` sums `foot_slip` instead: slip only, the harness quantity. `pen_xy − pen_t` in the harness shows how much of `xy` is rolling. The plan asked for the generic (Spot) term, so `xy` is the default; `tangent` is the variant to try if `xy` changes the gait.
6. **The decision is a rule, fixed before the numbers.** `go2_decide.py slip` reads the harness CSVs of the A runs (no penalty) and B runs (penalty), on flat ground, the grid and every slope, and applies the rule of TEST_AND_TINKER_P3.md 5.4.2:
   - keep the penalty if slip_v drops by ≥ 30 % on every terrain world, with every B seed below every A seed, and no foot rises by > 10 %;
   - and if the costs stay small: lin_err ≤ +5 %, through not lower by > 5 points, CoT ≤ +5 %.

   It writes `decisions/slip.md` (runs, rule, numbers, decision, the flags it implies). The P3.4 config card includes it. Commit it: `logs/` is git-ignored, `decisions/` is not.

### What changed, per file

**go2_terrain.py**
- `normal_from_gradient()`, `contact_slip_speed()` (numpy or torch, component by component).
- `normal_at()` (torch) and `normal_np()` (numpy) on every ground.
- `--check`: slip on a 20° plane is 0 rolling or sinking, the sliding speed when sliding; it equals the P1 formula on flat ground; torch = numpy.
- `--check --genesis`: spheres dropped on triangle centroids of sloped faces; the solver's contact normal at touchdown is compared with the lookup's normal. The rest gap is now measured on the faces below 5° only (on steeper ones the spheres roll away).

**go2_env.py**
- New buffers, updated in `step()`: `feet_ang` (foot angular velocity), `feet_normal` (ground normal under each foot; constant (0, 0, 1) on flat ground), `feet_force` (contact force vectors, world frame) and `foot_slip` [N, 4] (contact-point slip in contact, 0 in the air).
- `feet_force` and `foot_slip` are zeroed at a reset (the robot respawns in the air).
- `_reward_foot_slip()`, `reward_cfg["foot_slip_mode"]` (`"xy"` default, `"tangent"`); an unknown mode is an error.
- The `get_links_net_contact_force()` read is shared: norms as before, plus the foot vectors.

**go2_train.py**
- `--foot-slip [W]` (alone: `FOOT_SLIP_WEIGHT` = 0.1); `reward.foot_slip_mode` checked before Genesis starts.
- A `P3.2` line in the start-up block, now titled `--- terrain, curricula, foot slip, privileged observations (P3) ---`.

**go2_metrics.py**
- `contact_point_speed()`: `contact_slip_speed()` on the recorded `foot_normal` (terrain modes), the P1 formula on flat recordings.
- `m_slip()`: + `slip_share`, `pen_xy`, `pen_t`. `m_slip_feet()`: the per-foot table, printed one column per foot.
- Terrain recordings get `foot_normal` [T, N, 4, 3] (at the logged foot positions); every `.npz` gets `contact_threshold`.

**go2_compare.py**
- `--preset slip`: slip_v (all feet, then each foot), slip_share, slip_d, pen_xy, pen_t, lin_err, through, stride, swing_h, CoT.
- The default, gait and terrain presets select `slip_v[` (all feet), so their output is as before.

**go2_feet_plot.py**
- The speed curve is the contact point's speed along the ground on terrain recordings (`foot_normal`); flat recordings as before. The title gives each foot's mean stance slip.

**go2_decide.py** (new; no Genesis)
- Rules `slip` (P3.2) and `critic` (P3.3), thresholds as constants in its header, `decisions/<rule>.md`.

## P3.3: privileged observations for the critic and a teacher

P3.3 builds on P3.2. Opt-in: without `--privileged`, the env returns the `"policy"` group only and a run trains exactly as before, bit for bit. Every change is tagged `[P3.3]`. The deployed actor stays blind: a privileged *critic* changes nothing on the robot side. The companion guide is section 6 of **`TEST_AND_TINKER_P3.md`**.

```
python go2_train.py -e go2-p33 --terrain --cmd-curriculum --privileged critic     # asymmetric actor-critic
python go2_train.py -e go2-p33t --terrain --cmd-curriculum --privileged teacher \
    --privileged-entries base_lin_vel friction added_mass com_shift motor_strength foot_forces foot_slip height_scan
python go2_symmetry.py --privileged --slope 10                 # mirror rules of every entry, randomized values, slope
python go2_decide.py critic --off go2-p32 go2-p32-s2 --on go2-p33 go2-p33-s2 --ckpt 1999   # -> decisions/critic.md
python go2_export.py -e go2-p33 --ckpt 1999                     # unchanged; refuses a teacher run
```

### What P3.3 does, and why

1. **A second observation group.** When `obs_cfg["privileged"]` lists entries, `Go2Env.get_observations()` returns `TensorDict({"policy": ..., "privileged": ...})`.
   - rsl_rl 5.5.1 chooses the groups of each network with `obs_groups`, and `MLPModel` concatenates them in that order.
   - `--privileged critic`: `{"actor": ["policy"], "critic": ["policy", "privileged"]}`, the asymmetric actor-critic. PPO's advantages come from the critic. A critic that sees the true velocity, the randomized parameters and the contacts can predict returns better under randomization and terrain. The actor, the export and the robot side are unchanged.
   - `--privileged teacher`: the actor reads both groups too. It cannot be deployed (`go2_export.py` refuses it). It is the first half of teacher–student training (Lee et al. 2020): a student that reads `"policy"` only is distilled from it later (rsl_rl 5.5.1's `Distillation` takes `obs_groups` `"student"` / `"teacher"`; not part of P3.3).
   - legged_gym has the same hook (`num_privileged_obs`, `privileged_obs_buf`).
2. **The entries** (`go2_env.PRIVILEGED_SIZES`; `--privileged-entries` picks and orders them):

   | entry | size | scale | default |
   |---|---|---|---|
   | `base_lin_vel` | 3 | body frame × 2.0 (`obs_scales.lin_vel`) | yes |
   | `friction` | 1 | (μ − 0.8) / 0.4 | yes |
   | `added_mass` | 1 | (m − 0.5 kg) / 1.5 | yes |
   | `com_shift` | 3 | body frame / 0.03 m | yes |
   | `motor_strength` | 12 | (k − 1) / 0.1, policy joint order | yes |
   | `foot_forces` | 12 | body frame / 100 N, clipped ±3 | yes |
   | `foot_slip` | 4 | P3.2 slip / 0.2 m/s, clipped at 5 | yes |
   | `gains` | 2 | (Kp, Kd factor − 1) / 0.15 | |
   | `latency` | 1 | latency / policy step (0–1) | |
   | `foot_clearance` | 4 | sphere bottom above the ground / 0.1 m, clipped −1…3 | |
   | `height_scan` | 187 | legged_gym's 17 × 11 points (x −0.8…0.8, y −0.5…0.5 m, heading frame): clip(base z − 0.3 − h, −1, 1) × 5 | |

   The default (`PRIVILEGED_DEFAULT`, 36 values) is the P3.3 list: friction, added mass, CoM, motor strength, base velocity, contact forces, foot slip.
3. **Design choices.**
   - Noise-free, and scaled by fixed constants (not by the run's randomization ranges), so a value means the same in every run.
   - Vectors are in the body frame (velocity, CoM shift, foot forces), so the left–right mirror is a fixed permutation and sign. A world-frame vector would need the heading to mirror.
   - At a reset the robot respawns in the air at rest: velocities, foot forces and slip read 0; the randomized parameters are the new episode's.
   - The height scan uses legged_gym's grid and formula, with the Go2's standing height 0.3 m where legged_gym subtracts ANYmal's 0.5 m. On flat ground every point reads (base z − 0.3) × 5.
   - `foot_clearance` reads the foot positions again: with `reset_forward` (terrain) they are the respawned feet.
4. **Symmetry** (`go2_symmetry.py`). `--symmetry` augments every group, critic inputs included. Each entry has its mirror:
   - y flips for the body-frame vectors;
   - legs swap for per-joint and per-foot entries (foot forces also flip y);
   - scalars are unchanged;
   - the height scan flips its y index.

   An entry without a rule raises an error. `python go2_symmetry.py --privileged [--slope 10]` checks them all, in a mirrored rollout where env 0 gets a random CoM shift and random motor strengths and env 1 their mirror.
5. **CAPS / LCP with a teacher.** The CAPS spatial noise of the privileged part is 0 (`caps_sigma` padded with zeros); the LCP gradient penalty covers both groups.

### What changed, per file

**go2_env.py**
- `PRIVILEGED_SIZES`, `HEIGHT_SCAN_X` / `_Y` / `_OFFSET`, `privileged_layout(obs_cfg)` (validates names and repeats), `height_scan_points()`.
- `privileged_layout`, `num_privileged_obs`, `privileged_buf`, `height_scan_shape`; `_privileged_obs()`, `_height_scan()`.
- `_update_observation()` builds `privileged_buf` after the noise draw, so the random stream of `"policy"` is unchanged.
- `get_observations()` adds `"privileged"` only when entries are listed.

**go2_train.py**
- `--privileged critic|teacher`, `--privileged-entries ENTRY ...`, `PRIVILEGED_DEFAULT`.
- Checks before Genesis starts: unknown or repeated entries; an `obs_groups` set that reads `"privileged"` with no entries.
- A `P3.3` line in the start-up block (entries, sizes, which networks read them; "TEACHER" for a teacher).
- Teacher + CAPS: `caps_sigma` padded with zeros.

**go2_symmetry.py**
- `_privileged_mirror()`, `mirror_obs(env, obs, group)`, `mirror_observations(env, obs)`; `compute_symmetric_states()` mirrors every group.
- `--privileged` and `--slope DEG` for the check; it prints the size of each block's signal next to its error.

**go2_export.py**
- Refuses a run whose actor reads more than `"policy"`. Records the critic's entries in `deploy_cfg["source"]["critic_privileged"]`.

**go2_metrics.py**
- `sym_err` mirrors every group the actor reads (`mirror_observations`), so it works for teachers too.

**go2_decide.py**
- Rule `critic`: learning speed and final values from the TensorBoard files (`Train/mean_reward`, `Curriculum/terrain_level`; `Loss/value` shown), plus the harness costs of the `slip` rule.

Unchanged by P3.2 and P3.3: `go2_teleop.py`, `go2_fsm_sim.py`, `go2_eval.py`, `go2_smooth_ppo.py` and the deploy side (`go2_fsm.py`, `go2_deploy_sdk2.py`, ...). They pass the env's full observation to the policy, which picks its own groups.

### rsl_rl 5.5.1 and Genesis 1.4.2 facts behind the P3.2 and P3.3 code

1. `resolve_obs_groups()` checks that every group of `obs_groups` exists in the env's TensorDict. A missing default set falls back to a group of the same name, then to `"policy"`.
2. `MLPModel.get_latent()` is `torch.cat([obs[g] for g in obs_groups])`. `obs_normalization` defaults to False, hence the fixed scales above.
3. `RolloutStorage` stores every group of the TensorDict, read or not. Memory: 4096 envs × 24 steps × 36 floats = 14 MB for the default entries; with the height scan (+187), 88 MB.
4. The symmetry extension passes the whole TensorDict to `compute_symmetric_states()`; augmented samples feed the value loss too.
5. `export_policy_to_jit()` scripts the actor on its concatenated input, so a teacher's export would take the privileged inputs.
6. Genesis `RigidEntity.get_contacts()` returns the solver's contact normals (`"normal"`), used by the P3.2 normal check; `get_links_ang()` gives the link angular velocities (world frame) for the contact-point velocity.

### Checked here (P3.2 and P3.3)

- Default flags (no `--foot-slip`, no `--privileged`), against the P2 and P3.1 code:
  - bit-identical networks and optimizer states (68 tensors; 3 iterations, CPU, 32 envs, seed 1);
  - identical observations, rewards and resets over 60 steps with pushes (flat env);
  - the FSM bench gives identical logs (35 arrays; only the wall-clock timings differ);
  - the flat harness: every P2 column identical, same checksum; 30 columns added.
- The harness on terrain, P3.1 vs P3.2 code, same checkpoint:
  - identical recordings and checksums on a 15° slope and on grid tiles (hill, pit, rough at levels 0 and 9);
  - every column identical except `slip_v` (by design). On the slope it is 3.6 % higher, the horizontal/tangential
    ratio of the recorded stance samples is 0.967 (cos 15° = 0.966); on the hill and pit platforms it is unchanged.
- `go2_terrain.py --check --genesis`:
  - slip on a 20° face is 0 rolling or sinking, the sliding speed when sliding; equal to the P1 formula on flat ground;
    torch = numpy;
  - the solver's contact normal at touchdown is within 0.03° of the lookup's (12 spheres, faces 0 to 25°);
  - heights within 0.002 mm of Genesis'; spheres at rest 0.1 mm into the surface.
- Physics of the slip: a robot sliding down a 20° plane at friction 0.1, zero actions: foot slip 2.62 m/s for a base
  sliding at 2.60 m/s along the slope; the horizontal formula reads 0.939 of it (cos 20° = 0.940). `foot_slip` is 0 in
  the air, and the privileged entry clips at 5 (1 m/s).
- Training smoke runs (CPU, 32 envs, 3 iterations):
  - `--terrain --cmd-curriculum --foot-slip --privileged critic --symmetry`: actor 45 inputs, critic 81; `Episode/rew_foot_slip`
    and `Loss/symmetry` logged;
  - `--privileged teacher` with 8 entries (height scan included) `--caps-s 0.5 --lcp 0.002 --foot-slip 0.2
    --set reward.foot_slip_mode=tangent`: actor and critic 259 inputs; CAPS and LCP losses logged;
  - `--terrain --privileged teacher` (velocity, height scan, foot clearance) `--symmetry both`: 239 inputs;
  - `--rnn lstm --privileged critic`: actor LSTM(45, 256), critic LSTM(81, 256).
- Errors before Genesis starts: unknown or repeated entries, `--privileged-entries` without `--privileged`, an
  `obs_groups` set reading `"privileged"` with no entries, an unknown `foot_slip_mode`.
- `go2_symmetry.py --privileged`, every entry, with a random CoM shift and random motor strengths in env 0 and their
  mirror in env 1:
  - flat ground: largest block error 8.3e-4 (foot slip); the policy blocks as in P2 (≤ 4.6e-4);
  - 10° slope: largest 5.7e-3 (foot forces);
  - breaking a rule on purpose (foot-force y sign, feet not swapped, CoM y sign, height scan not flipped): 0.12 to 1.6.
- `go2_export.py`: a privileged-critic run exports (JIT difference 0, `critic_privileged` recorded) and passes
  `test_go2_deploy.py --policy` (45 observations); a teacher run is refused with its message.
- `go2_metrics.py` on a teacher run (`sym_err` over both groups); `go2_teleop.py --headless` with a teacher.
- `go2_compare.py --preset slip`; `go2_feet_plot.py` on flat and slope recordings.
- `go2_decide.py slip` and `critic` on the smoke runs: records written. Missing data fail a criterion; a seed that never
  reaches the reward target counts as slow. (The decisions themselves need trained runs.)

**Not checked:**
- Training to convergence (your GPU): the A/B runs of the slip decision, the privileged critic's learning speed, a teacher.
- Whether `xy` changes the gait on a trained policy (`pen_xy − pen_t`, stride, step frequency): a smoke run's policy does not walk.
- Distillation of a student from a teacher (not implemented).
- The robot (nothing changes there: the deployed actor reads the same 45 inputs).

## P3.4: Baseline v1.0 (final run, config card, checked export, harness report, git tag)

P3.4 builds on P3.3 and adds no training feature: a run with the same flags trains as before, bit for bit. Two behaviours change on purpose: `--set` given several times now adds up (until P3.3 the last `--set` silently replaced the others), and the run's manifest is written even without git. New files: `go2_provenance.py`, `go2_report.py`, `go2_card.py`. Every change is tagged `[P3.4]`. The companion guide is section 7 of **`TEST_AND_TINKER_P3.md`**.

In the bundle, `p3_4/` is the complete, runnable snapshot (P3.1 to P3.4). `patches/p3_4_baseline.patch` holds P3.4 against the P3.3 files: apply `p3_1_curriculum.patch`, then `p3_2_3_slip_privileged.patch`, then this one.

```
python go2_decide.py flags                                     # the go2_train.py flags of the KEEP decisions
V1FLAGS="--terrain --cmd-curriculum --set env.reset_forward=True $(python go2_decide.py flags)"
python go2_train.py -e go2-v1.0 $BASE $V1FLAGS                # $BASE: your P2 options; from a clean, committed tree
python go2_export.py -e go2-v1.0 --ckpt 1999 --onnx            # policy.pt + policy.onnx, both checked
python go2_report.py -e go2-v1.0 --ckpt 1999 --gate ~/unitree_mujoco   # the harness report (+ the gate suite)
python go2_card.py -e go2-v1.0 --ckpt 1999 --name "Baseline v1.0" --freeze baselines/v1.0 --strict
python go2_report.py -e go2-v2 --ckpt 1999 --ref go2-v1.0      # later: a candidate against v1.0
```

### What P3.4 does, and why

1. **Provenance: what trained the policy.** `go2_train.py` writes `logs/<exp>/manifest.json` when a run starts. P3.4 completes it:
   - git commit, dirty flag (tracked files), untracked `.py` files;
   - the command line (`argv`) and every argument, host, GPU, package versions, start time;
   - a **code fingerprint**: the sha256 of every `.py` file of the code folder.

   Until P3.3, a failing `git` call skipped the whole manifest; now its git fields are `null`. A dirty tree also leaves `git_diff.patch` and a warning at the start; untracked `.py` files get a warning too. Git alone says which commit a run started from, not whether a file was edited afterwards. The fingerprint does: `python go2_provenance.py logs/<exp>` and the card compare it with the files as they are now.
2. **`--set` adds up.** It is now `nargs="+", action="extend"`. Before, argparse kept only the last `--set` list: `--set env.reset_forward=True --set reward.foot_slip_mode=tangent` silently dropped the first. The v1.0 command line combines its own `--set` with the flags of the decisions, so this had to be fixed first.
3. **The checked export** (`go2_export.py --onnx`).
   - `policy.onnx`: rsl_rl 5.5.1 `export_policy_to_onnx()`, opset 18. MLP: input `obs` [1, D] → `actions` [1, 12]. Recurrent: + `h_in` (+ `c_in` for an LSTM) → `h_out` (`c_out`), which the caller carries from call to call and zeroes at a reset.
   - One file: torch 2.15 writes the weights to a separate `policy.onnx.data` by default; `go2_export.py` folds them back into `policy.onnx`, so the file is self-contained and its sha256 covers the weights.
   - Both exports are checked against the training policy (deterministic mean), one observation at a time as on the robot, on two input sets: 100 random observations (the P1.6 check), and the observations of a rollout of the training policy in the nominal world (8 envs × 200 steps, commands resampled as in training), each env's sequence replayed with its resets.
   - Any difference above `EXPORT_TOL` = 1e-4 stops the script. Float32 round-off is about 1e-7.
   - `exported/export_check.json` records every error, the sha256 of the files and the ONNX inputs and outputs. `deploy_cfg.json` → `source` gets the hashes, the errors, the git commit and the export time.
   - An export without `--onnx` deletes a stale `policy.onnx` (and `policy.onnx.data`), so the folder never holds two different networks.
   - Why the rollout check: random observations miss the states the policy really meets, and a recurrent export's state handling (carry, reset) only shows on sequences.
   - Why ONNX: runtimes without PyTorch (onnxruntime in C++, TensorRT); and torch 2.15 warns that `torch.jit.load` is deprecated. `policy.pt` still loads, and `go2_fsm.py` still runs it.
4. **The harness report** (`go2_report.py`). One command runs the standard set as `go2_metrics.py` subprocesses, and optionally the unitree_mujoco gate suite:
   - worlds: `flat`, `flat_dr`, `grid` (the run's tiles), `grid_unseen` (new random tiles), slopes ±10°, ±20°, ±25°;
   - outputs in `logs/<exp>/report/`: `REPORT.md`, `summary.json` (strict JSON: n/a is `null`), `plots/`, one log per world;
   - headline numbers: per key flat command (falls, lin_err, yaw_err, vx_meas, CoT, slip_v); per tile type the **frontier** (the last level of the run of levels, from the lowest, with through ≥ 80 %); on each slope vx_meas, through and falls at 0.5 m/s;
   - `--ref EXP` adds every headline number of both runs, the change and which way is better: the same command measures later candidates against v1.0. It also takes a frozen baseline folder (`--ref baselines/v1.0`), and it is read before the evaluation starts;
   - `--collect` rebuilds the report from existing CSVs, `--skip-existing` runs only the missing worlds. The envs per bin are read from the CSVs, so a collected report states what was measured.
5. **The config card** (`go2_card.py` → `logs/<exp>/CONFIG_CARD.md`, about 60 lines). It reads the run's own files only.
   - A freeze checklist opens it, one line per condition, `ok` or `NO` with the reason: code committed, code unchanged, export checked, report complete, decision slip, decision critic, reset_forward.
   - "code committed" also fails on untracked `.py` files (a checkout of the commit would lack them). "report complete" also asks for every tile type and level the run trained on, and 20 envs per bin (`REPORT_MIN_REPS`): a quick report cannot pass for the baseline's.
   - Then: identity and training (commit, versions, exact command, PPO, networks), world, interface and rewards, results, export, decisions, how to reproduce.
   - `--freeze DIR` copies the card and the small records into a git-tracked folder and prints the git and `gh` commands; `--strict` exits 1 unless every line is `ok`, and then freezes nothing.
6. **The decisions feed the command line.** `python go2_decide.py flags` prints the flags of the KEEP records in `decisions/`, and says on stderr what each record decided (or that it is missing). The card checks that the run follows the records.
7. **Tag and release.** `logs/` and `*.pt` are git-ignored. So the tag carries the records (`baselines/v1.0/`: card, deploy config, export check, report, decisions, manifest, configs), and a GitHub release on the tag carries the binaries (policy files, checkpoint, `terrain.npz`) with the card as notes. `gh release create --verify-tag` attaches the release to the pushed tag; without it, gh would create a missing tag from the default branch.

### What changed, per file

**go2_provenance.py** (new; standard library only)
- `sha256_file()`, `code_fingerprint()`, `git_info()`, `environment()`, `now()`, `compare_code()`.
- `python go2_provenance.py` prints the git state, the package versions and the code fingerprint; `python go2_provenance.py logs/<exp>` compares the run's fingerprint with the files now.

**go2_train.py**
- `run_manifest(log_dir, cfgs, args)`: the fields above, `git_diff.patch`, the two warnings.
- `--set` repeatable (`action="extend"`).
- The `subprocess` import is gone (the git calls moved to `go2_provenance.py`).

**go2_export.py**
- `--onnx`, `--check-steps` (200), `--check-envs` (8); `EXPORT_TOL` = 1e-4.
- `OnnxPolicy` (an onnxruntime session that carries the recurrent state), `single_file_onnx()`, `rollout_observations()`, `sequence_error()`.
- `export_check.json`; `deploy_cfg["source"]`: `git_commit`, `git_dirty`, `exported`, `sha256`, `max_abs_error_vs_training_policy`.
- The env is built with `--check-envs` envs (the deploy config does not depend on their number). The teacher refusal of P3.3 is unchanged.

**go2_report.py** (new; Genesis runs only in its `go2_metrics.py` subprocesses)
- `WORLDS`, `SLOPES`, `FLAT_BINS`, `FLAT_METRICS`, `SLOPE_METRICS`, `FRONTIER_THROUGH`, `HEADLINE_BETTER`.
- Options: `--reps`, `--seed`, `--worlds`, `--grid-types`, `--grid-levels`, `--cpu`, `--collect`, `--skip-existing`, `--gate`, `--gate-summary`, `--ref` (a run name, a run folder or a frozen `baselines/` folder).

**go2_card.py** (new; no Genesis)
- `STANDARD_WORLDS`, `RELEASE_FILES`, `REPORT_MIN_REPS`; `read_decision()`, `decision_matches()`, `switches()` (the flags a run's configs imply, for manifests from before P3.4).
- Options: `--name`, `--decisions`, `--freeze DIR`, `--tag`, `--strict`, `--log_dir`.

**go2_decide.py**
- Rule `flags`; option `--decisions`; `--off`, `--on` and `--ckpt` are required for `slip` and `critic` only.

Unchanged by P3.4: `go2_env.py`, `go2_metrics.py` and the deploy side. `go2_fsm.py` reads `deploy_cfg.json` as before; the new keys under `source` are records only.

### rsl_rl 5.5.1 and torch 2.15 facts behind the P3.4 code

1. `OnPolicyRunner.export_policy_to_onnx()` runs `torch.onnx.export(..., opset_version=18)` on the model's `as_onnx()` wrapper: `_OnnxMLPModel` (`obs` → `actions`) or `_OnnxRNNModel` (+ `h_in`, `c_in` → `h_out`, `c_out`), batch 1.
2. In torch 2.15, `torch.onnx.export` defaults to `dynamo=True` (since 2.9: the `torch.export`-based exporter, which imports `onnxscript`) and to `external_data=True` (weights in a separate `.data` file). rsl_rl passes neither.
3. torch 2.15 prints a `FutureWarning` on `torch.jit.load`: "`torch.jit.load` is deprecated. Please switch to `torch.export`." The PyTorch 2.8 release notes draft calls TorchScript "nearing end-of-life".
4. rsl_rl's logger writes `logs/<exp>/git/<repo>.diff` only for repositories registered with `runner.add_git_repo_to_log()`. By default that is rsl_rl itself, which pip installs outside any git repository ("Could not find git repository ... Skipping."), so the folder stays empty. The manifest and `git_diff.patch` record our code instead.

### Checked here (P3.4)

- Default flags: bit-identical networks and optimizer states with the P2 code (68 tensors; 3 iterations, CPU, 32 envs, seed 1).
- In a scratch git repository:
  - a clean run's manifest: commit, dirty False, no untracked file, `argv`, every argument, 27 file hashes; two `--set` both applied (start-up lines and configs);
  - `go2_provenance.py logs/<exp>`: "code unchanged since training", then "changed since training: ['go2_metrics.py']" after an edit;
  - a dirty run: the warning, `git_diff.patch` with the edit, `git_dirty` True; an untracked `scratch.py`: the warning, and "code committed" NO on the card;
  - outside git: the manifest with `null` git fields.
- `go2_export.py --onnx` (CPU), max |export − policy| on random observations / on the rollout:
  - MLP with a privileged critic (8 envs × 200 steps): TorchScript 0 / 1.3e-7, ONNX 1.8e-7 / 1.2e-7;
  - LSTM (4 × 100): TorchScript 0 / 6.0e-8, ONNX 1.2e-7 / 1.5e-7, inputs `obs` [1, 45], `h_in` [1, 1, 256], `c_in` [1, 1, 256]; GRU: TorchScript 0 / 6.0e-8, ONNX 1.2e-7 / 1.2e-7, inputs `obs`, `h_in`;
  - each time one self-contained `policy.onnx` (weights folded in, no `.data` file), and the check runs on that file;
  - a tolerance of 1e-12 stops the script with its message, writes `passed: false`, and the card says NO;
  - an export without `--onnx` removes an earlier `policy.onnx` (and a `policy.onnx.data`);
  - `test_go2_deploy.py --policy` passes on the new export.
- `go2_report.py` on a smoke run (CPU, `--reps 2`, tile subsets): 6 worlds in 250 s (32 to 49 s each); `--collect`; `--ref` with a run name, a run folder, a frozen folder and the file itself (falls in percentage points), an unknown `--ref` refused before any evaluation; missing worlds listed; `summary.json` loads as strict JSON.
- `go2_card.py`: runs from before P3.4 ("Command: unknown", the implied switches); a clean git run; KEEP, DROP and missing decision records, matching and not; `--freeze` (copies and printed commands); `--strict` (exit 1, nothing frozen); "report complete" NO for a quick report (2 envs per bin, tile subsets, missing slopes) and ok for a complete one.
- `go2_decide.py flags` with KEEP records, DROP records and none.
- The dry run of `TEST_AND_TINKER_P3.md` 7.2, as written, in a scratch git repository (CPU): training, export (errors ≤ 2.7e-7), deploy tests, the 10-world report in about 7 min, the card (code committed, code unchanged, export checked and reset_forward ok; report complete and the decisions NO, as expected there), `--freeze`, and `--strict` exiting 1.
- `py_compile` and pyflakes on every P3.4 file.

**Not checked:**
- The real v1.0 run, its report and the gate suite (your GPU, unitree_mujoco).
- `git push` and `gh release create` (the card prints them; nothing is pushed from here).
- `policy.onnx` in a C++ runtime or on the robot computer.

## P3 v2: the P2 results applied (2026-11-06)

Your P2 results (`P2_SUMMARY.md`) changed what P3 should assume and how it should judge. P3 v2 applies its "What this means for P3" list. It adds no training feature: default flags still train bit for bit as in P2, and `$BASE` now carries the P2 choices. Every change is tagged `[P3v2]`; the guide's section 2 lists them, and each section of `TEST_AND_TINKER_P3.md` explains its own.

In the bundle, `p3_v2/` is the complete, runnable snapshot. `patches/p3_v2_p2_results.patch` holds P3 v2 against the P3.4 files: apply it after `p3_4_baseline.patch`.

```
BASE="--seed 1 --symmetry --energy 1.5e-3"     # the P2 choices
P2=go2-p25-w1.5e-3; P2S2=go2-p25-s2            # the final P2 policy and its second seed
DEPLOY_PY=$(conda run -n go2deploy which python)
python go2_decide.py slip --off go2-p32 go2-p32-s2 go2-p32-s3 --on go2-p32-slip go2-p32-slip-s2 go2-p32-slip-s3 --ckpt 1999
python go2_report.py -e go2-v1.0 --ckpt 1999 --seeds go2-v1.0-s2 go2-v1.0-s3 --gate ~/unitree_mujoco --deploy-python $DEPLOY_PY
python go2_report.py -e go2-v2 --ckpt 1999 --seeds go2-v2-s2 go2-v2-s3 --ref baselines/v1.0
python go2_train.py -e go2-p31-noE $BASE --energy 0 --terrain --cmd-curriculum   # only if hill / pit levels stall
```

### What changes, and why

| P2 finding | Change |
|---|---|
| Final P2 configuration `--symmetry --energy 1.5e-3`; final policy `go2-p25-w1.5e-3` (seed 2: `go2-p25-s2`); no `go2-p2` run | Guide: `$BASE`, `$P2`, `$P2S2`; P3.1 compared with both P2 seeds |
| Training noise far above evaluation noise: over 3 identical trainings `lin_err` at 0.5 m/s 0.060–0.085, CoT at 1 m/s 0.595–0.706 | `go2_decide.py`: a cost counts only if it is beyond its tolerance **and** every --on seed is worse than every --off seed; 3 seeds per arm |
| One candidate against one baseline run is not a result | `go2_report.py --seeds`: seed band and verdicts; `go2_card.py`: checklist line `seed band` |
| Two conda environments; the first P2 batch failed with `No module named 'unitree_sdk2py'` | `go2_report.py --deploy-python` (default `$DEPLOY_PY`), checked before the evaluation |
| Two MuJoCo batches in parallel make the results invalid | `go2_sim_suite.py`: one battery at a time (a lock) |
| Energy may slow the terrain curriculum (expected, not tested) | Guide: what to watch (4.5, 4.9); `--energy 0` removes the term from `$BASE` for a comparison run |
| Energy and the foot-slip penalty both price foot motion | Both slip arms with `$BASE`; `stride` and `swing_h` on every terrain world of the slip record |
| Slow, lateral and turning commands still undershoot; a privileged critic is one candidate | The critic record shows them (not judged) |
| `--air-time-max` help overstated the uncapped term | Help text corrected |

1. **Decisions against the training noise.** Until P3.4, a cost failed as soon as the --on mean was beyond its tolerance (`lin_err` +5 % and +0.005 m/s, CoT +5 %, through −5 points, a foot's slip +10 %; for the critic: learning 5 % later, final reward −2 %, terrain level −0.2). P2's noise is larger than those tolerances, so with 2 seeds per arm the rule would often DROP a feature for noise. Now a cost also needs **every --on seed worse than every --off seed**. With no real effect that happens by chance 1 / C(n_on + n_off, n_on): 1 in 6 with 2 + 2 seeds, 1 in 20 with 3 + 3. The benefit criterion of the slip rule already asked for the same separation, and is unchanged. The record states the seed counts and that rate; below 3 seeds per arm the script says so. The price: a small real cost inside the seed spread passes. That is what the seeds can resolve, and the record keeps every seed's value.
2. **Seed bands in the report.** `--seeds EXP-s2 EXP-s3` adds the other training seeds of the configuration (their reports first). `REPORT.md` gets every headline number per run, the mean and the range; `summary.json` a `"band"`. The seeds' configurations are compared with this run's (`cfgs.pkl` and the manifest's arguments, except run names and seed): a difference is printed and recorded, and the card then says NO.
3. **Verdicts.** With a band on either side, `--ref` gives each headline number a verdict: better or worse only when the two sides' seed ranges do not overlap, else "within noise". One run on each side gets no verdict, and the report says why. `--ref baselines/v1.0` carries v1.0's band, because the frozen `summary.json` holds it.
4. **The gate in its own environment.** `go2_report.py --gate` starts `go2_sim_suite.py` with `--deploy-python` (default `$DEPLOY_PY`, as in `p2_overnight.sh`), which then runs the whole battery with that python. Before the evaluation starts, the report checks the export, the unitree_mujoco folder, and that this python imports `unitree_sdk2py`.
5. **One MuJoCo battery at a time.** `go2_sim_suite.py` takes an exclusive, non-blocking lock (`go2_sim_suite.lock` in the temp folder) before its first case. A second battery stops at once with a message. The kernel releases the lock when the process ends, also after a crash, so a lock cannot be left behind.
6. **`--energy 0` is off.** `$BASE` holds `--energy 1.5e-3`; a later `--energy 0` now adds no term, so `$BASE --energy 0` trains exactly as without the flag (checked bit for bit). That is the comparison run for the energy-against-terrain question (guide 4.3, run 5).

### What changed, per file

**go2_decide.py**
- `all_worse()`, `chance()`; every cost check of `harness_costs()`, `rule_slip()` (per foot) and `rule_critic()` (learning speed, final reward, terrain level) needs both conditions; failing types and slopes are listed.
- Seeds line in the printout and the record (`- seeds: ...`); a note below 3 seeds per arm.
- Evidence rows, not judged: `stride`, `swing_h` on every terrain world (slip); flat `lin_err` over the slow and lateral bins and `yaw_err` turning in place (critic).

**go2_report.py**
- `--seeds`, `--deploy-python`; `SEED_KEYS`; `find_summary()`, `config_of()`, `run_folder()`, `band_of()`, `side_of()`, `side_text()`, `verdict()`.
- `REPORT.md`: a "Seed band" table; the "Against" table shows both sides (mean [min–max]), the change of the means and the verdict; a note for one run on each side.
- `--gate`: export, folder and `unitree_sdk2py` checked first; the gate's exit code printed.

**go2_card.py**
- Checklist line `seed band` (≥ `BAND_MIN_RUNS` = 3 runs of one configuration); a results line with the band of `BAND_KEYS`.

**go2_sim_suite.py**
- `one_batch_at_a_time()`, called before the first case (not for `--list`).

**go2_train.py**
- `--energy 0` adds no term; help texts of `--energy` and `--air-time-max`.

### Checked here (P3 v2)

- Default flags: bit-identical networks and Adam moments with the P2 code (51 of 51 tensors; 3 iterations, CPU, 32 envs). `--symmetry --energy 1.5e-3 --energy 0` against `--symmetry`: 51 of 51 identical.
- `go2_decide.py slip` on synthetic 3 + 3 seed CSVs: a +17 % `lin_err` mean whose seed ranges overlap passes (the P3.4 rule would have DROPped); a consistent +36 % fails; the seeds line reads "1 in 20" (2 + 2: "1 in 6" and the note). `go2_decide.py critic` on smoke runs: the new detail lines and the evidence rows; `go2_card.py` still reads the new records.
- `go2_report.py`: a 3-run band (one configuration); `--ref` band against band (consistently worse numbers say "worse", equal ones "within noise"); one run each ("–" and the note); a seed with a changed reward weight (`WARNING`, and the card's `seed band` NO); `--seeds` without a report, `--gate` with a missing folder, a python without `unitree_sdk2py` and a python that does not exist: all refused before any evaluation.
- `go2_card.py`: `seed band` ok with 3 runs; NO with 2 runs, with another configuration, without a band.
- `go2_sim_suite.py`: a second holder of the lock is refused with the message while the first holds it; once the first ends, the lock is free again. `--list` unchanged.
- pyflakes on every changed file.

**Not checked:** the gate suite through `--deploy-python` against your go2deploy environment (no unitree_mujoco here); the energy-against-terrain effect (it needs your terrain runs).

## Rewards and terminations per phase

P1.4 to P1.6 keep every P1.3 value; their additions are the last rows.

| term | original | P1.1 | P1.2 | P1.3 | source |
|---|---|---|---|---|---|
| tracking_lin_vel | 1.0 | 1.0 | 1.0 | 1.0 | |
| tracking_ang_vel | 0.2 | 0.5 | 0.5 | 0.5 | legged_gym |
| lin_vel_z | −1.0 | −1.0 | −1.0 | −2.0 | legged_gym |
| base_height | −50 | −50 | −50 | −50 | |
| action_rate | −0.005 | −0.005 | −0.005 | −0.01 | legged_gym |
| similar_to_default | −0.1 | −0.1, ×5 standing | same | same | Isaac Lab Spot |
| feet_air_time | – | 1.0, cmd ≠ 0 | 1.0 | 1.0 | legged_gym |
| orientation | – | – | −2.5 | −2.5 | Isaac Lab Go2 flat |
| ang_vel_xy | – | – | – | −0.05 | legged_gym |
| torques | – | – | – | −1e-5 | legged_gym |
| dof_acc | – | – | – | −2.5e-7 | legged_gym |
| collision | – | – | – | −1.0 | legged_gym |
| dof_pos_limits | – | – | – | −10, soft 0.9 | unitree_rl_gym Go2 |
| dof_vel_limits | – | – | – | −1.0, soft 0.9 | my choice |
| tilt termination | 10° | 10° | 30° | 30° | |
| base contact termination | – | – | – | > 1 N | legged_gym |
| reward clipped at 0 | – | – | – | yes | legged_gym |
| **P1.4** action_smoothness | – | – | – | – | −0.1 (joint targets), walk-these-ways |
| **P1.5** rec_upright / rec_height / rec_pose | – | – | – | – | 1.0 / 1.0 / 0.5 with `--recovery`, my choice |
| **P1.5** recovery episodes | – | – | – | – | 15 % with `--recovery`; no fall termination, no clipping (MetalHead) |
| **P1.6** base_height | – | – | – | – | same weight; target 0.30 + dh with `--height-cmd` (walk-these-ways) |
| **P1.6** similar_to_default | – | – | – | – | same weight; reference = IK pose of the commanded height with `--height-cmd` |
| **P3.1** base_height, foot_clearance, rec_height | – | – | – | – | same weights; measured from the ground under the feet on terrain (identical on flat ground) |
| **P3.1** tilt termination | – | – | – | – | `max_slope_deg` + 30° = 55° with `--terrain` (base contact stays the fall criterion) |
| **P3.1** off-grid termination | – | – | – | – | base below the lowest ground − 1 m, grid runs only |
| **P3.2** foot_slip | – | – | – | – | −0.1 with `--foot-slip` (Isaac Lab Spot `foot_slip_penalty`, −0.5 next to tracking 5 + 5); mode `xy` or `tangent` |

## Not done (candidates for later)

- **Student distillation.** P3.3 trains a privileged teacher (`--privileged teacher`); distilling a proprioceptive student from it (rsl_rl 5.5.1 `Distillation`, observation sets `student` / `teacher`) is not implemented.
- **Torque–speed limit.** A DC-motor curve (Isaac Lab `DCMotor`) instead of a flat clip.
- **System identification on your Go2.** Identify armature, joint friction and latency. Menagerie also uses damping and frictionloss values that are not adopted here.
- **An onnxruntime deploy path.** P3.4 exports and checks `policy.onnx`; `go2_fsm.py` still runs `policy.pt`. A C++ runtime (onnxruntime, TensorRT) would drop the PyTorch dependency on the robot computer.
- **The energy term on terrain.** P2 calibrated `--energy 1.5e-3` on flat ground; climbing needs far more power. Whether it slows the terrain curriculum is to be seen in your P3.1 runs (`TEST_AND_TINKER_P3.md` 4.5); a terrain-specific weight is not implemented.
- **Reset with forward kinematics as the default** (P3.1 found the kick on flat ground, see "Fixes to earlier phases" in the P3.1 section). The v1.0 command line sets `--set env.reset_forward=True` and the card checks it; the default stays off, so runs with default flags keep training bit for bit as in P2.
- **Stairs.** P3.1's tiles are blind-friendly (slopes, rough, waves, low obstacles). Stairs need a perception input, or a student distilled from a teacher with the height scan (P3.3 has the scan entry, not the stair tiles).
- **A Genesis ↔ DDS bridge**, to run `go2_deploy_sdk2.py` against Genesis rather than unitree_mujoco.
- **The rest of walk-these-ways' command space** (gait frequency, phase, foot swing height, stance width, body pitch). It needs gait-phase observations and their rewards (e.g. the Raibert heuristic), so it is more than one more command.
- **LCP / CAPS for recurrent policies.** `go2_smooth_ppo.py` would need rsl_rl's padded-sequence mini-batches.

## Sources

- legged_gym: [legged_robot.py](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py), [legged_robot_config.py](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py)
- unitree_rl_gym: [go2_config.py](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/go2/go2_config.py)
- Isaac Lab:
  - [velocity_env_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py)
  - Spot [rewards.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/mdp/rewards.py) and [flat_env_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/flat_env_cfg.py)
  - Go2 [flat_env_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/flat_env_cfg.py)
- MuJoCo Menagerie: [unitree_go2/go2.xml](https://github.com/google-deepmind/mujoco_menagerie/blob/main/unitree_go2/go2.xml)
- Unitree: [go2_description.urdf](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/go2_description/urdf/go2_description.urdf)
- Genesis: [Domain randomization guide](https://genesis-world.readthedocs.io/en/latest/user_guide/policy_training/best_practices/domain_randomization.html)
- Genesis 1.4.2 package source: `engine/solvers/rigid/collider/contact.py`, `options/morphs.py`, `engine/entities/rigid_entity/rigid_entity.py`, `utils/urdf.py`, `engine/solvers/rigid/abd/forward_dynamics.py`
- P1.4:
  - walk-these-ways [corl_rewards.py](https://github.com/Improbable-AI/walk-these-ways/blob/master/go1_gym/envs/rewards/corl_rewards.py) and [train.py](https://github.com/Improbable-AI/walk-these-ways/blob/master/scripts/train.py)
  - Chen et al. 2024, [Lipschitz-Constrained Policies](https://arxiv.org/abs/2410.11825), and [their code](https://github.com/zixuan417/smooth-humanoid-locomotion) (`ppo_rma.py`, `*_walk_phase_config.py`)
  - Mysore et al., [CAPS](https://arxiv.org/abs/2012.06644), and its [supplementary](http://ai.bu.edu/caps/CAPS_SmoothControl_Supplementary.pdf)
  - rsl_rl 5.5.1 `algorithms/ppo.py` (installed package)
- P1.5:
  - unitree_rl_lab Go2 deploy [config.yaml](https://github.com/unitreerobotics/unitree_rl_lab/blob/main/deploy/robots/go2/config/config.yaml), `include/FSM/State_FixStand.h`, `State_Passive.h`, `robots/go2/src/State_RLBase.cpp`
  - Unitree [go2_stand_example.cpp](https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/go2/go2_stand_example.cpp) and its [Python version](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/go2/low_level/go2_stand_example.py)
  - [go2_deploy](https://github.com/lupinjia/go2_deploy) (gamepad mapping)
  - unitree_rl_gym [deploy_real.py](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/deploy/deploy_real/deploy_real.py), `common/remote_controller.py`, `common/rotation_helper.py`
  - [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) `simulate_python/config.py` (domain 1, interface `lo`)
  - [MetalHead](https://github.com/inspirai/MetalHead) README and `legged_gym/envs/base/legged_robot.py` (`_reset_root_states_rec`, `_reset_dofs_rec`)
- P1.6:
  - walk-these-ways: `body_height_cmd` and its observation scale in [legged_robot_config.py](https://github.com/Improbable-AI/walk-these-ways/blob/master/go1_gym/envs/base/legged_robot_config.py), `_reward_jump` in [corl_rewards.py](https://github.com/Improbable-AI/walk-these-ways/blob/master/go1_gym/envs/rewards/corl_rewards.py), ranges in [train.py](https://github.com/Improbable-AI/walk-these-ways/blob/master/scripts/train.py)
  - rsl_rl 5.5.1 (installed package): `models/rnn_model.py` (`RNNModel`, `_TorchLSTMModel`, `_TorchGRUModel`), `modules/rnn.py`, `storage/rollout_storage.py` (`recurrent_mini_batch_generator`), `runners/on_policy_runner.py` (checkpoint numbering)
  - Genesis 1.4.2 (installed package): `vis/keybindings.py`, `vis/viewer.py` (`register_keybinds`, `follow_entity`), `vis/camera.py` (`start_recording`, `stop_recording`), `ext/pyrender/viewer.py` and `vis/viewer_plugins/plugins/default_controls.py` (Genesis' own keys)
  - Go2 leg lengths: `urdf/go2/urdf/go2.urdf` bundled with Genesis
  - rsl_rl 5.5.1 (installed package), for TEST_AND_TINKER.md: `utils/logger.py` (TensorBoard tags, episode buffers), `algorithms/ppo.py` (adaptive learning rate), `modules/distribution.py` (`std_type`)
- P3.1:
  - Rudin et al., [Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning](https://arxiv.org/abs/2109.11978), CoRL 2021 (game-inspired curriculum)
  - legged_gym [terrain.py](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/utils/terrain.py) (`Terrain.curiculum()`, terrain types), [legged_robot.py](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py) (`_update_terrain_curriculum`, `update_command_curriculum`), [legged_robot_config.py](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py) (terrain and command defaults)
  - Isaac Lab [curriculums.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/curriculums.py) (`terrain_levels_vel`), `isaaclab/terrains/height_field/hf_terrains.py` (pyramid slopes)
  - Genesis 1.4.2 (installed package): `options/morphs.py` (`Terrain`, `Plane`), `utils/terrain.py`, `ext/isaacgym/terrain_utils.py`, `engine/solvers/rigid/collider/support_field.py`, `mpr.py` / `gjk_support.py` (`_func_support_prism`), `rigid_entity.py` (`set_qpos`, `skip_forward`); Genesis 1.4.3 (support field skips terrains)
  - rsl_rl 5.5.1 (installed package): `runners/on_policy_runner.py` (`save` / `load` and their `infos`)
- P3.2:
  - Isaac Lab Spot [rewards.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/mdp/rewards.py) (`foot_slip_penalty`) and [flat_env_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/flat_env_cfg.py) (`foot_slip` −0.5, threshold 1.0, tracking 5.0 + 5.0)
  - Isaac Lab velocity [mdp/rewards.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py) (`feet_slide`)
  - Genesis 1.4.2 (installed package): `engine/entities/rigid_entity/rigid_entity.py` (`get_contacts`, `get_links_ang`, `get_links_net_contact_force`), `engine/solvers/rigid/collider/collider.py` (contact data: `normal`, `position`)
- P3.3:
  - Pinto et al., [Asymmetric Actor Critic for Image-Based Robot Learning](https://arxiv.org/abs/1710.06542), RSS 2018
  - Lee et al., [Learning Quadrupedal Locomotion over Challenging Terrain](https://arxiv.org/abs/2010.11251), Science Robotics 2020 (teacher–student)
  - legged_gym [legged_robot_config.py](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py) (`num_privileged_obs`, `measured_points_x` / `_y`, `height_measurements` 5.0) and [legged_robot.py](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py) (`compute_observations` heights, `_get_heights`, `quat_apply_yaw`)
  - rsl_rl 5.5.1 (installed package): `utils/utils.py` (`resolve_obs_groups`), `models/mlp_model.py` (`get_latent`, `obs_normalization`), `algorithms/ppo.py` (`construct_algorithm`), `algorithms/distillation.py` (`student` / `teacher`), `extensions/symmetry.py` (`augment_batch`, `compute_loss`), `storage/rollout_storage.py`, `runners/on_policy_runner.py` (`export_policy_to_jit`)
- P3.4:
  - rsl_rl 5.5.1 (installed package): `runners/on_policy_runner.py` (`export_policy_to_jit`, `export_policy_to_onnx`, `add_git_repo_to_log`), `models/mlp_model.py` and `models/rnn_model.py` (`as_onnx`, `_OnnxMLPModel`, `_OnnxRNNModel`: input and output names), `utils/logger.py` (`_store_code_state`)
  - torch 2.15 (installed package): `onnx/__init__.py` (`export`: `dynamo=True` and `external_data=True` by default, "versionchanged 2.9"), `onnx/_internal/exporter/_core.py` (imports `onnxscript`), `jit/_serialization.py` (`torch.jit.load` FutureWarning)
  - PyTorch 2.8 release notes draft, [pytorch-labs/torch-release-notes#17](https://github.com/pytorch-labs/torch-release-notes/pull/17) (`dynamo=True` default from 2.9, TorchScript "nearing end-of-life")
  - ONNX Runtime [Python API](https://onnxruntime.ai/docs/api/python/api_summary.html) (`InferenceSession`, `get_inputs`, `run`); onnx 1.23.2 (installed package) `external_data_helper.py` (`load_external_data_for_tensor` reads the `.data` file and marks the tensor inline)
  - GitHub CLI [gh release create](https://cli.github.com/manual/gh_release_create) (assets, `--title`, `--notes-file`, `--verify-tag`)
- P3 v2: `P2_SUMMARY.md` (your P2 results, 2026-11-06): final configuration, run names, training-noise band, two conda environments, one MuJoCo batch at a time, energy and terrain, the commands still weak.
