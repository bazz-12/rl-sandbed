# Go2 locomotion: P1.1 to P1.6 changes

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

## Not done (candidates for later)

- **Privileged critic.** The env keeps `friction`, `added_mass` and `com_shift`. Add them, plus `base_lin_vel`, to a `critic` group in rsl_rl's `obs_groups` for an asymmetric actor-critic.
- **Torque–speed limit.** A DC-motor curve (Isaac Lab `DCMotor`) instead of a flat clip.
- **System identification on your Go2.** Identify armature, joint friction and latency. Menagerie also uses damping and frictionloss values that are not adopted here.
- **ONNX export** (rsl_rl `export_policy_to_onnx`), if the robot computer should not run torch.
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
