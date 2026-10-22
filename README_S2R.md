# Go2 sim-to-real tools

- Runbook (what to run, in which order, what to look for): https://claude.ai/code/artifact/44329c17-a922-4b59-92f0-aaa4f4bc1ca0
- Code guide (what each file does and how): https://claude.ai/code/artifact/734b1dbd-5abe-4d9a-b5cf-e1e80774fd54

Changes to your existing files are tagged `[S2R]` (grep for it); the rest of your code is untouched.

## Files

| File | Env | Role |
| --- | --- | --- |
| `go2_deploy_sdk2.py` | go2deploy | runs the state machine on the robot (`--net`), in unitree_mujoco (`--sim`) or through the robot's own code path against a simulated robot (`--rehearsal`); logs every step |
| `go2_fsm.py` | both | state machine: damping, stand-up, policy, sit-down, joint test; safety checks and the L1+A tilt interlock |
| `go2_inputs.py` | both | remote decoding, command profiles, joint-test plan, scripted scenarios |
| `go2_log.py` | both | the log format shared by the three worlds |
| `go2_commands.py` | both | command deadzone (unchanged) |
| `go2_probe.py` | go2deploy | read-only check of rt/lowstate: rates, jitter, IMU, joints, feet, remote |
| `go2_mujoco_headless.py` | go2deploy | unitree_mujoco without a window: pinned base, friction, scripted remote, ground truth; robot-like imperfections (noise, delays, lost messages, payload, weak motors, pushes), wiring faults, fake robot services, video |
| `go2_sim_suite.py` | go2deploy | runs whole batteries of cases (gate, stress, faults, rehearsal, actuators) and checks each run against its expectation |
| `run_sim_gate.sh` | both | one H3 case in one command: Genesis bench, headless unitree_mujoco, comparison |
| `go2_analyze.py` | any (numpy, matplotlib) | compares logs of Genesis, unitree_mujoco and the robot; joint test; H3 verdict; CSV for go2_compare.py |
| `go2_kinematics.py` | any | leg kinematics, contact and velocity estimates used by go2_analyze.py |
| `test_go2_deploy.py` | go2deploy | 31 checks of the state machine, inputs, kinematics, log format, simulator models and DDS services, no robot needed |
| `go2_fsm_sim.py` | sandbed | Genesis bench: same scenarios and log format, `--suspended`, `--latency-ms`, `--friction`, joint losses |
| `scene_flat.xml` | - | flat-ground Go2 scene for unitree_mujoco (copy next to its `scene.xml`) |

## Order of work

Part 1, no robot needed (unitree_mujoco settings: `SIMULATE_DT = 0.002`, `scene_flat.xml` copied next to `scene.xml`):

1. `python test_go2_deploy.py --policy logs/<exp>/exported --dds`
2. Genesis bench: `python go2_fsm_sim.py -e <exp> --scenario walk --profile fwd_back --log`
3. One gate case by hand: `./run_sim_gate.sh <exp> fwd_back walk`
4. The gate: `python go2_sim_suite.py --policy logs/<exp>/exported --um ~/unitree_mujoco --suite gate --rt`
5. Stress: `--suite stress`; safety drills: `--suite faults`; joint test: `--suite actuators`
6. Robot code path: `--suite rehearsal`, then by hand with your gamepad on the simulator PC
   (`go2_mujoco_headless.py --robot-services --joystick xbox` + `go2_deploy_sdk2.py --rehearsal`)

`python go2_sim_suite.py --list --suite all` prints every case and what it must do. Each case runs twice by default
(unitree_mujoco over DDS is not deterministic); run the suite alone on the machine.

Part 2, with the robot:

7. Read only: `python go2_probe.py --net enp3s0 --seconds 60 --policy logs/<exp>/exported`
8. Hung: `python go2_deploy_sdk2.py --policy logs/<exp>/exported --net enp3s0 --zero-action --keys`
9. Staged protocol: see the runbook.

Gamepad: L1+R2 stand up, L1+A policy, L1+R1 sit down, L1+Y damping, L1+X start the `--profile`,
L1+B joint test (`--allow-joint-test`, robot hung). Keyboard with `--keys`: 1 2 3 4 5, Space = damping.
`--rt` gives the DDS threads and the control loop real-time priority (needs root or an rtprio limit).
