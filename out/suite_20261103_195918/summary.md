# Simulation suite, 2026-11-03 20:14

policy `/home/rbuche/projects/sandbed/logs/go2-p18h/exported`, scene `scene_flat.xml`, physics 2 ms, 2 runs per case. Tracking = measured / commanded velocity per profile segment; loop and state age while the robot is driven (damping left out)

| case | run | expect | result | why | tracking (measured/command) | tilt max [deg] | torque peak [%] | loop p99 [ms] | state age max [ms] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stand60_mu1 | 1 | pass | ok | PASS |  | 2 | 13 | 2.1 | 17 |
| stand60_mu1 | 2 | pass | ok | PASS |  | 2 | 13 | 2.1 | 13 |
| slow_mu1 | 1 | pass | ok | PASS | vx +0.18/+0.25 vx -0.19/-0.25 | 3 | 29 | 2.1 | 13 |
| slow_mu1 | 2 | pass | ok | PASS | vx +0.18/+0.25 vx -0.19/-0.25 | 3 | 28 | 2.2 | 14 |
| fwd_back_mu1 | 1 | pass | ok | PASS | vx +0.45/+0.50 vx -0.50/-0.50 | 2 | 33 | 2.1 | 12 |
| fwd_back_mu1 | 2 | pass | ok | PASS | vx +0.45/+0.50 vx -0.50/-0.50 | 2 | 33 | 2.2 | 12 |
| lateral_mu1 | 1 | pass | ok | PASS | vy +0.23/+0.30 vy -0.23/-0.30 | 3 | 39 | 2.1 | 11 |
| lateral_mu1 | 2 | pass | ok | PASS | vy +0.23/+0.30 vy -0.23/-0.30 | 3 | 39 | 2.2 | 11 |
| yaw_mu1 | 1 | pass | ok | PASS | wz +0.60/+0.60 wz -0.51/-0.60 | 3 | 52 | 2.1 | 17 |
| yaw_mu1 | 2 | pass | ok | PASS | wz +0.60/+0.60 wz -0.51/-0.60 | 2 | 53 | 2.1 | 13 |
| stand60_mu04 | 1 | safe | ok | safe |  | 2 | 12 | 2.2 | 9 |
| stand60_mu04 | 2 | safe | ok | safe |  | 2 | 12 | 2.1 | 13 |
| slow_mu04 | 1 | safe | ok | safe | vx +0.20/+0.25 vx -0.20/-0.25 | 3 | 28 | 2.1 | 8 |
| slow_mu04 | 2 | safe | ok | safe | vx +0.20/+0.25 vx -0.19/-0.25 | 3 | 28 | 2.2 | 14 |
| fwd_back_mu04 | 1 | safe | ok | safe | vx +0.46/+0.50 vx -0.50/-0.50 | 2 | 34 | 2.1 | 13 |
| fwd_back_mu04 | 2 | safe | ok | safe | vx +0.46/+0.50 vx -0.50/-0.50 | 2 | 34 | 2.1 | 14 |
| lateral_mu04 | 1 | safe | ok | safe | vy +0.24/+0.30 vy -0.24/-0.30 | 3 | 39 | 2.3 | 12 |
| lateral_mu04 | 2 | safe | ok | safe | vy +0.24/+0.30 vy -0.24/-0.30 | 3 | 39 | 2.1 | 10 |
| yaw_mu04 | 1 | safe | ok | safe | wz +0.63/+0.60 wz -0.53/-0.60 | 3 | 45 | 2.1 | 14 |
| yaw_mu04 | 2 | safe | ok | safe | wz +0.63/+0.60 wz -0.53/-0.60 | 3 | 42 | 2.2 | 8 |
