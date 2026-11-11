# Simulation suite, 2026-11-03 10:52

policy `/home/rbuche/projects/sandbed/logs/go2-p16/exported`, scene `scene_flat.xml`, physics 2 ms, 2 runs per case. Tracking = measured / commanded velocity per profile segment; loop and state age while the robot is driven (damping left out)

| case | run | expect | result | why | tracking (measured/command) | tilt max [deg] | torque peak [%] | loop p99 [ms] | state age max [ms] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stand60_mu1 | 1 | pass | ok | PASS |  | 2 | 18 | 2.1 | 12 |
| stand60_mu1 | 2 | pass | ok | PASS |  | 2 | 18 | 2.3 | 14 |
| slow_mu1 | 1 | pass | NOT OK | profile seg 1: lin_err 0.22 m/s > 0.15; profile seg 3: lin_err 0.25 m/s > 0.15 | vx +0.03/+0.25 vx -0.00/-0.25 | 2 | 35 | 2.1 | 10 |
| slow_mu1 | 2 | pass | NOT OK | profile seg 1: lin_err 0.21 m/s > 0.15; profile seg 3: lin_err 0.25 m/s > 0.15 | vx +0.04/+0.25 vx -0.00/-0.25 | 7 | 72 | 2.1 | 13 |
| fwd_back_mu1 | 1 | pass | ok | PASS | vx +0.41/+0.50 vx -0.37/-0.50 | 2 | 33 | 2.2 | 10 |
| fwd_back_mu1 | 2 | pass | ok | PASS | vx +0.41/+0.50 vx -0.37/-0.50 | 2 | 33 | 2.1 | 11 |
| lateral_mu1 | 1 | pass | ok | PASS | vy +0.19/+0.30 vy -0.16/-0.30 | 2 | 46 | 2.1 | 11 |
| lateral_mu1 | 2 | pass | ok | PASS | vy +0.19/+0.30 vy -0.16/-0.30 | 2 | 47 | 2.1 | 13 |
| yaw_mu1 | 1 | pass | NOT OK | profile seg 1: yaw_err 0.58 rad/s > 0.2; profile seg 3: yaw_err 0.46 rad/s > 0.2 | wz +0.02/+0.60 wz -0.14/-0.60 | 2 | 39 | 2.1 | 13 |
| yaw_mu1 | 2 | pass | NOT OK | profile seg 1: yaw_err 0.58 rad/s > 0.2; profile seg 3: yaw_err 0.41 rad/s > 0.2 | wz +0.02/+0.60 wz -0.19/-0.60 | 2 | 43 | 2.2 | 14 |
| stand60_mu04 | 1 | safe | ok | safe |  | 2 | 18 | 2.1 | 14 |
| stand60_mu04 | 2 | safe | ok | safe |  | 2 | 18 | 2.1 | 11 |
| slow_mu04 | 1 | safe | ok | safe (tracking: profile seg 3: lin_err 0.25 m/s > 0.15) | vx +0.16/+0.25 vx +0.00/-0.25 | 2 | 27 | 2.1 | 9 |
| slow_mu04 | 2 | safe | ok | safe (tracking: profile seg 3: lin_err 0.25 m/s > 0.15) | vx +0.17/+0.25 vx -0.00/-0.25 | 2 | 28 | 2.2 | 13 |
| fwd_back_mu04 | 1 | safe | ok | safe | vx +0.45/+0.50 vx -0.42/-0.50 | 2 | 34 | 2.1 | 15 |
| fwd_back_mu04 | 2 | safe | ok | safe | vx +0.45/+0.50 vx -0.42/-0.50 | 2 | 33 | 2.2 | 12 |
| lateral_mu04 | 1 | safe | ok | safe | vy +0.22/+0.30 vy -0.20/-0.30 | 2 | 43 | 2.1 | 13 |
| lateral_mu04 | 2 | safe | ok | safe | vy +0.22/+0.30 vy -0.20/-0.30 | 2 | 43 | 2.1 | 12 |
| yaw_mu04 | 1 | safe | ok | safe (tracking: profile seg 1: yaw_err 0.34 rad/s > 0.2; profile seg 3: yaw_err 0.28 rad/s > 0.2) | wz +0.26/+0.60 wz -0.32/-0.60 | 2 | 38 | 2.1 | 13 |
| yaw_mu04 | 2 | safe | ok | safe (tracking: profile seg 1: yaw_err 0.33 rad/s > 0.2; profile seg 3: yaw_err 0.28 rad/s > 0.2) | wz +0.27/+0.60 wz -0.32/-0.60 | 2 | 39 | 2.2 | 10 |
