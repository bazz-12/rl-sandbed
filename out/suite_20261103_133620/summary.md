# Simulation suite, 2026-11-03 13:51

policy `/home/rbuche/projects/sandbed/logs/go2-p17h/exported`, scene `scene_flat.xml`, physics 2 ms, 2 runs per case. Tracking = measured / commanded velocity per profile segment; loop and state age while the robot is driven (damping left out)

| case | run | expect | result | why | tracking (measured/command) | tilt max [deg] | torque peak [%] | loop p99 [ms] | state age max [ms] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stand60_mu1 | 1 | pass | ok | PASS |  | 3 | 14 | 2.1 | 14 |
| stand60_mu1 | 2 | pass | ok | PASS |  | 3 | 14 | 2.2 | 14 |
| slow_mu1 | 1 | pass | NOT OK | profile seg 3: lin_err 0.16 m/s > 0.15 | vx +0.12/+0.25 vx -0.10/-0.25 | 3 | 30 | 2.1 | 13 |
| slow_mu1 | 2 | pass | NOT OK | profile seg 3: lin_err 0.20 m/s > 0.15 | vx +0.12/+0.25 vx -0.06/-0.25 | 3 | 31 | 2.2 | 11 |
| fwd_back_mu1 | 1 | pass | ok | PASS | vx +0.38/+0.50 vx -0.40/-0.50 | 3 | 36 | 2.2 | 9 |
| fwd_back_mu1 | 2 | pass | ok | PASS | vx +0.39/+0.50 vx -0.40/-0.50 | 3 | 36 | 2.1 | 15 |
| lateral_mu1 | 1 | pass | ok | PASS | vy +0.20/+0.30 vy -0.18/-0.30 | 4 | 37 | 2.1 | 13 |
| lateral_mu1 | 2 | pass | ok | PASS | vy +0.18/+0.30 vy -0.19/-0.30 | 7 | 43 | 2.1 | 11 |
| yaw_mu1 | 1 | pass | NOT OK | profile seg 1: yaw_err 0.31 rad/s > 0.2; profile seg 3: yaw_err 0.37 rad/s > 0.2 | wz +0.29/+0.60 wz -0.23/-0.60 | 3 | 47 | 2.1 | 13 |
| yaw_mu1 | 2 | pass | NOT OK | profile seg 1: yaw_err 0.31 rad/s > 0.2; profile seg 3: yaw_err 0.37 rad/s > 0.2 | wz +0.29/+0.60 wz -0.23/-0.60 | 3 | 47 | 2.1 | 9 |
| stand60_mu04 | 1 | safe | ok | safe |  | 2 | 13 | 2.1 | 12 |
| stand60_mu04 | 2 | safe | ok | safe |  | 2 | 13 | 2.1 | 15 |
| slow_mu04 | 1 | safe | ok | safe (tracking: profile seg 3: lin_err 0.15 m/s > 0.15) | vx +0.14/+0.25 vx -0.10/-0.25 | 2 | 30 | 2.1 | 13 |
| slow_mu04 | 2 | safe | ok | safe (tracking: profile seg 3: lin_err 0.16 m/s > 0.15) | vx +0.14/+0.25 vx -0.10/-0.25 | 3 | 32 | 2.1 | 13 |
| fwd_back_mu04 | 1 | safe | ok | safe | vx +0.40/+0.50 vx -0.40/-0.50 | 2 | 36 | 2.1 | 13 |
| fwd_back_mu04 | 2 | safe | ok | safe | vx +0.40/+0.50 vx -0.40/-0.50 | 2 | 37 | 2.1 | 20 |
| lateral_mu04 | 1 | safe | ok | safe | vy +0.22/+0.30 vy -0.19/-0.30 | 2 | 37 | 2.2 | 12 |
| lateral_mu04 | 2 | safe | ok | safe | vy +0.22/+0.30 vy -0.19/-0.30 | 2 | 37 | 2.1 | 12 |
| yaw_mu04 | 1 | safe | ok | safe (tracking: profile seg 1: yaw_err 0.30 rad/s > 0.2; profile seg 3: yaw_err 0.35 rad/s > 0.2) | wz +0.30/+0.60 wz -0.25/-0.60 | 3 | 47 | 2.1 | 15 |
| yaw_mu04 | 2 | safe | ok | safe (tracking: profile seg 1: yaw_err 0.30 rad/s > 0.2; profile seg 3: yaw_err 0.35 rad/s > 0.2) | wz +0.30/+0.60 wz -0.25/-0.60 | 3 | 46 | 2.2 | 12 |
