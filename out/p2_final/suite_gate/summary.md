# Simulation suite, 2026-11-06 09:49

policy `/home/rbuche/projects/sandbed/logs/go2-p25-w1.5e-3/exported`, scene `scene_flat.xml`, physics 2 ms, 2 runs per case. Tracking = measured / commanded velocity per profile segment; gait = air_t [s] / diag_sync [%] per moving segment (go2_analyze.py, estimated contacts); loop and state age while the robot is driven (damping left out)

| case | run | expect | result | why | tracking (measured/command) | gait (air_t/diag_sync) | tilt max [deg] | torque peak [%] | loop p99 [ms] | state age max [ms] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stand60_mu1 | 1 | pass | ok | PASS |  |  | 2 | 12 | 4.7 | 11 |
| stand60_mu1 | 2 | pass | ok | PASS |  |  | 2 | 12 | 5.0 | 9 |
| slow_mu1 | 1 | pass | ok | PASS | vx +0.16/+0.25 vx -0.13/-0.25 | 0.26/84 0.24/86 | 3 | 33 | 4.6 | 9 |
| slow_mu1 | 2 | pass | ok | PASS | vx +0.16/+0.25 vx -0.15/-0.25 | 0.26/84 0.26/88 | 2 | 31 | 4.9 | 8 |
| fwd_back_mu1 | 1 | pass | ok | PASS | vx +0.44/+0.50 vx -0.44/-0.50 | 0.29/89 0.26/89 | 2 | 34 | 4.3 | 7 |
| fwd_back_mu1 | 2 | pass | ok | PASS | vx +0.45/+0.50 vx -0.45/-0.50 | 0.29/90 0.26/89 | 2 | 34 | 4.5 | 8 |
| lateral_mu1 | 1 | pass | ok | PASS | vy +0.22/+0.30 vy -0.23/-0.30 | 0.26/90 0.26/91 | 2 | 39 | 4.6 | 10 |
| lateral_mu1 | 2 | pass | ok | PASS | vy +0.22/+0.30 vy -0.23/-0.30 | 0.26/91 0.27/92 | 2 | 38 | 4.5 | 8 |
| yaw_mu1 | 1 | pass | ok | PASS | wz +0.54/+0.60 wz -0.54/-0.60 | 0.21/87 0.21/86 | 2 | 43 | 4.6 | 9 |
| yaw_mu1 | 2 | pass | ok | PASS | wz +0.54/+0.60 wz -0.54/-0.60 | 0.21/87 0.21/87 | 2 | 43 | 4.5 | 9 |
| stand60_mu04 | 1 | safe | ok | safe |  |  | 1 | 11 | 5.1 | 8 |
| stand60_mu04 | 2 | safe | ok | safe |  |  | 1 | 11 | 4.8 | 14 |
| slow_mu04 | 1 | safe | ok | safe | vx +0.18/+0.25 vx -0.15/-0.25 | 0.25/83 0.25/90 | 2 | 30 | 4.8 | 8 |
| slow_mu04 | 2 | safe | ok | safe | vx +0.18/+0.25 vx -0.15/-0.25 | 0.25/83 0.25/90 | 2 | 29 | 4.9 | 10 |
| fwd_back_mu04 | 1 | safe | ok | safe | vx +0.45/+0.50 vx -0.45/-0.50 | 0.29/89 0.26/90 | 2 | 35 | 4.9 | 8 |
| fwd_back_mu04 | 2 | safe | ok | safe | vx +0.45/+0.50 vx -0.44/-0.50 | 0.30/88 0.27/91 | 2 | 35 | 5.0 | 7 |
| lateral_mu04 | 1 | safe | ok | safe | vy +0.23/+0.30 vy -0.24/-0.30 | 0.27/92 0.26/90 | 2 | 37 | 4.8 | 8 |
| lateral_mu04 | 2 | safe | ok | safe | vy +0.23/+0.30 vy -0.24/-0.30 | 0.27/91 0.26/91 | 2 | 38 | 4.5 | 7 |
| yaw_mu04 | 1 | safe | ok | safe | wz +0.56/+0.60 wz -0.56/-0.60 | 0.20/87 0.21/77 | 2 | 46 | 4.4 | 8 |
| yaw_mu04 | 2 | safe | ok | safe | wz +0.56/+0.60 wz -0.56/-0.60 | 0.21/84 0.21/86 | 2 | 40 | 5.0 | 10 |
