# Simulation suite, 2026-11-06 09:30

policy `/home/rbuche/projects/sandbed/logs/go2-p23/exported`, scene `scene_flat.xml`, physics 2 ms, 1 runs per case. Tracking = measured / commanded velocity per profile segment; gait = air_t [s] / diag_sync [%] per moving segment (go2_analyze.py, estimated contacts); loop and state age while the robot is driven (damping left out)

| case | run | expect | result | why | tracking (measured/command) | gait (air_t/diag_sync) | tilt max [deg] | torque peak [%] | loop p99 [ms] | state age max [ms] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| slow_mu1 | 1 | pass | ok | PASS | vx +0.15/+0.25 vx -0.12/-0.25 | 0.22/82 0.23/84 | 2 | 32 | 2.3 | 7 |
| fwd_back_mu1 | 1 | pass | ok | PASS | vx +0.42/+0.50 vx -0.45/-0.50 | 0.25/85 0.25/92 | 2 | 34 | 5.0 | 11 |
| lateral_mu1 | 1 | pass | ok | PASS | vy +0.24/+0.30 vy -0.25/-0.30 | 0.24/80 0.24/81 | 3 | 36 | 5.7 | 11 |
| yaw_mu1 | 1 | pass | ok | PASS | wz +0.56/+0.60 wz -0.56/-0.60 | 0.24/85 0.24/84 | 2 | 39 | 5.0 | 14 |
