# Simulation suite, 2026-11-06 09:33

policy `/home/rbuche/projects/sandbed/logs/go2-p25-w1.5e-3/exported`, scene `scene_flat.xml`, physics 2 ms, 1 runs per case. Tracking = measured / commanded velocity per profile segment; gait = air_t [s] / diag_sync [%] per moving segment (go2_analyze.py, estimated contacts); loop and state age while the robot is driven (damping left out)

| case | run | expect | result | why | tracking (measured/command) | gait (air_t/diag_sync) | tilt max [deg] | torque peak [%] | loop p99 [ms] | state age max [ms] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| slow_mu1 | 1 | pass | ok | PASS | vx +0.16/+0.25 vx -0.15/-0.25 | 0.26/84 0.26/90 | 2 | 31 | 5.0 | 8 |
| fwd_back_mu1 | 1 | pass | ok | PASS | vx +0.44/+0.50 vx -0.44/-0.50 | 0.30/89 0.27/90 | 2 | 34 | 5.1 | 9 |
| lateral_mu1 | 1 | pass | ok | PASS | vy +0.22/+0.30 vy -0.23/-0.30 | 0.27/90 0.27/90 | 2 | 37 | 5.2 | 9 |
| yaw_mu1 | 1 | pass | ok | PASS | wz +0.54/+0.60 wz -0.54/-0.60 | 0.22/84 0.22/85 | 2 | 43 | 5.3 | 10 |
