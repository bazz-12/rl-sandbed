# Simulation suite, 2026-11-03 20:37

policy `/home/rbuche/projects/sandbed/logs/go2-p18h/exported`, scene `scene_flat.xml`, physics 2 ms, 2 runs per case. Tracking = measured / commanded velocity per profile segment; loop and state age while the robot is driven (damping left out)

| case | run | expect | result | why | tracking (measured/command) | tilt max [deg] | torque peak [%] | loop p99 [ms] | state age max [ms] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| noise1 | 1 | pass | ok | PASS | vx +0.46/+0.50 vx -0.50/-0.50 | 5 | 39 | 2.2 | 12 |
| noise1 | 2 | pass | ok | PASS | vx +0.44/+0.50 vx -0.51/-0.50 | 4 | 38 | 2.1 | 9 |
| noise2 | 1 | safe | ok | safe | vx +0.44/+0.50 vx -0.50/-0.50 | 7 | 46 | 2.1 | 15 |
| noise2 | 2 | safe | ok | safe | vx +0.44/+0.50 vx -0.49/-0.50 | 8 | 47 | 2.1 | 13 |
| cmd_delay10 | 1 | pass | ok | PASS | vx +0.43/+0.50 vx -0.48/-0.50 | 2 | 35 | 2.1 | 13 |
| cmd_delay10 | 2 | pass | ok | PASS | vx +0.44/+0.50 vx -0.48/-0.50 | 3 | 34 | 2.1 | 16 |
| cmd_delay20 | 1 | pass | ok | PASS | vx +0.43/+0.50 vx -0.47/-0.50 | 3 | 35 | 2.1 | 12 |
| cmd_delay20 | 2 | pass | ok | PASS | vx +0.43/+0.50 vx -0.47/-0.50 | 3 | 34 | 2.1 | 15 |
| cmd_delay40 | 1 | safe | ok | safe | vx +0.41/+0.50 vx -0.44/-0.50 | 3 | 34 | 2.2 | 15 |
| cmd_delay40 | 2 | safe | ok | safe | vx +0.41/+0.50 vx -0.44/-0.50 | 3 | 34 | 2.1 | 12 |
| state_delay10 | 1 | safe | ok | safe | vx +0.44/+0.50 vx -0.48/-0.50 | 2 | 34 | 2.2 | 10 |
| state_delay10 | 2 | safe | ok | safe | vx +0.45/+0.50 vx -0.49/-0.50 | 2 | 34 | 2.2 | 12 |
| drop20 | 1 | pass | ok | PASS | vx +0.45/+0.50 vx -0.49/-0.50 | 2 | 33 | 2.1 | 15 |
| drop20 | 2 | pass | ok | PASS | vx +0.45/+0.50 vx -0.50/-0.50 | 2 | 33 | 2.3 | 16 |
| payload2 | 1 | pass | ok | PASS | vx +0.46/+0.50 vx -0.50/-0.50 | 3 | 36 | 2.1 | 13 |
| payload2 | 2 | pass | ok | PASS | vx +0.46/+0.50 vx -0.50/-0.50 | 3 | 36 | 2.1 | 9 |
| payload5 | 1 | safe | ok | safe | vx +0.48/+0.50 vx -0.51/-0.50 | 4 | 42 | 2.2 | 13 |
| payload5 | 2 | safe | ok | safe | vx +0.48/+0.50 vx -0.51/-0.50 | 4 | 42 | 2.1 | 13 |
| com3cm | 1 | pass | ok | PASS | vx +0.46/+0.50 vx -0.49/-0.50 | 3 | 35 | 2.1 | 13 |
| com3cm | 2 | pass | ok | PASS | vx +0.46/+0.50 vx -0.49/-0.50 | 3 | 34 | 2.1 | 6 |
| weak90 | 1 | pass | ok | PASS | vx +0.44/+0.50 vx -0.49/-0.50 | 3 | 32 | 2.1 | 16 |
| weak90 | 2 | pass | ok | PASS | vx +0.44/+0.50 vx -0.50/-0.50 | 3 | 32 | 2.1 | 15 |
| weak75 | 1 | safe | ok | safe | vx +0.43/+0.50 vx -0.49/-0.50 | 3 | 30 | 2.1 | 11 |
| weak75 | 2 | safe | ok | safe | vx +0.43/+0.50 vx -0.49/-0.50 | 3 | 30 | 2.1 | 13 |
| push05 | 1 | safe | ok | safe | vx +0.48/+0.50 vx -0.47/-0.50 | 3 | 47 | 2.1 | 12 |
| push05 | 2 | safe | ok | safe | vx +0.45/+0.50 vx -0.47/-0.50 | 5 | 36 | 2.1 | 9 |
| pd_per_step | 1 | pass | ok | PASS | vx +0.44/+0.50 vx -0.49/-0.50 | 2 | 33 | 2.2 | 8 |
| pd_per_step | 2 | pass | ok | PASS | vx +0.44/+0.50 vx -0.48/-0.50 | 3 | 33 | 2.1 | 15 |
| offset3 | 1 | safe | ok | safe | vx +0.44/+0.50 vx -0.49/-0.50 | 2 | 34 | 2.2 | 11 |
| offset3 | 2 | safe | ok | safe | vx +0.44/+0.50 vx -0.49/-0.50 | 2 | 34 | 2.1 | 11 |
