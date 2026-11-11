# Simulation suite, 2026-11-06 10:08

policy `/home/rbuche/projects/sandbed/logs/go2-p25-w1.5e-3/exported`, scene `scene_flat.xml`, physics 2 ms, 2 runs per case. Tracking = measured / commanded velocity per profile segment; gait = air_t [s] / diag_sync [%] per moving segment (go2_analyze.py, estimated contacts); loop and state age while the robot is driven (damping left out)

| case | run | expect | result | why | tracking (measured/command) | gait (air_t/diag_sync) | tilt max [deg] | torque peak [%] | loop p99 [ms] | state age max [ms] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| noise1 | 1 | pass | ok | PASS | vx +0.44/+0.50 vx -0.43/-0.50 | 0.29/88 0.26/90 | 5 | 37 | 4.9 | 7 |
| noise1 | 2 | pass | ok | PASS | vx +0.46/+0.50 vx -0.44/-0.50 | 0.28/87 0.26/89 | 5 | 38 | 5.2 | 8 |
| noise2 | 1 | safe | ok | safe | vx +0.47/+0.50 vx -0.44/-0.50 | 0.26/88 0.24/90 | 7 | 43 | 4.9 | 10 |
| noise2 | 2 | safe | ok | safe | vx +0.45/+0.50 vx -0.46/-0.50 | 0.25/85 0.25/86 | 6 | 44 | 5.2 | 9 |
| cmd_delay10 | 1 | pass | ok | PASS | vx +0.42/+0.50 vx -0.43/-0.50 | 0.32/88 0.28/91 | 2 | 34 | 4.4 | 9 |
| cmd_delay10 | 2 | pass | ok | PASS | vx +0.43/+0.50 vx -0.43/-0.50 | 0.31/88 0.28/92 | 2 | 34 | 4.5 | 9 |
| cmd_delay20 | 1 | pass | ok | PASS | vx +0.41/+0.50 vx -0.42/-0.50 | 0.34/88 0.30/93 | 2 | 34 | 4.5 | 7 |
| cmd_delay20 | 2 | pass | ok | PASS | vx +0.41/+0.50 vx -0.42/-0.50 | 0.34/87 0.30/94 | 2 | 34 | 4.7 | 11 |
| cmd_delay40 | 1 | safe | ok | safe | vx +0.38/+0.50 vx -0.42/-0.50 | 0.41/89 0.32/96 | 3 | 41 | 5.2 | 11 |
| cmd_delay40 | 2 | safe | ok | safe | vx +0.38/+0.50 vx -0.43/-0.50 | 0.41/90 0.32/96 | 3 | 42 | 4.8 | 8 |
| state_delay10 | 1 | safe | ok | safe | vx +0.43/+0.50 vx -0.44/-0.50 | 0.32/88 0.28/92 | 2 | 34 | 4.8 | 11 |
| state_delay10 | 2 | safe | ok | safe | vx +0.43/+0.50 vx -0.44/-0.50 | 0.32/88 0.28/92 | 2 | 34 | 5.2 | 11 |
| drop20 | 1 | pass | ok | PASS | vx +0.44/+0.50 vx -0.44/-0.50 | 0.30/89 0.27/91 | 2 | 34 | 4.8 | 13 |
| drop20 | 2 | pass | ok | PASS | vx +0.44/+0.50 vx -0.44/-0.50 | 0.30/90 0.27/91 | 2 | 34 | 5.1 | 14 |
| payload2 | 1 | pass | ok | PASS | vx +0.46/+0.50 vx -0.45/-0.50 | 0.28/88 0.25/90 | 3 | 37 | 4.6 | 10 |
| payload2 | 2 | pass | ok | PASS | vx +0.46/+0.50 vx -0.45/-0.50 | 0.28/88 0.25/90 | 2 | 37 | 4.7 | 8 |
| payload5 | 1 | safe | ok | safe | vx +0.47/+0.50 vx -0.47/-0.50 | 0.25/86 0.23/89 | 3 | 42 | 4.3 | 9 |
| payload5 | 2 | safe | ok | safe | vx +0.47/+0.50 vx -0.47/-0.50 | 0.25/86 0.23/89 | 3 | 42 | 5.2 | 10 |
| com3cm | 1 | pass | ok | PASS | vx +0.45/+0.50 vx -0.44/-0.50 | 0.29/85 0.27/86 | 2 | 35 | 4.6 | 10 |
| com3cm | 2 | pass | ok | PASS | vx +0.45/+0.50 vx -0.43/-0.50 | 0.29/86 0.27/87 | 2 | 35 | 5.9 | 11 |
| weak90 | 1 | pass | ok | PASS | vx +0.44/+0.50 vx -0.44/-0.50 | 0.30/89 0.27/90 | 3 | 33 | 5.1 | 9 |
| weak90 | 2 | pass | ok | PASS | vx +0.44/+0.50 vx -0.44/-0.50 | 0.30/89 0.26/90 | 3 | 32 | 4.1 | 6 |
| weak75 | 1 | safe | ok | safe | vx +0.43/+0.50 vx -0.45/-0.50 | 0.29/90 0.26/91 | 3 | 29 | 4.3 | 7 |
| weak75 | 2 | safe | ok | safe | vx +0.43/+0.50 vx -0.45/-0.50 | 0.29/90 0.26/90 | 3 | 29 | 4.9 | 7 |
| push05 | 1 | safe | ok | safe | vx +0.47/+0.50 vx -0.42/-0.50 | 0.29/88 0.27/89 | 3 | 34 | 4.6 | 8 |
| push05 | 2 | safe | ok | safe | vx +0.44/+0.50 vx -0.42/-0.50 | 0.28/87 0.27/90 | 7 | 42 | 4.7 | 10 |
| pd_per_step | 1 | pass | ok | PASS | vx +0.43/+0.50 vx -0.43/-0.50 | 0.30/88 0.27/90 | 2 | 34 | 4.7 | 9 |
| pd_per_step | 2 | pass | ok | PASS | vx +0.43/+0.50 vx -0.43/-0.50 | 0.31/89 0.27/91 | 2 | 34 | 4.8 | 8 |
| offset3 | 1 | safe | ok | safe | vx +0.42/+0.50 vx -0.43/-0.50 | 0.32/91 0.27/90 | 2 | 34 | 5.2 | 9 |
| offset3 | 2 | safe | ok | safe | vx +0.43/+0.50 vx -0.43/-0.50 | 0.32/91 0.27/89 | 2 | 34 | 4.9 | 10 |
