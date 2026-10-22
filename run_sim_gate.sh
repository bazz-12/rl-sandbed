#!/usr/bin/env bash
# run_sim_gate.sh - [S2R] the H3 gate in one command: one scenario in the Genesis bench and in unitree_mujoco (headless,
# through go2_deploy_sdk2.py --sim), then go2_analyze.py on both. Run it from the folder that holds the go2_*.py files
# and logs/<exp>/exported (python go2_export.py -e <exp> --ckpt N first).
#
#     ./run_sim_gate.sh <exp> [profile] [scenario]      e.g.  ./run_sim_gate.sh go2-p16 fwd_back walk
#
# Environment (defaults in brackets):
#     GENESIS_PY      python of the Genesis env       [conda run --no-capture-output -n sandbed python]
#     DEPLOY_PY       python of the deploy env        [conda run --no-capture-output -n go2deploy python]
#     UNITREE_MUJOCO  unitree_mujoco checkout          [~/unitree_mujoco]
#     SCENE           MuJoCo scene                     [$UNITREE_MUJOCO/unitree_robots/go2/scene_flat.xml]
#     SIM_ARGS        extra go2_mujoco_headless.py options, e.g. "--foot-friction 1.0"   [--dt 0.002]
#     DEPLOY_ARGS     extra go2_deploy_sdk2.py options, e.g. "--rt"                       []
#     TAG             names the logs, plots and CSV (one per variant, e.g. gate_fwd_back_mu1)  [gate_<scenario>_<profile>]
set -euo pipefail
EXP=${1:?usage: ./run_sim_gate.sh <exp> [profile] [scenario]}
PROFILE=${2:-fwd_back}
SCENARIO=${3:-walk}
GENESIS_PY=${GENESIS_PY:-"conda run --no-capture-output -n sandbed python"}
DEPLOY_PY=${DEPLOY_PY:-"conda run --no-capture-output -n go2deploy python"}
UM=${UNITREE_MUJOCO:-$HOME/unitree_mujoco}
SCENE=${SCENE:-$UM/unitree_robots/go2/scene_flat.xml}
SIM_ARGS=${SIM_ARGS:---dt 0.002}
DEPLOY_ARGS=${DEPLOY_ARGS:-}
TAG=${TAG:-gate_${SCENARIO}_${PROFILE}}
OUT=logs/deploy
mkdir -p "$OUT"

echo "=== 1/3 Genesis bench: $SCENARIO / $PROFILE ==="
$GENESIS_PY go2_fsm_sim.py -e "$EXP" --scenario "$SCENARIO" --profile "$PROFILE" --log "$OUT/genesis_$TAG.npz"

echo "=== 2/3 unitree_mujoco (headless) + go2_deploy_sdk2.py --sim ==="
$DEPLOY_PY go2_mujoco_headless.py --um "$UM" --scene "$SCENE" $SIM_ARGS --truth "$OUT/mjtruth_$TAG.npz" \
    --exit-idle 3 --duration 600 > "$OUT/mjsim_$TAG.txt" 2>&1 &
SIM=$!
sleep 4
$DEPLOY_PY go2_deploy_sdk2.py --policy "logs/$EXP/exported" --sim --scenario "$SCENARIO" --profile "$PROFILE" \
    --yes --tag "$TAG" $DEPLOY_ARGS
wait $SIM || true
tail -n 2 "$OUT/mjsim_$TAG.txt"

echo "=== 3/3 comparison (A = Genesis, B = unitree_mujoco) ==="
MJ=$(ls -t "$OUT"/mujoco_*_"$TAG".npz | head -n 1)
$DEPLOY_PY go2_analyze.py "$OUT/genesis_$TAG.npz" "$MJ" "$OUT/mjtruth_$TAG.npz" --plots "out/$TAG" --csv "out/$TAG"
