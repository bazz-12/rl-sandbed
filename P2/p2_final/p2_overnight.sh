#!/usr/bin/env bash
# p2_overnight.sh - P2 final batch: second seeds, evaluations, comparisons, deploy checks, sim-to-sim, suites.
#
#   conda activate sandbed                         # Genesis env: training, metrics, bench, analysis
#   DRY=1 bash p2_overnight.sh                     # print the commands only (check paths first)
#   systemd-inhibit --what=sleep:idle --why="P2 overnight" nohup bash p2_overnight.sh > p2_overnight.log 2>&1 &
#   STAGES="bench mujoco suites" bash p2_overnight.sh    # only some stages (see STAGES below)
#
# Two Python environments: everything Genesis runs with the active python (sandbed); everything unitree_mujoco
# (go2_sim_suite.py, which starts go2_mujoco_headless.py and go2_deploy_sdk2.py with its own interpreter) runs with
# DEPLOY_PY, the python of the env that has unitree_sdk2py and mujoco (default: miniforge's go2deploy env).
#
# Run it from the folder with go2_*.py and logs/. Steps run one after the other (the unitree_mujoco suites need the
# machine to themselves); a failed step is recorded and the batch goes on. Trainings, evaluations and exports already
# done are skipped, so a re-run only redoes what is missing (and the quick checks).
# Everything ends up in out/p2_final/: SUMMARY.txt (one line per step), the comparison tables, analyze CSVs.
#
# The final P2 configuration (--symmetry --energy 1.5e-3) is exactly go2-p25-w1.5e-3, so that run is the final
# policy (seed 1) and go2-p25-s2 is its second seed: no separate go2-p2 retraining. go2-p23-s2 (symmetry only, seed 2)
# is trained too, so both configurations have 2 seeds.

set -u
UM=${UM:-$HOME/unitree_mujoco}                                   # unitree_mujoco checkout
DEPLOY_PY=${DEPLOY_PY:-$HOME/miniforge3/envs/go2deploy/bin/python}  # python with unitree_sdk2py + mujoco
STAGES=${STAGES:-"train eval compare deploy bench mujoco suites"}
CKPT=1999
KEEP="--symmetry --energy 1.5e-3"
FINAL=go2-p25-w1.5e-3                         # final policy, seed 1
FINAL_S2=go2-p25-s2                           # its second seed
NOEN=go2-p23                                  # same without energy: reference for "does the saving transfer?"
NOEN_S2=go2-p23-s2                            # its second seed
REF=go2-p20                                   # baseline; falls back to go2-p21-c0.45 (same reward, cap never binds)
PROFILES="slow fwd_back lateral yaw"
OUT=out/p2_final
DRY=${DRY:-0}

mkdir -p "$OUT"
SUMMARY=$OUT/SUMMARY.txt
[ "$DRY" = 1 ] || echo "=== P2 batch, started $(date '+%F %T'), stages: $STAGES ===" >> "$SUMMARY"

want() { [[ " $STAGES " == *" $1 "* ]]; }
step() {  # step "name" command...   (runs, logs OK/FAIL with duration to SUMMARY.txt)
    local name=$1; shift
    if [ "$DRY" = 1 ]; then echo "[$name] $*"; return 0; fi
    echo; echo "##### $(date '+%T') $name: $*"
    local t0=$SECONDS
    "$@"
    local rc=$?
    printf '%-8s %-55s %5d min\n' "$([ $rc = 0 ] && echo OK || echo "FAIL $rc")" "$name" $(((SECONDS - t0) / 60)) \
        >> "$SUMMARY"
    return $rc
}
note() { if [ "$DRY" = 1 ]; then echo "(note) $*"; else echo "$*" >> "$SUMMARY"; fi; }
have_ckpt() { [ -f "logs/$1/model_$CKPT.pt" ]; }
csv() { echo "logs/$1/metrics_ckpt${CKPT}_seed0${2:+_$2}.csv"; }
evaluate() {  # evaluate EXP [tag]   (skipped when its CSV exists)
    [ -f "$(csv "$1" "${2:-}")" ] && { echo "skip: $(csv "$1" "${2:-}") exists"; return 0; }
    if [ -n "${2:-}" ]; then python go2_metrics.py -e "$1" --ckpt $CKPT --keep_dr --tag "$2"
    else python go2_metrics.py -e "$1" --ckpt $CKPT; fi
}
compare_to() {  # compare_to FILE csv...   (table saved to a file)
    local file=$1; shift
    python go2_compare.py "$@" --preset gait > "$OUT/$file" 2>&1
}
analyze_to() {  # analyze_to FILE args...   (go2_analyze.py report saved to a file)
    local file=$1; shift
    python go2_analyze.py "$@" > "$OUT/$file" 2>&1
}

# ---------------------------------------------------------------------------------------------------- checks
have_ckpt "$REF" || REF=go2-p21-c0.45
for e in "$FINAL" "$NOEN" "$REF"; do
    have_ckpt "$e" || echo "WARNING: logs/$e/model_$CKPT.pt missing: its steps will fail"
done
MUJOCO=1
if want mujoco || want suites; then
    if [ ! -d "$UM/simulate_python" ]; then
        echo "WARNING: no unitree_mujoco at $UM (export UM=...): unitree_mujoco steps skipped"; MUJOCO=0
    elif ! "$DEPLOY_PY" -c "import unitree_sdk2py, mujoco, torch" 2>/dev/null; then
        echo "WARNING: $DEPLOY_PY cannot import unitree_sdk2py, mujoco and torch (export DEPLOY_PY=...):" \
             "unitree_mujoco steps skipped"; MUJOCO=0
    fi
    [ "$MUJOCO" = 1 ] || note "SKIP     unitree_mujoco steps (no simulator or no deploy python, see p2_overnight.log)"
fi

# ---------------------------------------------------------------------------------------------------- 1) training
if want train; then
    have_ckpt "$FINAL_S2" || step "train $FINAL_S2" python go2_train.py -e "$FINAL_S2" --seed 2 $KEEP
    have_ckpt "$NOEN_S2" || step "train $NOEN_S2" python go2_train.py -e "$NOEN_S2" --seed 2 --symmetry
fi

# ---------------------------------------------------------------------------------------------------- 2) evaluation
if want eval; then
    for e in "$NOEN_S2" "$FINAL_S2" "$REF" "$NOEN" "$FINAL"; do step "metrics $e" evaluate "$e"; done
    for e in "$REF" "$FINAL" "$FINAL_S2"; do step "metrics $e --keep_dr" evaluate "$e" dr; done
fi

# ---------------------------------------------------------------------------------------------------- 3) comparisons
# A = baseline, B, C = symmetry only (2 seeds), D, E = final (2 seeds)
if want compare; then
    step "compare nominal" compare_to compare_nominal.txt \
        "$(csv $REF)" "$(csv $NOEN)" "$(csv $NOEN_S2)" "$(csv $FINAL)" "$(csv $FINAL_S2)"
    step "compare training DR" compare_to compare_dr.txt "$(csv $REF dr)" "$(csv $FINAL dr)" "$(csv $FINAL_S2 dr)"
fi

# ---------------------------------------------------------------------------------------------------- 4) deploy path
# exports are also needed by the bench and unitree_mujoco stages
if want deploy || want bench || want mujoco || want suites; then
    for e in "$FINAL" "$NOEN"; do
        [ -f "logs/$e/exported/policy.pt" ] || step "export $e" python go2_export.py -e "$e" --ckpt $CKPT
    done
fi
if want deploy; then
    step "test_go2_deploy $FINAL" python test_go2_deploy.py --policy "logs/$FINAL/exported"
    step "fsm_sim full" python go2_fsm_sim.py -e "$FINAL" --out "$OUT/fsm_full.png"
    step "fsm_sim 16 robots DR" python go2_fsm_sim.py -e "$FINAL" -n 16 --dr --out "$OUT/fsm_full_dr16.png"
    step "fsm_sim estop" python go2_fsm_sim.py -e "$FINAL" --scenario estop --out "$OUT/fsm_estop.png"
fi

# ---------------------------------------------------------------------------------------------------- 5) Genesis bench
# Both policies (with and without energy) through the deploy path on the Genesis bench; each log analyzed alone, so
# the bench results exist even when unitree_mujoco cannot run.
bench() {  # bench EXP
    local e=$1 dir=logs/deploy/$1 p
    mkdir -p "$dir" "$OUT/$e"
    for p in $PROFILES; do
        step "bench $e $p" python go2_fsm_sim.py -e "$e" --scenario walk --profile "$p" \
            --log "$dir/genesis_${e}_$p.npz" --out "$OUT/$e/fsm_$p.png"
        step "analyze bench $e $p" analyze_to "$e/analyze_bench_$p.txt" "$dir/genesis_${e}_$p.npz" \
            --csv "$OUT/$e" --plots "$OUT/$e/plots_bench_$p"
    done
}
if want bench; then
    bench "$NOEN"
    bench "$FINAL"
    # does the energy term lower power through the deploy path? (A = without energy, B = with)
    for p in $PROFILES; do
        step "energy in bench $p" compare_to "energy_in_bench_$p.txt" \
            "$OUT/$NOEN/metrics_genesis_${NOEN}_$p.csv" "$OUT/$FINAL/metrics_genesis_${FINAL}_$p.csv"
    done
fi

# ---------------------------------------------------------------------------------------------------- 6) unitree_mujoco
# Same profiles in unitree_mujoco (go2_sim_suite.py with the deploy python), then each MuJoCo log against its bench log
s2s() {  # s2s EXP
    local e=$1 dir=logs/deploy/$1 p mj cases=""
    mkdir -p "$dir" "$OUT/$e"
    for p in $PROFILES; do cases="$cases${cases:+,}${p}_mu1"; done
    step "unitree_mujoco $e ($cases)" "$DEPLOY_PY" go2_sim_suite.py --policy "logs/$e/exported" --um "$UM" \
        --only "$cases" --repeat 1 --log-dir "$dir" --out "$OUT/$e/suite"
    for p in $PROFILES; do
        mj=$(ls -t "$dir"/mujoco_*_suite_${p}_mu1_0.npz 2>/dev/null | head -1)
        [ "$DRY" = 1 ] && mj="$dir/mujoco_<stamp>_suite_${p}_mu1_0.npz"
        if [ -z "$mj" ]; then note "FAIL     no unitree_mujoco log for $e $p (see $OUT/$e/suite)"; continue; fi
        if [ ! -f "$dir/genesis_${e}_$p.npz" ] && [ "$DRY" != 1 ]; then
            note "FAIL     no bench log for $e $p: run the bench stage first"; continue
        fi
        step "analyze bench vs mujoco $e $p" analyze_to "$e/analyze_$p.txt" "$dir/genesis_${e}_$p.npz" "$mj" \
            "$dir/mjtruth_suite_${p}_mu1_0.npz" --csv "$OUT/$e" --plots "$OUT/$e/plots_$p"
        step "compare bench vs mujoco $e $p" compare_to "$e/bench_vs_mujoco_$p.txt" \
            "$OUT/$e/metrics_genesis_${e}_$p.csv" "$OUT/$e/metrics_$(basename "$mj" .npz).csv"
        [ "$DRY" = 1 ] || cp "$OUT/$e/metrics_$(basename "$mj" .npz).csv" "$OUT/$e/mujoco_$p.csv"
    done
}
if want mujoco && [ "$MUJOCO" = 1 ]; then
    s2s "$NOEN"
    s2s "$FINAL"
    # the question of P2.5: does the energy saving survive in unitree_mujoco? (A = without energy, B = with)
    for p in $PROFILES; do
        step "energy in mujoco $p" compare_to "energy_in_mujoco_$p.txt" \
            "$OUT/$NOEN/mujoco_$p.csv" "$OUT/$FINAL/mujoco_$p.csv"
    done
fi

# ---------------------------------------------------------------------------------------------------- 7) suites
if want suites && [ "$MUJOCO" = 1 ]; then
    step "suite gate" "$DEPLOY_PY" go2_sim_suite.py --policy "logs/$FINAL/exported" --um "$UM" --suite gate \
        --out "$OUT/suite_gate"
    step "suite stress" "$DEPLOY_PY" go2_sim_suite.py --policy "logs/$FINAL/exported" --um "$UM" --suite stress \
        --out "$OUT/suite_stress"
fi

[ "$DRY" = 1 ] || echo "=== finished $(date '+%F %T') ===" >> "$SUMMARY"
[ "$DRY" = 1 ] || cat "$SUMMARY"
