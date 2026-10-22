#!/usr/bin/env bash
# Overnight training queue for the Go2 P1.6 experiments.
# Usage (inside tmux):  ./run_all.sh
#   - blocks suspend/idle while it runs (systemd-inhibit), released automatically at the end
#   - activates the sandbed conda env and checks CUDA before starting
#   - logs every run to nightly/<date>_<time>/<exp>.log and continues after a failure
#   - logs GPU stats once a minute to gpu.csv, writes summary.txt, sends a desktop notification at the end
# Note: go2_train.py deletes logs/<exp>/ before training, so re-running overwrites previous results.

# Re-launch under systemd-inhibit so the laptop cannot sleep while this script runs
SELF="$(readlink -f "$0")"
if [ -z "${RUN_ALL_INHIBITED:-}" ]; then
  export RUN_ALL_INHIBITED=1
  exec systemd-inhibit --what=sleep:idle --who="run_all.sh" --why="Go2 training queue" --mode=block "$SELF" "$@"
fi

cd "$(dirname "$SELF")" || exit 1

# Environment
CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-sandbed}"
# shellcheck disable=SC1090
source "$CONDA_SH" && conda activate "$ENV_NAME" || { echo "Cannot activate conda env '$ENV_NAME'"; exit 1; }
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "CUDA is not available to torch: check nvidia-smi before training"; exit 1; }

# Output folder
OUT="nightly/$(date +%F_%H%M)"
mkdir -p "$OUT"
SUMMARY="$OUT/summary.txt"
echo "Started $(date '+%F %T') | commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a) | env $ENV_NAME" > "$SUMMARY"

# GPU log, one line per minute (stopped automatically when the script exits)
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm \
  --format=csv -l 60 > "$OUT/gpu.csv" 2>/dev/null &
GPU_LOG_PID=$!
trap 'kill "$GPU_LOG_PID" 2>/dev/null' EXIT

n_ok=0
n_fail=0
run() {  # run "<label>" -e <exp> [more go2_train.py flags]
  local label="$1"; shift
  local exp="$2"
  local start rc dur
  start=$(date +%s)
  echo
  echo "===== $(date +%T)  $label  ($exp) ====="
  python go2_train.py "$@" 2>&1 | tee "$OUT/$exp.log"
  rc=${PIPESTATUS[0]}
  dur=$(( ($(date +%s) - start) / 60 ))
  if [ "$rc" -eq 0 ]; then
    n_ok=$((n_ok + 1))
    printf "OK      %-14s %5d min            %s\n" "$exp" "$dur" "$*" >> "$SUMMARY"
  else
    n_fail=$((n_fail + 1))
    printf "FAILED  %-14s %5d min  exit %-3d  %s\n" "$exp" "$dur" "$rc" "$*" >> "$SUMMARY"
  fi
}

# Queue (edit here)
run "Baseline"                      -e go2-p16
run "[1/7] height command"          -e go2-p16h    --height-cmd
run "[2/7] LSTM"                    -e go2-p16r    --rnn lstm
run "[3/7] GRU"                     -e go2-p16g    --rnn gru
run "[4/7] height + LSTM"           -e go2-p16hr   --height-cmd --rnn lstm
run "[5/7] LSTM hidden 128"         -e go2-p16r128 --rnn lstm --rnn-hidden 128
run "[6/7] LSTM, 96 steps per env"  -e go2-p16r96  --rnn lstm --steps-per-env 96
run "[7/7] height + LSTM, 1500 its" -e go2-p16hr15 --height-cmd --rnn lstm --max-iterations 1500

echo "Finished $(date '+%F %T') | $n_ok OK, $n_fail failed" >> "$SUMMARY"
echo
cat "$SUMMARY"
command -v notify-send >/dev/null && notify-send -u critical "Go2 training queue finished" "$n_ok OK, $n_fail failed"
[ "$n_fail" -eq 0 ]
