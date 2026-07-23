#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

LOG_DIR="$REPO_DIR/exp/queued_runs"
mkdir -p "$LOG_DIR"

STAMP="$(date +%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/train_3ch_${STAMP}.log"

mapfile -t TRAIN_PIDS < <(
  pgrep -af "python.*train.py" \
    | grep -v "queue_train_3ch_after_current.sh" \
    | awk '{print $1}'
)

if [ "${#TRAIN_PIDS[@]}" -gt 0 ]; then
  echo "[QUEUE] Waiting for current train.py PID(s): ${TRAIN_PIDS[*]}" | tee -a "$LOG_FILE"
  for pid in "${TRAIN_PIDS[@]}"; do
    while kill -0 "$pid" 2>/dev/null; do
      sleep 60
    done
    echo "[QUEUE] PID $pid finished." | tee -a "$LOG_FILE"
  done
else
  echo "[QUEUE] No running train.py process was found. Starting 3-channel training now." | tee -a "$LOG_FILE"
fi

echo "[QUEUE] Starting 3-channel training at $(date)" | tee -a "$LOG_FILE"
echo "[QUEUE] Command: python train.py --use_channels 0,1,2" | tee -a "$LOG_FILE"

python train.py --use_channels 0,1,2 2>&1 | tee -a "$LOG_FILE"

echo "[QUEUE] Finished 3-channel training at $(date)" | tee -a "$LOG_FILE"
