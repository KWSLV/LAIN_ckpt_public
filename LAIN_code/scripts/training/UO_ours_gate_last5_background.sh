#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/Lain/UO_ours_gate_last5}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/checkpoints}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SESSION_NAME="${SESSION_NAME:-lain_ours_gate_last5_$TIMESTAMP}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/train_$TIMESTAMP.log}"
RESUME="${RESUME:-}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PRINT_INTERVAL="${PRINT_INTERVAL:-500}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "[ERROR] tmux is not installed." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$RUN_ROOT/wandb"
TRAIN_COMMAND="cd '$ROOT' && RUN_ROOT='$RUN_ROOT' OUTPUT_DIR='$OUTPUT_DIR' RESUME='$RESUME' BATCH_SIZE='$BATCH_SIZE' TEST_BATCH_SIZE='$TEST_BATCH_SIZE' NUM_WORKERS='$NUM_WORKERS' PREFETCH_FACTOR='$PREFETCH_FACTOR' PRINT_INTERVAL='$PRINT_INTERVAL' CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' bash scripts/training/UO_ours_gate_last5.sh 2>&1 | tee -a '$LOG_FILE'"
tmux new-session -d -s "$SESSION_NAME" "$TRAIN_COMMAND"

sleep 1
if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[ERROR] Training exited immediately. Check: $LOG_FILE" >&2
  exit 1
fi

echo "[INFO] Delayed SceneGate training started: $SESSION_NAME"
if [[ -n "$RESUME" ]]; then
  echo "[INFO] Resume checkpoint: $RESUME"
fi
echo "[INFO] Epochs 1-15: SceneGate bypassed"
echo "[INFO] Epochs 16-20: SceneGate enabled at lr=1e-3"
echo "[INFO] Checkpoints: $OUTPUT_DIR"
echo "[INFO] Log: $LOG_FILE"
echo "[INFO] Follow: tail -f $LOG_FILE"
echo "[INFO] Attach: tmux attach -t $SESSION_NAME"
