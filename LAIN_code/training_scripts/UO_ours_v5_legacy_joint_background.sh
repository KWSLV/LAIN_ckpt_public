#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/Lain/UO_ours_v5_legacy_joint}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/checkpoints}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
RESUME="${RESUME:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SESSION_NAME="${SESSION_NAME:-lain_v5_legacy_joint_$TIMESTAMP}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/train_$TIMESTAMP.log}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "[ERROR] tmux is not installed." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$RUN_ROOT/wandb"
TRAIN_COMMAND="cd '$ROOT' && RUN_ROOT='$RUN_ROOT' OUTPUT_DIR='$OUTPUT_DIR' RESUME='$RESUME' CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' bash scripts/training/UO_ours_v5_legacy_joint.sh 2>&1 | tee -a '$LOG_FILE'"
tmux new-session -d -s "$SESSION_NAME" "$TRAIN_COMMAND"

sleep 1
if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[ERROR] Training exited immediately. Check: $LOG_FILE" >&2
  exit 1
fi

echo "[INFO] Legacy collapsing V5 joint training started: $SESSION_NAME"
echo "[INFO] Checkpoints: $OUTPUT_DIR"
echo "[INFO] Log: $LOG_FILE"
echo "[INFO] Follow: tail -f $LOG_FILE"
echo "[INFO] Attach: tmux attach -t $SESSION_NAME"
