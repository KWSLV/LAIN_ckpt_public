#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SEED="${SEED:-66}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/Lain/UO_a1_text_only_last_r4a16_seed${SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/checkpoints}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SESSION_NAME="${SESSION_NAME:-lain_a1_text_r4a16_s${SEED}}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/train_$TIMESTAMP.log}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "[ERROR] tmux is not installed." >&2
  exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[ERROR] tmux session already exists: $SESSION_NAME" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$RUN_ROOT/wandb"

TRAIN_COMMAND="cd '$ROOT' && SEED='$SEED' RUN_ROOT='$RUN_ROOT' OUTPUT_DIR='$OUTPUT_DIR' bash scripts/training/UO_a1_text_only_r4a16.sh 2>&1 | tee -a '$LOG_FILE'"
tmux new-session -d -s "$SESSION_NAME" "$TRAIN_COMMAND"

sleep 2
if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[ERROR] A1 exited immediately. Check: $LOG_FILE" >&2
  exit 1
fi

echo "[INFO] A1 background training started."
echo "[INFO] tmux: $SESSION_NAME"
echo "[INFO] output: $OUTPUT_DIR"
echo "[INFO] log: $LOG_FILE"
echo "[INFO] follow: tail -f $LOG_FILE"
