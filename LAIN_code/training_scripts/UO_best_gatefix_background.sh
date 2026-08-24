#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/Lain/UO_best_gatefix}"
CHECKPOINT_DIR="${OUTPUT_DIR:-$RUN_ROOT/checkpoints}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SESSION_NAME="${SESSION_NAME:-lain_uo_gatefix_$TIMESTAMP}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/train_$TIMESTAMP.log}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "[ERROR] tmux is not installed. Run: apt-get update && apt-get install -y tmux" >&2
  exit 1
fi

mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR" "$RUN_ROOT/wandb"

TRAIN_COMMAND="cd '$ROOT' && RUN_ROOT='$RUN_ROOT' OUTPUT_DIR='$CHECKPOINT_DIR' bash scripts/training/UO_best_gatefix.sh 2>&1 | tee -a '$LOG_FILE'"
tmux new-session -d -s "$SESSION_NAME" "$TRAIN_COMMAND"

sleep 1
if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[ERROR] Training exited immediately. Check: $LOG_FILE" >&2
  exit 1
fi

echo "[INFO] Training started in tmux session: $SESSION_NAME"
echo "[INFO] Checkpoints: $CHECKPOINT_DIR"
echo "[INFO] Terminal log: $LOG_FILE"
echo "[INFO] Attach: tmux attach -t $SESSION_NAME"
echo "[INFO] Follow log: tail -f $LOG_FILE"
