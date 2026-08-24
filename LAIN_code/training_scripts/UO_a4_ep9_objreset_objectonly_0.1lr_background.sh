#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TRAIN_SCRIPT="UO_a4_ep9_objreset_objectonly_0.1lr.sh"
TRAIN_NAME="${TRAIN_SCRIPT%.sh}"
SEED="${SEED:-66}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/Lain/${TRAIN_NAME}_seed${SEED}}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SESSION_NAME="${SESSION_NAME:-lain_a4_ep9_objreset_objectonly_0_1lr_s${SEED}}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/train_$TIMESTAMP.log}"

mkdir -p "$LOG_DIR"
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[ERROR] tmux session already exists: $SESSION_NAME" >&2
  exit 1
fi

COMMAND="cd '$ROOT' && SEED='$SEED' RUN_ROOT='$RUN_ROOT' bash 'scripts/training/$TRAIN_SCRIPT' 2>&1 | tee -a '$LOG_FILE'"
tmux new-session -d -s "$SESSION_NAME" "$COMMAND"

sleep 2
if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[ERROR] Training exited immediately. Check: $LOG_FILE" >&2
  exit 1
fi

echo "[INFO] Background training started."
echo "[INFO] W&B run name: $TRAIN_NAME"
echo "[INFO] tmux: $SESSION_NAME"
echo "[INFO] log: $LOG_FILE"
