#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SEED="${SEED:-66}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/Lain/UO_a2_object_only_last_r4_seed${SEED}}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SESSION_NAME="${SESSION_NAME:-lain_a2_object_r4_s${SEED}}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/train_$TIMESTAMP.log}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "[ERROR] tmux is not installed." >&2
  exit 1
fi
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[ERROR] tmux session already exists: $SESSION_NAME" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
COMMAND="cd '$ROOT' && SEED='$SEED' RUN_ROOT='$RUN_ROOT' bash scripts/training/UO_a2_a3_common.sh a2 2>&1 | tee -a '$LOG_FILE'"
tmux new-session -d -s "$SESSION_NAME" "$COMMAND"

sleep 2
if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[ERROR] A2 exited immediately. Check: $LOG_FILE" >&2
  exit 1
fi

echo "[INFO] A2 background training started."
echo "[INFO] tmux: $SESSION_NAME"
echo "[INFO] log: $LOG_FILE"
