#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/Lain/UO_a4all_seed66}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/gate_off_checkpoint_eval/logs}"
SESSION_NAME="${SESSION_NAME:-lain_a4all_gate_off_eval_s66}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$LOG_DIR/batch_$TIMESTAMP.log}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "[ERROR] tmux is not installed." >&2
  exit 1
fi
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[ERROR] tmux session already exists: $SESSION_NAME" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
COMMAND="cd '$ROOT' && RUN_ROOT='$RUN_ROOT' bash scripts/eval/UO_a4all_gate_off_checkpoints.sh 2>&1 | tee -a '$LOG_FILE'"
tmux new-session -d -s "$SESSION_NAME" "$COMMAND"
sleep 2

if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[ERROR] Gate-OFF checkpoint evaluation exited immediately. Check: $LOG_FILE" >&2
  exit 1
fi

echo "[INFO] Gate-OFF checkpoint evaluation queued in background."
echo "[INFO] tmux: $SESSION_NAME"
echo "[INFO] log: $LOG_FILE"
