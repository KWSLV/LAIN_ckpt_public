#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SEED="${SEED:-66}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/Lain/UO_a1_text_only_last_r4a16_seed${SEED}}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SESSION_NAME="${SESSION_NAME:-lain_a1_text_diag_s${SEED}}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/text_contribution_$TIMESTAMP.log}"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[ERROR] tmux session already exists: $SESSION_NAME" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
COMMAND="cd '$ROOT' && SEED='$SEED' RUN_ROOT='$RUN_ROOT' bash scripts/eval/UO_a1_text_only_r4a16_contribution.sh 2>&1 | tee -a '$LOG_FILE'"
tmux new-session -d -s "$SESSION_NAME" "$COMMAND"

sleep 2
if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[ERROR] Contribution evaluation exited immediately. Check: $LOG_FILE" >&2
  exit 1
fi

echo "[INFO] Text contribution evaluation queued behind A1 training."
echo "[INFO] tmux: $SESSION_NAME"
echo "[INFO] log: $LOG_FILE"
