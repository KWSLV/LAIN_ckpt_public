#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SEED="${SEED:-66}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/Lain/UO_objectbest_gateonly_v5h128_a004_seed${SEED}}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SESSION_NAME="${SESSION_NAME:-lain_wait5h_objectbest_gate_a004_s${SEED}}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/deferred_launcher_$TIMESTAMP.log}"
WAIT_SCRIPT="scripts/training/UO_objectbest_gateonly_v5h128_a004_wait5h.sh"

mkdir -p "$LOG_DIR"
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[ERROR] Deferred launcher tmux session already exists: $SESSION_NAME" >&2
  exit 1
fi

COMMAND="cd '$ROOT' && SEED='$SEED' RUN_ROOT='$RUN_ROOT' bash '$WAIT_SCRIPT' 2>&1 | tee -a '$LOG_FILE'"
tmux new-session -d -s "$SESSION_NAME" "$COMMAND"

sleep 2
if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[ERROR] Deferred launcher exited immediately. Check: $LOG_FILE" >&2
  exit 1
fi

echo "[INFO] Deferred Gate launcher started."
echo "[INFO] It will wait 5 hours, then check server activity every 10 minutes."
echo "[INFO] tmux: $SESSION_NAME"
echo "[INFO] log: $LOG_FILE"
