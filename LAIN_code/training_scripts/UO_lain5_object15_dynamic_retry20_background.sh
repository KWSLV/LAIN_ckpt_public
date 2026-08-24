#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

TRAIN_SCRIPT="UO_lain5_object15_dynamic_retry20.sh"
TRAIN_NAME="${TRAIN_SCRIPT%.sh}"
SEED="${SEED:-66}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/Lain/${TRAIN_NAME}_seed${SEED}}"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$LOG_DIR"

if pgrep -af "$TRAIN_SCRIPT" | grep -v "${TRAIN_SCRIPT}_background.sh" >/dev/null; then
  echo "[ERROR] $TRAIN_SCRIPT is already running." >&2
  pgrep -af "$TRAIN_SCRIPT" >&2 || true
  exit 1
fi

LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
nohup bash "scripts/training/$TRAIN_SCRIPT" >"$LOG_FILE" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$RUN_ROOT/train.pid"

echo "Started $TRAIN_NAME"
echo "PID: $PID"
echo "Log: $LOG_FILE"

