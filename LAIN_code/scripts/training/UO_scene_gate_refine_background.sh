#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/training/UO_scene_gate_refine_background.sh /absolute/path/to/best_checkpoint.pt"
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BASE_CHECKPOINT="$1"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/Lain/UO_scene_gate_refine}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SESSION_NAME="${SESSION_NAME:-lain_gate_refine_$TIMESTAMP}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/train_$TIMESTAMP.log}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "[ERROR] tmux is not installed." >&2
  exit 1
fi

mkdir -p "$RUN_ROOT/checkpoints" "$LOG_DIR" "$RUN_ROOT/wandb"
TRAIN_COMMAND="cd '$ROOT' && RUN_ROOT='$RUN_ROOT' bash scripts/training/UO_scene_gate_refine.sh '$BASE_CHECKPOINT' 2>&1 | tee -a '$LOG_FILE'"
tmux new-session -d -s "$SESSION_NAME" "$TRAIN_COMMAND"

sleep 1
if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[ERROR] Training exited immediately. Check: $LOG_FILE" >&2
  exit 1
fi

echo "[INFO] Gate refinement started: $SESSION_NAME"
echo "[INFO] Output: $RUN_ROOT"
echo "[INFO] Follow log: tail -f $LOG_FILE"
