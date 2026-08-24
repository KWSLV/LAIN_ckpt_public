#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/Lain/UO_lain_scene_gate}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/checkpoints}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SESSION_NAME="${SESSION_NAME:-lain_scene_gate_$TIMESTAMP}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/train_$TIMESTAMP.log}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "[ERROR] tmux is not installed." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$RUN_ROOT/wandb"
TRAIN_COMMAND="cd '$ROOT' && RUN_ROOT='$RUN_ROOT' OUTPUT_DIR='$OUTPUT_DIR' bash scripts/training/UO_lain_scene_gate.sh 2>&1 | tee -a '$LOG_FILE'"
tmux new-session -d -s "$SESSION_NAME" "$TRAIN_COMMAND"

sleep 1
if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[ERROR] Training exited immediately. Check: $LOG_FILE" >&2
  exit 1
fi

echo "[INFO] LAIN + SceneGate training started: $SESSION_NAME"
echo "[INFO] Checkpoints: $OUTPUT_DIR"
echo "[INFO] Log: $LOG_FILE"
echo "[INFO] Follow: tail -f $LOG_FILE"
echo "[INFO] Attach: tmux attach -t $SESSION_NAME"
