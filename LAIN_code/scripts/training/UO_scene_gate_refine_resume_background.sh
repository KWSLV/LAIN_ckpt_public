#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/training/UO_scene_gate_refine_resume_background.sh /absolute/path/to/gate_refine_checkpoint.pt"
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CHECKPOINT="$1"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[ERROR] Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/Lain/UO_scene_gate_refine_resume}"
EXTRA_EPOCHS="${EXTRA_EPOCHS:-3}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SESSION_NAME="${SESSION_NAME:-lain_gate_resume_$TIMESTAMP}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/train_$TIMESTAMP.log}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "[ERROR] tmux is not installed." >&2
  exit 1
fi

mkdir -p "$RUN_ROOT/checkpoints" "$LOG_DIR" "$RUN_ROOT/wandb"
TRAIN_COMMAND="cd '$ROOT' && RUN_ROOT='$RUN_ROOT' EXTRA_EPOCHS='$EXTRA_EPOCHS' bash scripts/training/UO_scene_gate_refine_resume.sh '$CHECKPOINT' 2>&1 | tee -a '$LOG_FILE'"
tmux new-session -d -s "$SESSION_NAME" "$TRAIN_COMMAND"

sleep 1
if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[ERROR] Training exited immediately. Check: $LOG_FILE" >&2
  exit 1
fi

echo "[INFO] Gate refinement resumed: $SESSION_NAME"
echo "[INFO] Resume checkpoint: $CHECKPOINT"
echo "[INFO] Additional epochs: $EXTRA_EPOCHS"
echo "[INFO] Output: $RUN_ROOT"
echo "[INFO] Follow log: tail -f $LOG_FILE"
