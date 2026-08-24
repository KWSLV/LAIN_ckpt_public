#!/usr/bin/env bash
set -euo pipefail

# Start UO_two_stage_last_r4a16_gate_continue.sh in a new tmux session.

if [[ $# -ne 1 ]]; then
  echo "Usage: bash $0 /absolute/path/to/stage2_scene_gate_checkpoint.pt" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CHECKPOINT="$1"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[ERROR] Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

ADDITIONAL_EPOCHS="${ADDITIONAL_EPOCHS:-10}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/Lain/UO_two_stage_last_r4a16/stage2_scene_gate_continue_from_ep02_plus10}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/checkpoints}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SESSION_NAME="${SESSION_NAME:-lain_last_r4a16_gate_continue_$TIMESTAMP}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/train_$TIMESTAMP.log}"

BATCH_SIZE="${BATCH_SIZE:-8}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PRINT_INTERVAL="${PRINT_INTERVAL:-500}"
KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-2}"
GATE_LR_OVERRIDE="${GATE_LR_OVERRIDE:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "[ERROR] tmux is not installed." >&2
  exit 1
fi
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[ERROR] tmux session already exists: $SESSION_NAME" >&2
  exit 1
fi
if [[ -d "$OUTPUT_DIR" ]] && find "$OUTPUT_DIR" -mindepth 1 -print -quit | grep -q .; then
  echo "[ERROR] Output directory is not empty; refusing to overwrite: $OUTPUT_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$RUN_ROOT/wandb"

TRAIN_COMMAND="cd '$ROOT' && \
RUN_ROOT='$RUN_ROOT' OUTPUT_DIR='$OUTPUT_DIR' \
ADDITIONAL_EPOCHS='$ADDITIONAL_EPOCHS' \
BATCH_SIZE='$BATCH_SIZE' TEST_BATCH_SIZE='$TEST_BATCH_SIZE' \
NUM_WORKERS='$NUM_WORKERS' PREFETCH_FACTOR='$PREFETCH_FACTOR' \
PRINT_INTERVAL='$PRINT_INTERVAL' KEEP_LAST_CHECKPOINTS='$KEEP_LAST_CHECKPOINTS' \
GATE_LR_OVERRIDE='$GATE_LR_OVERRIDE' \
CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' \
bash scripts/training/UO_two_stage_last_r4a16_gate_continue.sh '$CHECKPOINT' \
2>&1 | tee -a '$LOG_FILE'"

tmux new-session -d -s "$SESSION_NAME" "$TRAIN_COMMAND"

sleep 2
if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[ERROR] Training exited immediately. Check: $LOG_FILE" >&2
  exit 1
fi

echo "[INFO] SceneGate continuation started in tmux: $SESSION_NAME"
echo "[INFO] Resume checkpoint: $CHECKPOINT"
echo "[INFO] Additional epochs requested: $ADDITIONAL_EPOCHS"
echo "[INFO] SceneGate resume LR override: ${GATE_LR_OVERRIDE:-unchanged}"
echo "[INFO] Epoch range will be calculated from checkpoint metadata."
echo "[INFO] Original checkpoint directory is preserved."
echo "[INFO] New output: $OUTPUT_DIR"
echo "[INFO] Follow log: tail -f $LOG_FILE"
