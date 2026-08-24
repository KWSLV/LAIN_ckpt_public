#!/usr/bin/env bash
set -euo pipefail

# Continue the SceneGate-only stage of UO_two_stage_last_r4a16.
#
# The checkpoint must already contain a trained v5 SceneGate. This script uses
# --resume (not --init_from), so model weights, optimizer, scheduler, epoch,
# iteration, and AMP scaler continue from the checkpoint. SceneGate is NOT reset.
#
# Usage:
#   ADDITIONAL_EPOCHS=10 \
#   RUN_ROOT=/root/autodl-tmp/Lain/my_new_run \
#   bash scripts/training/UO_two_stage_last_r4a16_gate_continue.sh \
#     /absolute/path/to/ckpt_xxxxx_02.pt

if [[ $# -ne 1 ]]; then
  echo "Usage: bash $0 /absolute/path/to/stage2_scene_gate_checkpoint.pt" >&2
  exit 2
fi

CHECKPOINT="$1"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[ERROR] Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

ADDITIONAL_EPOCHS="${ADDITIONAL_EPOCHS:-10}"
if [[ ! "$ADDITIONAL_EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] ADDITIONAL_EPOCHS must be a positive integer." >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export PATH="/root/miniconda3/bin:$PATH"
source scripts/server_assets.sh
prepare_lain_server_assets

# main.py interprets --epochs as the target TOTAL epoch count. Read the saved
# epoch and add the requested number, so epoch 2 + 10 really trains epochs 3-12.
COMPLETED_EPOCHS="$(
  /root/miniconda3/bin/python - "$CHECKPOINT" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
epoch = checkpoint.get("epoch")
if not isinstance(epoch, int) or epoch < 0:
    raise SystemExit("[ERROR] Checkpoint has no valid integer epoch.")
print(epoch)
PY
)"
TARGET_EPOCHS=$((COMPLETED_EPOCHS + ADDITIONAL_EPOCHS))

RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/Lain/UO_two_stage_last_r4a16/stage2_scene_gate_continue_from_ep${COMPLETED_EPOCHS}_plus${ADDITIONAL_EPOCHS}}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/checkpoints}"
WANDB_DIR="${WANDB_DIR:-$RUN_ROOT/wandb}"
KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-2}"
GATE_LR_OVERRIDE="${GATE_LR_OVERRIDE:-}"

BATCH_SIZE="${BATCH_SIZE:-8}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PRINT_INTERVAL="${PRINT_INTERVAL:-500}"
SEED="${SEED:-66}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ -d "$OUTPUT_DIR" ]] && find "$OUTPUT_DIR" -mindepth 1 -print -quit | grep -q .; then
  echo "[ERROR] Output directory is not empty; refusing to overwrite: $OUTPUT_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$WANDB_DIR"

# Prevent two managed LAIN jobs from sharing the same GPU.
exec 9>"/tmp/lain_cuda_${CUDA_VISIBLE_DEVICES}.lock"
if ! flock -n 9; then
  echo "[ERROR] Another managed LAIN job is using CUDA device $CUDA_VISIBLE_DEVICES." >&2
  exit 1
fi

# Also catch training processes that were started by older scripts without the
# lock above. Compute-mode GPU processes should be absent before torchrun starts.
GPU_PROCESSES="$(
  nvidia-smi \
    --query-compute-apps=pid,process_name \
    --format=csv,noheader,nounits 2>/dev/null || true
)"
if [[ -n "${GPU_PROCESSES//[[:space:]]/}" ]]; then
  echo "[ERROR] CUDA device already has a compute process; training was not started:" >&2
  echo "$GPU_PROCESSES" >&2
  exit 1
fi

PORT="${PORT:-$((RANDOM % 5000 + 15000))}"
RDZV_ID="${RDZV_ID:-$((RANDOM % 5000 + 15000))}"

echo "[Continue] Source checkpoint: $CHECKPOINT"
echo "[Continue] Completed epochs: $COMPLETED_EPOCHS"
echo "[Continue] Additional epochs: $ADDITIONAL_EPOCHS"
echo "[Continue] Target total epochs: $TARGET_EPOCHS"
echo "[Continue] Expected new epochs: $((COMPLETED_EPOCHS + 1))-$TARGET_EPOCHS"
echo "[Continue] Output: $OUTPUT_DIR"
echo "[Continue] SceneGate remains enabled and is not reset."
if [[ -n "$GATE_LR_OVERRIDE" ]]; then
  echo "[Continue] SceneGate resume LR override: $GATE_LR_OVERRIDE"
  RESUME_GATE_LR_ARGS=(
    --lr-scene-gate "$GATE_LR_OVERRIDE"
    --override-resume-scene-gate-lr
  )
else
  RESUME_GATE_LR_ARGS=(--lr-scene-gate 1e-3)
fi

WANDB_DIR="$WANDB_DIR" WANDB__SERVICE_WAIT=300 \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
torchrun \
  --rdzv_id "$RDZV_ID" \
  --rdzv_backend c10d \
  --nproc_per_node 1 \
  --rdzv_endpoint "127.0.0.1:$PORT" \
  main.py \
  --pretrained "$DETR_PRETRAINED" \
  --clip_dir_vit "$CLIP_PRETRAINED" \
  --resume "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --hico-image-root "$HICO_IMAGE_ROOT" \
  --dataset hicodet \
  --zs \
  --zs_type unseen_object \
  --num_classes 117 \
  --epochs "$TARGET_EPOCHS" \
  --keep-last-checkpoints "$KEEP_LAST_CHECKPOINTS" \
  --batch-size "$BATCH_SIZE" \
  --test-batch-size "$TEST_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --prefetch-factor "$PREFETCH_FACTOR" \
  --seed "$SEED" \
  --weight-decay 1e-4 \
  --lr-drop 10 \
  --clip-max-norm 0.1 \
  --lr-head 1e-3 \
  --lr-vit 1e-3 \
  --lr-text-adapter 5e-4 \
  --lr-obj-cond-adapter 5e-4 \
  "${RESUME_GATE_LR_ARGS[@]}" \
  --alpha 0.5 \
  --gamma 0.2 \
  --hyper_lambda 2.8 \
  --box-score-thresh 0.2 \
  --fg-iou-thresh 0.5 \
  --min-instances 3 \
  --max-instances 15 \
  --use_hotoken \
  --use_prompt \
  --use_exp \
  --CSC \
  --N_CTX 36 \
  --use_insadapter \
  --adapt_dim 32 \
  --use_prior \
  --adapter_alpha 1.0 \
  --adapter_num_layers 1 \
  --adapter_pos last \
  --use_text_adapter \
  --text_adapter_dim 64 \
  --lora_rank 4 \
  --lora_alpha 16 \
  --adapter_residual_scale 0.05 \
  --adapter_dropout 0.1 \
  --use_obj_cond_adapter \
  --obj_cond_rank 4 \
  --use_scene_gate \
  --scene_gate_type pair \
  --scene_gate_version v5 \
  --scene_gate_hidden_dim 128 \
  --scene_gate_rank 16 \
  --scene_gate_dropout 0.1 \
  --scene_gate_alpha 0.02 \
  --scene_gate_activation standardized_tanh \
  --scene_gate_init_std 1e-3 \
  --scene_gate_start_epoch 1 \
  --train_scene_gate_only \
  --scene-gate-diagnostics \
  --scene-gate-compare-off \
  --adapter-contribution-diagnostics \
  --skip-object-adapter-contribution \
  --amp \
  --fast-cuda \
  --print-interval "$PRINT_INTERVAL"
