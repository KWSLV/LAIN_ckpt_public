#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export PATH="/root/miniconda3/bin:$PATH"
source scripts/server_assets.sh
prepare_lain_server_assets

SCRIPT_NAME="$(basename "$0" .sh)"
SEED="${SEED:-66}"
EPOCHS="${EPOCHS:-10}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-/root/autodl-tmp/Lain/UO_a4all_seed66/checkpoints/ckpt_35055_09.pt}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/Lain/${SCRIPT_NAME}_seed${SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/checkpoints}"
WANDB_DIR="${WANDB_DIR:-$RUN_ROOT/wandb}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PORT="${PORT:-$((RANDOM % 5000 + 15000))}"
RDZV_ID="${RDZV_ID:-$((RANDOM % 5000 + 15000))}"

if [[ ! -f "$SOURCE_CHECKPOINT" ]]; then
  echo "[ERROR] Missing A4all epoch-9 checkpoint: $SOURCE_CHECKPOINT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$WANDB_DIR" "$RUN_ROOT/logs"

exec 9>"/tmp/lain_cuda_${CUDA_VISIBLE_DEVICES}.lock"
if ! flock -n 9; then
  echo "[ERROR] Another managed LAIN job is using CUDA device $CUDA_VISIBLE_DEVICES." >&2
  exit 1
fi

echo "[$SCRIPT_NAME] Source: $SOURCE_CHECKPOINT"
echo "[$SCRIPT_NAME] Inherit: every checkpoint tensor except Text/Object/SceneGate modules"
echo "[$SCRIPT_NAME] Text: OFF"
echo "[$SCRIPT_NAME] SceneGate: OFF"
echo "[$SCRIPT_NAME] Object: rank=4, fresh initialization, the only trainable module"
echo "[$SCRIPT_NAME] Object lr=2e-4, epochs=$EPOCHS"
echo "[$SCRIPT_NAME] Checkpoint: every epoch plus best Unseen"
echo "[$SCRIPT_NAME] Diagnostics: Object ON/OFF contribution every epoch"
echo "[$SCRIPT_NAME] W&B run name: $SCRIPT_NAME"
echo "[$SCRIPT_NAME] Output: $OUTPUT_DIR"

WANDB_DIR="$WANDB_DIR" \
WANDB_NAME="$SCRIPT_NAME" \
WANDB_RUN_GROUP="A4_ep9_module_reset" \
WANDB__SERVICE_WAIT=300 \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
torchrun \
  --rdzv_id "$RDZV_ID" \
  --rdzv_backend c10d \
  --nproc_per_node 1 \
  --rdzv_endpoint "127.0.0.1:$PORT" \
  main.py \
  --pretrained "$DETR_PRETRAINED" \
  --clip_dir_vit "$CLIP_PRETRAINED" \
  --init_from "$SOURCE_CHECKPOINT" \
  --reset_obj_cond_adapter_on_load \
  --train_obj_cond_adapter_only \
  --output-dir "$OUTPUT_DIR" \
  --hico-image-root "$HICO_IMAGE_ROOT" \
  --dataset hicodet \
  --partitions train2015 test2015 \
  --zs --zs_type unseen_object \
  --num_classes 117 \
  --epochs "$EPOCHS" \
  --batch-size 8 \
  --test-batch-size 8 \
  --num-workers 8 \
  --prefetch-factor 4 \
  --seed "$SEED" \
  --weight-decay 1e-4 \
  --lr-drop 6 \
  --clip-max-norm 0.1 \
  --lr-head 5e-4 \
  --lr-vit 2e-4 \
  --lr-obj-cond-adapter 2e-4 \
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
  --adapter_pos all \
  --use_obj_cond_adapter \
  --obj_cond_rank 4 \
  --adapter-contribution-diagnostics \
  --amp \
  --fast-cuda \
  --keep-last-checkpoints 0 \
  --keep-best-unseen-checkpoint \
  --print-interval 500

echo "[$SCRIPT_NAME] Complete."
echo "[$SCRIPT_NAME] Checkpoints: $OUTPUT_DIR"
echo "[$SCRIPT_NAME] Diagnostics: $OUTPUT_DIR/scene_gate_diagnostics.json"
