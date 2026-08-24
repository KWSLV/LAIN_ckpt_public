#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

source scripts/server_assets.sh
prepare_lain_server_assets

PORT="${PORT:-$((RANDOM % 5000 + 5000))}"
RDZV_ID="${RDZV_ID:-$((RANDOM % 5000 + 5000))}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/Lain/UO_unseen_tuned}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/checkpoints}"
WANDB_DIR="${WANDB_DIR:-$RUN_ROOT/wandb}"

mkdir -p "$OUTPUT_DIR" "$WANDB_DIR"

# Unseen-oriented configuration:
# - Lower CLIP/head learning rates preserve pretrained semantic geometry.
# - Shared 24-token prompts are used by omitting --CSC, reducing seen overfit.
# - The visual adapter stays in the last ViT block for a small visual update.
# - Low-rank text/object adapters limit memorisation of seen co-occurrences.
# - A small nonzero V5 gate initialization lets both gate layers learn early.
# - Gate OFF comparison is omitted during training to avoid a second full eval.
WANDB_DIR="$WANDB_DIR" WANDB__SERVICE_WAIT=300 \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
torchrun \
  --rdzv_id "$RDZV_ID" \
  --rdzv_backend c10d \
  --nproc_per_node 1 \
  --rdzv_endpoint "127.0.0.1:$PORT" \
  main.py \
  --pretrained "$DETR_PRETRAINED" \
  --clip_dir_vit "$CLIP_PRETRAINED" \
  --output-dir "$OUTPUT_DIR" \
  --hico-image-root "$HICO_IMAGE_ROOT" \
  --dataset hicodet \
  --zs --zs_type unseen_object \
  --num_classes 117 \
  --epochs "${EPOCHS:-20}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --prefetch-factor "${PREFETCH_FACTOR:-4}" \
  --seed "${SEED:-66}" \
  --weight-decay 1e-4 \
  --lr-drop 10 \
  --lr-vit 3e-4 \
  --lr-head 5e-4 \
  --alpha 0.5 --gamma 0.2 --hyper_lambda 2.8 \
  --box-score-thresh 0.2 --fg-iou-thresh 0.5 \
  --min-instances 3 --max-instances 15 \
  --use_hotoken --use_prompt --N_CTX 24 \
  --use_insadapter --adapt_dim 32 --use_prior --adapter_alpha 0.5 \
  --adapter_num_layers 1 --adapter_pos last \
  --use_text_adapter --text_adapter_dim 64 --lora_rank 2 --lora_alpha 4 \
  --adapter_residual_scale 0.03 --adapter_dropout 0.1 \
  --lr-text-adapter 3e-4 \
  --use_obj_cond_adapter --obj_cond_rank 2 \
  --lr-obj-cond-adapter 2e-4 \
  --use_scene_gate --scene_gate_type pair --scene_gate_version v5 \
  --scene_gate_hidden_dim 64 --scene_gate_rank 16 \
  --scene_gate_dropout 0.1 --scene_gate_init_std 1e-3 \
  --scene-gate-diagnostics --scene-gate-compare-off \
  --amp --print-interval "${PRINT_INTERVAL:-100}"
