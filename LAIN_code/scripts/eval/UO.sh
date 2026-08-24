#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/eval/UO.sh /absolute/path/to/checkpoint.pt"
  exit 2
fi

cd "$(dirname "$0")/../.."

source scripts/server_assets.sh
prepare_lain_server_assets

CHECKPOINT="$1"
PORT="${PORT:-$((RANDOM % 5000 + 5000))}"
RDZV_ID="${RDZV_ID:-$((RANDOM % 5000 + 5000))}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/Lain/UO_scene_text_obj_eval}"

mkdir -p "$OUTPUT_DIR"

WANDB_MODE=disabled CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
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
  --dataset hicodet --num_classes 117 \
  --zs --zs_type unseen_object \
  --num-workers "${NUM_WORKERS:-8}" --prefetch-factor "${PREFETCH_FACTOR:-4}" \
  --use_hotoken --use_prompt --use_exp --CSC --N_CTX 36 \
  --use_insadapter --adapt_dim 32 --use_prior --adapter_alpha 1. \
  --adapter_pos last \
  --use_text_adapter --text_adapter_dim 64 --lora_rank 8 --lora_alpha 8 \
  --adapter_residual_scale 0.05 --adapter_dropout 0.1 \
  --use_obj_cond_adapter --obj_cond_rank 4 \
  --use_scene_gate --scene_gate_type pair --scene_gate_version v5 \
  --scene_gate_hidden_dim 128 --scene_gate_rank 16 --scene_gate_dropout 0.1 \
  --scene_gate_init_std 1e-3 \
  --scene-gate-diagnostics --scene-gate-compare-off \
  --eval
