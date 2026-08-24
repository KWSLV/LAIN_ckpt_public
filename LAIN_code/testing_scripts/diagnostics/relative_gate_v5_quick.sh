#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/diagnostics/relative_gate_v5_quick.sh /absolute/path/to/checkpoint.pt"
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

source scripts/server_assets.sh
prepare_lain_server_assets

CHECKPOINT="$1"
PORT="${PORT:-$((RANDOM % 5000 + 5000))}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/Lain/UO_scene_gate_only}"

WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
torchrun --rdzv_id "$RANDOM" --rdzv_backend c10d --nproc_per_node 1 \
  --rdzv_endpoint "127.0.0.1:$PORT" \
  main.py \
  --pretrained "$DETR_PRETRAINED" \
  --clip_dir_vit "$CLIP_PRETRAINED" \
  --init_from "$CHECKPOINT" --output-dir "$OUTPUT_DIR" \
  --hico-image-root "$HICO_IMAGE_ROOT" \
  --dataset hicodet --zs --zs_type unseen_object --num_classes 117 \
  --epochs "${EPOCHS:-3}" --batch-size "${BATCH_SIZE:-8}" \
  --num-workers "${NUM_WORKERS:-4}" --prefetch-factor 4 \
  --use_hotoken --use_prompt --use_exp --CSC --N_CTX 36 \
  --use_insadapter --adapt_dim 32 --use_prior --adapter_alpha 1. --adapter_pos last \
  --use_text_adapter --text_adapter_dim 64 --lora_rank 2 --lora_alpha 8 \
  --use_obj_cond_adapter --obj_cond_rank 4 \
  --use_scene_gate --scene_gate_type pair --scene_gate_version v5 \
  --scene_gate_hidden_dim 128 --scene_gate_dropout 0.1 \
  --train_scene_gate_only --scene-gate-diagnostics --scene-gate-compare-off \
  --amp --print-interval "${PRINT_INTERVAL:-100}"
