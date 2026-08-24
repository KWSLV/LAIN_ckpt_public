#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
source scripts/server_assets.sh
prepare_lain_server_assets

PORT="${PORT:-$((RANDOM % 5000 + 5000))}"
RDZV_ID="${RDZV_ID:-$((RANDOM % 5000 + 5000))}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/Lain/UO_ours_joint}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/checkpoints}"
WANDB_DIR="${WANDB_DIR:-$RUN_ROOT/wandb}"

mkdir -p "$OUTPUT_DIR" "$WANDB_DIR"
if compgen -G "$OUTPUT_DIR/ckpt_*.pt" >/dev/null; then
  echo "[ERROR] Refusing to overwrite an existing from-scratch run: $OUTPUT_DIR" >&2
  echo "[ERROR] Set RUN_ROOT to a new directory for a new experiment." >&2
  exit 1
fi

# Full model: original LAIN + Text/ObjCond Adapters + SceneGate.
WANDB_DIR="$WANDB_DIR" WANDB__SERVICE_WAIT=300 \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
torchrun --rdzv_id "$RDZV_ID" --rdzv_backend c10d --nproc_per_node 1 \
  --rdzv_endpoint "127.0.0.1:$PORT" main.py \
  --pretrained "$DETR_PRETRAINED" --clip_dir_vit "$CLIP_PRETRAINED" \
  --output-dir "$OUTPUT_DIR" --hico-image-root "$HICO_IMAGE_ROOT" \
  --dataset hicodet --zs --zs_type unseen_object --num_classes 117 \
  --epochs "${EPOCHS:-20}" --batch-size "${BATCH_SIZE:-8}" \
  --num-workers "${NUM_WORKERS:-8}" --prefetch-factor "${PREFETCH_FACTOR:-4}" \
  --seed "${SEED:-66}" --weight-decay 1e-4 --lr-drop 10 --clip-max-norm 0.1 \
  --lr-head 1e-3 --lr-vit 1e-3 \
  --alpha 0.5 --gamma 0.2 --hyper_lambda 2.8 \
  --box-score-thresh 0.2 --fg-iou-thresh 0.5 \
  --min-instances 3 --max-instances 15 \
  --use_hotoken --use_prompt --use_exp --CSC --N_CTX 36 \
  --use_insadapter --adapt_dim 32 --use_prior --adapter_alpha 1.0 \
  --adapter_num_layers 1 --adapter_pos last \
  --use_text_adapter --text_adapter_dim 64 --lora_rank 8 --lora_alpha 8 \
  --adapter_residual_scale 0.05 --adapter_dropout 0.1 \
  --use_obj_cond_adapter --obj_cond_rank 4 \
  --lr-text-adapter 5e-4 --lr-obj-cond-adapter 5e-4 \
  --use_scene_gate --scene_gate_type pair --scene_gate_version v5 \
  --scene_gate_hidden_dim 128 --scene_gate_rank 16 --scene_gate_dropout 0.1 \
  --scene_gate_alpha 0.02 --scene_gate_activation standardized_tanh \
  --scene_gate_init_std 1e-3 \
  --scene-gate-diagnostics --scene-gate-compare-off \
  --amp --print-interval "${PRINT_INTERVAL:-500}"
