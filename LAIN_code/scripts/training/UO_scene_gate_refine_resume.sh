#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/training/UO_scene_gate_refine_resume.sh /absolute/path/to/gate_refine_checkpoint.pt"
  exit 2
fi

cd "$(dirname "$0")/../.."
source scripts/server_assets.sh
prepare_lain_server_assets

CHECKPOINT="$1"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[ERROR] Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

PORT="${PORT:-$((RANDOM % 5000 + 5000))}"
RDZV_ID="${RDZV_ID:-$((RANDOM % 5000 + 5000))}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/Lain/UO_scene_gate_refine_resume}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/checkpoints}"
WANDB_DIR="${WANDB_DIR:-$RUN_ROOT/wandb}"
mkdir -p "$OUTPUT_DIR" "$WANDB_DIR"

# A resumed optimizer keeps its checkpoint LR unless GATE_LR is explicitly
# supplied. When supplied, update the actual SceneGate parameter group.
GATE_LR_ARGS=()
if [[ -n "${GATE_LR:-}" ]]; then
  GATE_LR_ARGS=(--lr-scene-gate "$GATE_LR" --override-resume-scene-gate-lr)
fi

# Continue a SceneGate-only refinement checkpoint without resetting the gate,
# optimizer, scheduler, epoch, iteration, or AMP scaler.
WANDB_DIR="$WANDB_DIR" WANDB__SERVICE_WAIT=300 \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
torchrun --rdzv_id "$RDZV_ID" --rdzv_backend c10d --nproc_per_node 1 \
  --rdzv_endpoint "127.0.0.1:$PORT" main.py \
  --pretrained "$DETR_PRETRAINED" --clip_dir_vit "$CLIP_PRETRAINED" \
  --resume "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" --hico-image-root "$HICO_IMAGE_ROOT" \
  --dataset hicodet --zs --zs_type unseen_object --num_classes 117 \
  --epochs "${EXTRA_EPOCHS:-3}" --batch-size "${BATCH_SIZE:-8}" \
  --num-workers "${NUM_WORKERS:-8}" --prefetch-factor "${PREFETCH_FACTOR:-4}" \
  --seed "${SEED:-66}" --weight-decay 1e-4 --lr-drop 10 --clip-max-norm 0.1 \
  --lr-head 1e-3 "${GATE_LR_ARGS[@]}" \
  --alpha 0.5 --gamma 0.2 --hyper_lambda 2.8 \
  --box-score-thresh 0.2 --fg-iou-thresh 0.5 \
  --min-instances 3 --max-instances 15 \
  --use_hotoken --use_prompt --use_exp --CSC --N_CTX 36 \
  --use_insadapter --adapt_dim 32 --use_prior --adapter_alpha 1.0 \
  --adapter_num_layers 1 --adapter_pos last \
  --use_text_adapter --text_adapter_dim 64 --lora_rank 8 --lora_alpha 8 \
  --adapter_residual_scale 0.05 --adapter_dropout 0.1 \
  --use_obj_cond_adapter --obj_cond_rank 4 \
  --use_scene_gate --scene_gate_type pair --scene_gate_version v5 \
  --scene_gate_hidden_dim 128 --scene_gate_rank 16 --scene_gate_dropout 0.1 \
  --scene_gate_alpha "${SCENE_GATE_ALPHA:-0.02}" \
  --scene_gate_activation standardized_tanh --scene_gate_init_std 1e-3 \
  --scene-gate-post-l2-norm \
  --train_scene_gate_only --scene-gate-diagnostics --scene-gate-compare-off \
  --amp --print-interval "${PRINT_INTERVAL:-500}"
