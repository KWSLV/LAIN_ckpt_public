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
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-/root/autodl-tmp/Lain/UO_objectbest_gateonly_v5h128_a004_refine_from_ep01_lr5e5_seed66/checkpoints/ckpt_15580_04.pt}"
SOURCE_UNSEEN="${SOURCE_UNSEEN:-37.775465258102386}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/Lain/${SCRIPT_NAME}_seed${SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/checkpoints}"
WANDB_DIR="${WANDB_DIR:-$RUN_ROOT/wandb}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PORT="${PORT:-$((RANDOM % 5000 + 15000))}"
RDZV_ID="${RDZV_ID:-$((RANDOM % 5000 + 15000))}"

OBJECT_LR="${OBJECT_LR:-5e-6}"
GATE_LR="${GATE_LR:-1e-5}"
SIGNIFICANT_DELTA="${SIGNIFICANT_DELTA:-0.03}"
SMALL_LR_FACTOR="${SMALL_LR_FACTOR:-0.5}"
BAD_LR_FACTOR="${BAD_LR_FACTOR:-0.2}"
NO_BEST_PATIENCE="${NO_BEST_PATIENCE:-2}"

if [[ ! -f "$SOURCE_CHECKPOINT" ]]; then
  echo "[ERROR] Missing Gate epoch-4 checkpoint: $SOURCE_CHECKPOINT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$WANDB_DIR" "$RUN_ROOT/logs"

exec 9>"/tmp/lain_cuda_${CUDA_VISIBLE_DEVICES}.lock"
if ! flock -n 9; then
  echo "[ERROR] Another managed LAIN job is using CUDA device $CUDA_VISIBLE_DEVICES." >&2
  exit 1
fi

echo "[$SCRIPT_NAME] Source: $SOURCE_CHECKPOINT"
echo "[$SCRIPT_NAME] Protected initial best Unseen: $SOURCE_UNSEEN"
echo "[$SCRIPT_NAME] Trainable: ObjectConditionedAdapter + SceneGate only"
echo "[$SCRIPT_NAME] Frozen: DETR, original LAIN, CLIP/prompt, Text Adapter"
echo "[$SCRIPT_NAME] Initial LR: Object=$OBJECT_LR Gate=$GATE_LR"
echo "[$SCRIPT_NAME] Policy: delta >= $SIGNIFICANT_DELTA keep LR; small gain/miss x$SMALL_LR_FACTOR"
echo "[$SCRIPT_NAME] Policy: decline > $SIGNIFICANT_DELTA rollback best, reset optimizer, LR x$BAD_LR_FACTOR"
echo "[$SCRIPT_NAME] Early stop: $NO_BEST_PATIENCE consecutive epochs without new best"
echo "[$SCRIPT_NAME] W&B run name: $SCRIPT_NAME"
echo "[$SCRIPT_NAME] Output: $OUTPUT_DIR"

WANDB_DIR="$WANDB_DIR" \
WANDB_NAME="$SCRIPT_NAME" \
WANDB_RUN_GROUP="ObjectGate_joint_bestrollback" \
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
  --train-object-and-scene-gate-only \
  --joint-best-rollback-policy \
  --joint-rollback-initial-best-unseen "$SOURCE_UNSEEN" \
  --joint-rollback-significant-delta "$SIGNIFICANT_DELTA" \
  --joint-rollback-small-lr-factor "$SMALL_LR_FACTOR" \
  --joint-rollback-bad-lr-factor "$BAD_LR_FACTOR" \
  --joint-rollback-patience "$NO_BEST_PATIENCE" \
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
  --lr-drop 100 \
  --clip-max-norm 0.1 \
  --lr-head 5e-4 \
  --lr-vit 2e-4 \
  --lr-obj-cond-adapter "$OBJECT_LR" \
  --lr-scene-gate "$GATE_LR" \
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
  --use_scene_gate \
  --scene_gate_type pair \
  --scene_gate_version v5 \
  --scene_gate_hidden_dim 128 \
  --scene_gate_dropout 0.1 \
  --scene_gate_alpha 0.04 \
  --scene_gate_activation standardized_tanh \
  --scene_gate_init_std 1e-3 \
  --scene_gate_start_epoch 1 \
  --scene-gate-post-l2-norm \
  --scene-gate-diagnostics \
  --scene-gate-compare-off \
  --adapter-contribution-diagnostics \
  --amp \
  --fast-cuda \
  --keep-last-checkpoints 0 \
  --keep-best-unseen-checkpoint \
  --print-interval 500

echo "[$SCRIPT_NAME] Complete."
echo "[$SCRIPT_NAME] Checkpoints: $OUTPUT_DIR"
echo "[$SCRIPT_NAME] Diagnostics: $OUTPUT_DIR/scene_gate_diagnostics.json"

