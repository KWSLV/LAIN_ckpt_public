#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "g1" && "$1" != "g2" && "$1" != "g3" ) ]]; then
  echo "Usage: bash $0 {g1|g2|g3}" >&2
  exit 2
fi

VARIANT="$1"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export PATH="/root/miniconda3/bin:$PATH"
source scripts/server_assets.sh
prepare_lain_server_assets

SEED="${SEED:-66}"
EPOCHS=10
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-/root/autodl-tmp/Lain/UO_a4all_v5lowrank_seed66/checkpoints/ckpt_23370_06.pt}"

case "$VARIANT" in
  g1)
    GATE_ALPHA="0.02"
    RUN_NAME="UO_a4lr_ep6_gateonly_g1_a002_seed${SEED}"
    ;;
  g2)
    GATE_ALPHA="0.04"
    RUN_NAME="UO_a4lr_ep6_gateonly_g2_a004_seed${SEED}"
    ;;
  g3)
    GATE_ALPHA="0.08"
    RUN_NAME="UO_a4lr_ep6_gateonly_g3_a008_seed${SEED}"
    ;;
esac

RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/Lain/$RUN_NAME}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/checkpoints}"
WANDB_DIR="${WANDB_DIR:-$RUN_ROOT/wandb}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PORT="${PORT:-$((RANDOM % 5000 + 15000))}"
RDZV_ID="${RDZV_ID:-$((RANDOM % 5000 + 15000))}"

if [[ ! -f "$SOURCE_CHECKPOINT" ]]; then
  echo "[ERROR] Missing epoch-6 source checkpoint: $SOURCE_CHECKPOINT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$WANDB_DIR" "$RUN_ROOT/logs"

exec 9>"/tmp/lain_cuda_${CUDA_VISIBLE_DEVICES}.lock"
if ! flock -n 9; then
  echo "[ERROR] Another managed LAIN job is using CUDA device $CUDA_VISIBLE_DEVICES." >&2
  exit 1
fi

echo "[${VARIANT^^}] Source: $SOURCE_CHECKPOINT"
echo "[${VARIANT^^}] Text: OFF"
echo "[${VARIANT^^}] Object: ON and frozen"
echo "[${VARIANT^^}] Trainable module: fresh V5-lowrank Gate only"
echo "[${VARIANT^^}] Gate alpha=$GATE_ALPHA, rank=16, lr=3e-4"
echo "[${VARIANT^^}] Epochs: 10; checkpoint: every epoch"
echo "[${VARIANT^^}] W&B: per-epoch Gate/Object contribution curves"
echo "[${VARIANT^^}] Output: $OUTPUT_DIR"

WANDB_DIR="$WANDB_DIR" \
WANDB_RUN_GROUP="A4LR_ep6_gateonly" \
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
  --reset_scene_gate_on_load \
  --train_scene_gate_only \
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
  --lr-scene-gate 3e-4 \
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
  --scene_gate_version v5_lowrank \
  --scene_gate_rank 16 \
  --scene_gate_dropout 0.1 \
  --scene_gate_alpha "$GATE_ALPHA" \
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

echo "[${VARIANT^^}] Complete."
echo "[${VARIANT^^}] Checkpoints: $OUTPUT_DIR"
echo "[${VARIANT^^}] Diagnostics: $OUTPUT_DIR/scene_gate_diagnostics.json"
