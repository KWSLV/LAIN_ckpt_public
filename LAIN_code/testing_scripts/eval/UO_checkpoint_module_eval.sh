#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: bash $0 {object_only|object_gate} /absolute/path/to/checkpoint.pt [tag]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export PATH="/root/miniconda3/bin:$PATH"
source scripts/server_assets.sh
prepare_lain_server_assets

MODE="$1"
CHECKPOINT="$2"
TAG="${3:-$(basename "$CHECKPOINT" .pt)}"
EVAL_ROOT="${EVAL_ROOT:-/root/autodl-tmp/Lain/UO_checkpoint_module_eval}"
OUTPUT_DIR="${OUTPUT_DIR:-$EVAL_ROOT/$TAG}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PORT="${PORT:-$((RANDOM % 5000 + 15000))}"
RDZV_ID="${RDZV_ID:-$((RANDOM % 5000 + 15000))}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[ERROR] Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

case "$MODE" in
  object_only)
    MODULE_ARGS=(
      --use_obj_cond_adapter
      --obj_cond_rank 4
      --adapter-contribution-diagnostics
    )
    ;;
  object_gate)
    MODULE_ARGS=(
      --use_obj_cond_adapter
      --obj_cond_rank 4
      --use_scene_gate
      --scene_gate_type pair
      --scene_gate_version v5
      --scene_gate_hidden_dim 128
      --scene_gate_rank 16
      --scene_gate_dropout 0.1
      --scene_gate_alpha 0.04
      --scene_gate_activation standardized_tanh
      --scene_gate_init_std 1e-3
      --scene_gate_start_epoch 1
      --scene-gate-post-l2-norm
      --scene-gate-diagnostics
      --scene-gate-compare-off
      --adapter-contribution-diagnostics
    )
    ;;
  *)
    echo "[ERROR] Unsupported mode: $MODE" >&2
    exit 2
    ;;
esac

mkdir -p "$OUTPUT_DIR"

exec 9>"/tmp/lain_cuda_${CUDA_VISIBLE_DEVICES}.lock"
if ! flock -n 9; then
  echo "[ERROR] Another managed LAIN job is using CUDA device $CUDA_VISIBLE_DEVICES." >&2
  exit 1
fi

echo "[UO Eval] mode=$MODE"
echo "[UO Eval] checkpoint=$CHECKPOINT"
echo "[UO Eval] output=$OUTPUT_DIR"
echo "[UO Eval] adapter_pos=all, Text=OFF, Object=ON"
if [[ "$MODE" == "object_gate" ]]; then
  echo "[UO Eval] Gate=ON, V5 H128 alpha=0.04 standardized_tanh post_l2=ON"
else
  echo "[UO Eval] Gate=OFF"
fi

WANDB_MODE=disabled CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
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
  --partitions train2015 test2015 \
  --zs --zs_type unseen_object \
  --num_classes 117 \
  --test-batch-size 8 \
  --num-workers 8 \
  --prefetch-factor 4 \
  --seed 66 \
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
  "${MODULE_ARGS[@]}" \
  --amp \
  --fast-cuda \
  --eval

echo "[UO Eval] result=$OUTPUT_DIR/scene_gate_diagnostics.json"
