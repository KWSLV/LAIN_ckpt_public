#!/usr/bin/env bash
set -euo pipefail

# Evaluate the fixed epoch-20 A1 checkpoint twice:
#   1. Text Adapter ON (the trained model)
#   2. Text Adapter OFF (same checkpoint, all other modules unchanged)
# The evaluator writes the existing scene_gate_diagnostics.{json,csv} schema,
# whose adapter-contribution columns contain the Text ON-minus-OFF result.

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export PATH="/root/miniconda3/bin:$PATH"
source scripts/server_assets.sh
prepare_lain_server_assets

SEED="${SEED:-66}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/Lain/UO_a1_text_only_last_r4a16_seed${SEED}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$RUN_ROOT/checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/text_contribution_eval}"
LOGICAL_EPOCH=20
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$OUTPUT_DIR"

# Queue behind the active training job instead of competing for GPU memory.
exec 9>"/tmp/lain_cuda_${CUDA_VISIBLE_DEVICES}.lock"
if ! flock -w "${LOCK_TIMEOUT_SECONDS:-21600}" 9; then
  echo "[ERROR] Timed out waiting for CUDA device $CUDA_VISIBLE_DEVICES." >&2
  exit 1
fi

epoch_padded="$(printf '%02d' "$LOGICAL_EPOCH")"
shopt -s nullglob
matches=("$CHECKPOINT_DIR"/ckpt_*_"$epoch_padded".pt)
shopt -u nullglob
if (( ${#matches[@]} == 0 )); then
  echo "[ERROR] Epoch-$LOGICAL_EPOCH checkpoint not found in $CHECKPOINT_DIR." >&2
  exit 1
fi
CHECKPOINT="${matches[-1]}"

GPU_PROCESSES="$(
  nvidia-smi \
    --query-compute-apps=pid,process_name \
    --format=csv,noheader,nounits 2>/dev/null || true
)"
if [[ -n "${GPU_PROCESSES//[[:space:]]/}" ]]; then
  echo "[ERROR] CUDA lock was acquired but GPU processes are still present:" >&2
  echo "$GPU_PROCESSES" >&2
  exit 1
fi

PORT="${PORT:-$((RANDOM % 5000 + 15000))}"
RDZV_ID="${RDZV_ID:-$((RANDOM % 5000 + 15000))}"

echo "[A1 Text Contribution] checkpoint: $CHECKPOINT"
echo "[A1 Text Contribution] passes: Text ON, then Text OFF"
echo "[A1 Text Contribution] output: $OUTPUT_DIR"

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
  --test-batch-size "$TEST_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --prefetch-factor "$PREFETCH_FACTOR" \
  --seed "$SEED" \
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
  --adapter-contribution-diagnostics \
  --skip-object-adapter-contribution \
  --amp \
  --fast-cuda \
  --eval

for name in scene_gate_diagnostics.json scene_gate_diagnostics.csv; do
  if [[ ! -f "$OUTPUT_DIR/$name" ]]; then
    echo "[ERROR] Expected diagnostic file was not generated: $OUTPUT_DIR/$name" >&2
    exit 1
  fi
  cp -f -- "$OUTPUT_DIR/$name" "$CHECKPOINT_DIR/$name"
done

echo "[A1 Text Contribution] diagnostics copied to $CHECKPOINT_DIR"
