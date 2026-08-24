#!/usr/bin/env bash
set -euo pipefail

# Post-hoc Text Adapter contribution evaluation for checkpoints from:
#   UO_two_stage_last_r4a16/stage1_text_obj
#
# Stage 1 has no SceneGate. The two sequential passes are:
#   full model and Text Adapter OFF. Object Adapter remains enabled.

if [[ $# -ne 1 ]]; then
  echo "Usage: bash $0 /absolute/path/to/stage1_text_obj_checkpoint.pt" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export PATH="/root/miniconda3/bin:$PATH"
source scripts/server_assets.sh
prepare_lain_server_assets

CHECKPOINT="$1"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[ERROR] Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/Lain/UO_two_stage_last_r4a16/stage1_module_contribution_eval}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PORT="${PORT:-$((RANDOM % 5000 + 15000))}"
RDZV_ID="${RDZV_ID:-$((RANDOM % 5000 + 15000))}"

mkdir -p "$OUTPUT_DIR"

exec 9>"/tmp/lain_cuda_${CUDA_VISIBLE_DEVICES}.lock"
if ! flock -n 9; then
  echo "[ERROR] Another managed LAIN job is using CUDA device $CUDA_VISIBLE_DEVICES." >&2
  exit 1
fi
GPU_PROCESSES="$(
  nvidia-smi \
    --query-compute-apps=pid,process_name \
    --format=csv,noheader,nounits 2>/dev/null || true
)"
if [[ -n "${GPU_PROCESSES//[[:space:]]/}" ]]; then
  echo "[ERROR] GPU is busy; contribution evaluation was not started:" >&2
  echo "$GPU_PROCESSES" >&2
  exit 1
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
  --zs \
  --zs_type unseen_object \
  --num_classes 117 \
  --test-batch-size "$TEST_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --prefetch-factor "$PREFETCH_FACTOR" \
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
  --adapter_pos last \
  --use_text_adapter \
  --text_adapter_dim 64 \
  --lora_rank 4 \
  --lora_alpha 16 \
  --adapter_residual_scale 0.05 \
  --adapter_dropout 0.1 \
  --use_obj_cond_adapter \
  --obj_cond_rank 4 \
  --adapter-contribution-diagnostics \
  --skip-object-adapter-contribution \
  --amp \
  --fast-cuda \
  --eval
