#!/usr/bin/env bash
set -euo pipefail

# Clean A1 ablation for the paper:
# Visual LAIN Adapter + Text Adapter only. Object Adapter and SceneGate are off.
# The reported checkpoint is predeclared as epoch 20; do not select an epoch
# from the per-epoch test curve.

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export PATH="/root/miniconda3/bin:$PATH"
source scripts/server_assets.sh
prepare_lain_server_assets

SEED="${SEED:-66}"
EPOCHS=20
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/Lain/UO_a1_text_only_last_r4a16_seed${SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/checkpoints}"
WANDB_DIR="${WANDB_DIR:-$RUN_ROOT/wandb}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PRINT_INTERVAL="${PRINT_INTERVAL:-500}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$OUTPUT_DIR" "$WANDB_DIR" "$RUN_ROOT/logs"

# Refuse to share the same GPU with another managed LAIN job.
exec 9>"/tmp/lain_cuda_${CUDA_VISIBLE_DEVICES}.lock"
if ! flock -n 9; then
  echo "[ERROR] Another managed LAIN job is using CUDA device $CUDA_VISIBLE_DEVICES." >&2
  exit 1
fi

available_kb="$(df -Pk "$RUN_ROOT" | awk 'NR==2 {print $4}')"
minimum_kb=$((4 * 1024 * 1024))
if (( available_kb < minimum_kb )); then
  echo "[ERROR] Less than 4 GiB is available under $RUN_ROOT." >&2
  echo "[ERROR] Archive old checkpoints before starting A1." >&2
  exit 1
fi

find_epoch_checkpoint() {
  local epoch="$1"
  local epoch_padded
  local matches=()
  epoch_padded="$(printf '%02d' "$epoch")"
  shopt -s nullglob
  matches=("$OUTPUT_DIR"/ckpt_*_"$epoch_padded".pt)
  shopt -u nullglob
  if (( ${#matches[@]} == 0 )); then
    return 1
  fi
  printf '%s\n' "${matches[-1]}"
}

find_latest_checkpoint() {
  local matches=()
  shopt -s nullglob
  matches=("$OUTPUT_DIR"/ckpt_*.pt)
  shopt -u nullglob
  if (( ${#matches[@]} == 0 )); then
    return 1
  fi
  printf '%s\n' "${matches[-1]}"
}

validate_existing_args() {
  local args_file="$OUTPUT_DIR/args.txt"
  [[ ! -f "$args_file" ]] && return 0

  /root/miniconda3/bin/python - "$args_file" "$SEED" <<'PY'
import json
import sys

path, seed = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    args = json.load(handle)

expected = {
    "dataset": "hicodet",
    "zs": True,
    "zs_type": "unseen_object",
    "epochs": 20,
    "seed": int(seed),
    "adapter_pos": "last",
    "use_text_adapter": True,
    "text_adapter_dim": 64,
    "lora_rank": 4,
    "lora_alpha": 16,
    "use_obj_cond_adapter": False,
    "use_scene_gate": False,
}
mismatches = {
    key: (args.get(key), value)
    for key, value in expected.items()
    if args.get(key) != value
}
if mismatches:
    details = ", ".join(
        f"{key}: found={found!r}, expected={wanted!r}"
        for key, (found, wanted) in mismatches.items()
    )
    raise SystemExit(f"[ERROR] Existing A1 directory has incompatible args: {details}")
PY
}

validate_existing_args

final_checkpoint="$(find_epoch_checkpoint "$EPOCHS" || true)"
if [[ -n "$final_checkpoint" ]]; then
  echo "[A1] Epoch-$EPOCHS checkpoint already exists: $final_checkpoint"
  exit 0
fi

resume_checkpoint="$(find_latest_checkpoint || true)"
load_args=()
if [[ -n "$resume_checkpoint" ]]; then
  load_args=(--resume "$resume_checkpoint")
  echo "[A1] Resuming interrupted run from: $resume_checkpoint"
else
  echo "[A1] Starting a clean Visual + Text run from DETR and CLIP pretrained weights."
fi

PORT="${PORT:-$((RANDOM % 5000 + 15000))}"
RDZV_ID="${RDZV_ID:-$((RANDOM % 5000 + 15000))}"

echo "[A1] Modules: Visual Adapter ON | Text Adapter ON | Object Adapter OFF | SceneGate OFF"
echo "[A1] Text Adapter: dim=64, rank=4, alpha=16, residual=0.05, dropout=0.1"
echo "[A1] Protocol: UO, seed=$SEED, epochs=$EPOCHS, fixed final checkpoint=epoch 20"
echo "[A1] Output: $OUTPUT_DIR"

WANDB_DIR="$WANDB_DIR" WANDB__SERVICE_WAIT=300 \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
torchrun \
  --rdzv_id "$RDZV_ID" \
  --rdzv_backend c10d \
  --nproc_per_node 1 \
  --rdzv_endpoint "127.0.0.1:$PORT" \
  main.py \
  --pretrained "$DETR_PRETRAINED" \
  --clip_dir_vit "$CLIP_PRETRAINED" \
  "${load_args[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --hico-image-root "$HICO_IMAGE_ROOT" \
  --dataset hicodet \
  --partitions train2015 test2015 \
  --zs --zs_type unseen_object \
  --num_classes 117 \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --test-batch-size "$TEST_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --prefetch-factor "$PREFETCH_FACTOR" \
  --seed "$SEED" \
  --weight-decay 1e-4 \
  --lr-drop 10 \
  --clip-max-norm 0.1 \
  --lr-head 1e-3 \
  --lr-vit 1e-3 \
  --lr-text-adapter 5e-4 \
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
  --amp \
  --fast-cuda \
  --keep-last-checkpoints 1 \
  --print-interval "$PRINT_INTERVAL"

final_checkpoint="$(find_epoch_checkpoint "$EPOCHS" || true)"
if [[ -z "$final_checkpoint" ]]; then
  echo "[ERROR] A1 exited without an epoch-$EPOCHS checkpoint." >&2
  exit 1
fi

echo "[A1] Complete: $final_checkpoint"
