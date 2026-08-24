#!/usr/bin/env bash
set -euo pipefail

# Shared implementation for the last/all two-stage experiments.
# Call this file through UO_two_stage_last_r4a16.sh or
# UO_two_stage_all_r4a16.sh so the adapter position is explicit.

if [[ $# -ne 1 || ( "$1" != "last" && "$1" != "all" ) ]]; then
  echo "Usage: bash $0 {last|all}" >&2
  exit 2
fi

ADAPTER_POS="$1"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export PATH="/root/miniconda3/bin:$PATH"
source scripts/server_assets.sh
prepare_lain_server_assets

# The two experiments differ only in this visual Adapter position.
# All other architecture, optimizer and data parameters are shared below.
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/Lain/UO_two_stage_${ADAPTER_POS}_r4a16}"
STAGE1_ROOT="$RUN_ROOT/stage1_text_obj"
STAGE2_ROOT="$RUN_ROOT/stage2_scene_gate"
STAGE1_OUTPUT="$STAGE1_ROOT/checkpoints"
STAGE2_OUTPUT="$STAGE2_ROOT/checkpoints"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-20}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-2}"
KEEP_STAGE1_CHECKPOINTS="${KEEP_STAGE1_CHECKPOINTS:-2}"
KEEP_STAGE2_CHECKPOINTS="${KEEP_STAGE2_CHECKPOINTS:-2}"

BATCH_SIZE="${BATCH_SIZE:-8}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PRINT_INTERVAL="${PRINT_INTERVAL:-500}"
SEED="${SEED:-66}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p \
  "$STAGE1_OUTPUT" "$STAGE1_ROOT/wandb" \
  "$STAGE2_OUTPUT" "$STAGE2_ROOT/wandb" \
  "$RUN_ROOT/logs"

LOG_FILE="${LOG_FILE:-$RUN_ROOT/logs/two_stage_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[Two-stage] Visual Adapter position: $ADAPTER_POS"
echo "[Two-stage] Stage 1: LAIN + Text Adapter r4/a16 + ObjCond r4, no SceneGate"
echo "[Two-stage] Stage 2: reset and train only v5 standardized-tanh SceneGate"
echo "[Two-stage] Results: $RUN_ROOT"
echo "[Two-stage] Log: $LOG_FILE"

if (( KEEP_STAGE1_CHECKPOINTS < 1 || KEEP_STAGE2_CHECKPOINTS < 1 )); then
  echo "[ERROR] Both checkpoint-retention counts must be at least 1." >&2
  exit 1
fi

# A single GPU cannot safely run two LAIN training jobs at once. The lock fails
# immediately instead of silently sharing the GPU and corrupting timing/results.
exec 9>"/tmp/lain_cuda_${CUDA_VISIBLE_DEVICES}.lock"
if ! flock -n 9; then
  echo "[ERROR] Another managed LAIN job is already using CUDA device $CUDA_VISIBLE_DEVICES." >&2
  exit 1
fi

available_kb="$(df -Pk "$RUN_ROOT" | awk 'NR==2 {print $4}')"
minimum_kb=$((4 * 1024 * 1024))
if (( available_kb < minimum_kb )); then
  echo "[ERROR] Less than 4 GiB is available under $RUN_ROOT." >&2
  echo "[ERROR] Free disk space before starting this two-stage run." >&2
  exit 1
fi

find_epoch_checkpoint() {
  local directory="$1"
  local epoch="$2"
  local epoch_padded
  local matches=()
  epoch_padded="$(printf '%02d' "$epoch")"
  shopt -s nullglob
  matches=("$directory"/ckpt_*_"$epoch_padded".pt)
  shopt -u nullglob
  if (( ${#matches[@]} == 0 )); then
    return 1
  fi
  printf '%s\n' "${matches[-1]}"
}

find_latest_checkpoint() {
  local directory="$1"
  local matches=()
  shopt -s nullglob
  matches=("$directory"/ckpt_*.pt)
  shopt -u nullglob
  if (( ${#matches[@]} == 0 )); then
    return 1
  fi
  printf '%s\n' "${matches[-1]}"
}

validate_existing_args() {
  local args_file="$1"
  local expected_stage="$2"
  [[ ! -f "$args_file" ]] && return 0

  /root/miniconda3/bin/python - "$args_file" "$ADAPTER_POS" "$expected_stage" <<'PY'
import json
import sys

path, adapter_pos, stage = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    args = json.load(handle)

expected = {
    "adapter_pos": adapter_pos,
    "use_text_adapter": True,
    "lora_rank": 4,
    "lora_alpha": 16,
    "use_obj_cond_adapter": True,
    "obj_cond_rank": 4,
}
if stage == "stage1":
    expected["use_scene_gate"] = False
else:
    expected.update({
        "use_scene_gate": True,
        "scene_gate_version": "v5",
        "scene_gate_activation": "standardized_tanh",
        "train_scene_gate_only": True,
    })

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
    raise SystemExit(f"[ERROR] Existing {stage} directory has incompatible args: {details}")
PY
}

validate_existing_args "$STAGE1_OUTPUT/args.txt" stage1
validate_existing_args "$STAGE2_OUTPUT/args.txt" stage2

# Parameters shared by both stages. Text r4/a16 keeps alpha/rank=4, matching
# the original r2/a8 residual strength while giving the requested rank 4.
MODEL_ARGS=(
  --dataset hicodet
  --zs
  --zs_type unseen_object
  --num_classes 117
  --batch-size "$BATCH_SIZE"
  --test-batch-size "$TEST_BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --prefetch-factor "$PREFETCH_FACTOR"
  --seed "$SEED"
  --weight-decay 1e-4
  --lr-drop 10
  --clip-max-norm 0.1
  --alpha 0.5
  --gamma 0.2
  --hyper_lambda 2.8
  --box-score-thresh 0.2
  --fg-iou-thresh 0.5
  --min-instances 3
  --max-instances 15
  --use_hotoken
  --use_prompt
  --use_exp
  --CSC
  --N_CTX 36
  --use_insadapter
  --adapt_dim 32
  --use_prior
  --adapter_alpha 1.0
  --adapter_num_layers 1
  --adapter_pos "$ADAPTER_POS"
  --use_text_adapter
  --text_adapter_dim 64
  --lora_rank 4
  --lora_alpha 16
  --adapter_residual_scale 0.05
  --adapter_dropout 0.1
  --use_obj_cond_adapter
  --obj_cond_rank 4
  --amp
  --fast-cuda
  --print-interval "$PRINT_INTERVAL"
)

run_torch() {
  local port
  local rdzv_id
  port="${PORT:-$((RANDOM % 5000 + 15000))}"
  rdzv_id="${RDZV_ID:-$((RANDOM % 5000 + 15000))}"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  torchrun \
    --rdzv_id "$rdzv_id" \
    --rdzv_backend c10d \
    --nproc_per_node 1 \
    --rdzv_endpoint "127.0.0.1:$port" \
    main.py "$@"
}

stage1_final="$(find_epoch_checkpoint "$STAGE1_OUTPUT" "$STAGE1_EPOCHS" || true)"
if [[ -z "$stage1_final" ]]; then
  stage1_resume="$(find_latest_checkpoint "$STAGE1_OUTPUT" || true)"
  stage1_load_args=()
  if [[ -n "$stage1_resume" ]]; then
    stage1_load_args=(--resume "$stage1_resume")
    echo "[Stage 1] Resuming interrupted training from: $stage1_resume"
  else
    echo "[Stage 1] Starting from DETR + CLIP pretrained weights."
  fi

  # SceneGate is deliberately absent in stage 1. This prevents it from
  # perturbing the Text/Object Adapter base and avoids an extra OFF evaluation.
  WANDB_DIR="$STAGE1_ROOT/wandb" WANDB__SERVICE_WAIT=300 \
  run_torch \
    --pretrained "$DETR_PRETRAINED" \
    --clip_dir_vit "$CLIP_PRETRAINED" \
    "${stage1_load_args[@]}" \
    --output-dir "$STAGE1_OUTPUT" \
    --hico-image-root "$HICO_IMAGE_ROOT" \
    --epochs "$STAGE1_EPOCHS" \
    --keep-last-checkpoints "$KEEP_STAGE1_CHECKPOINTS" \
    --lr-head 1e-3 \
    --lr-vit 1e-3 \
    --lr-text-adapter 5e-4 \
    --lr-obj-cond-adapter 5e-4 \
    "${MODEL_ARGS[@]}"

  stage1_final="$(find_epoch_checkpoint "$STAGE1_OUTPUT" "$STAGE1_EPOCHS" || true)"
fi

if [[ -z "$stage1_final" ]]; then
  echo "[ERROR] Stage 1 finished without an epoch-$STAGE1_EPOCHS checkpoint." >&2
  exit 1
fi
echo "[Stage 1] Complete: $stage1_final"

stage2_final="$(find_epoch_checkpoint "$STAGE2_OUTPUT" "$STAGE2_EPOCHS" || true)"
if [[ -z "$stage2_final" ]]; then
  stage2_resume="$(find_latest_checkpoint "$STAGE2_OUTPUT" || true)"
  if [[ -n "$stage2_resume" ]]; then
    stage2_load_args=(--resume "$stage2_resume")
    echo "[Stage 2] Resuming interrupted gate-only training from: $stage2_resume"
  else
    stage2_load_args=(
      --init_from "$stage1_final"
      --reset_scene_gate_on_load
    )
    echo "[Stage 2] Loading stage-1 weights and freshly initializing SceneGate."
  fi

  # Stable gate:
  # raw -> per-image pair centering -> std normalization -> tanh -> recenter.
  # Only SceneGate is trainable. Every epoch evaluates both Gate ON and OFF.
  WANDB_DIR="$STAGE2_ROOT/wandb" WANDB__SERVICE_WAIT=300 \
  run_torch \
    --pretrained "$DETR_PRETRAINED" \
    --clip_dir_vit "$CLIP_PRETRAINED" \
    "${stage2_load_args[@]}" \
    --output-dir "$STAGE2_OUTPUT" \
    --hico-image-root "$HICO_IMAGE_ROOT" \
    --epochs "$STAGE2_EPOCHS" \
    --keep-last-checkpoints "$KEEP_STAGE2_CHECKPOINTS" \
    --lr-head 1e-3 \
    --lr-vit 1e-3 \
    --lr-text-adapter 5e-4 \
    --lr-obj-cond-adapter 5e-4 \
    --lr-scene-gate 1e-3 \
    "${MODEL_ARGS[@]}" \
    --use_scene_gate \
    --scene_gate_type pair \
    --scene_gate_version v5 \
    --scene_gate_hidden_dim 128 \
    --scene_gate_rank 16 \
    --scene_gate_dropout 0.1 \
    --scene_gate_alpha 0.02 \
    --scene_gate_activation standardized_tanh \
    --scene_gate_init_std 1e-3 \
    --train_scene_gate_only \
    --scene-gate-diagnostics \
    --scene-gate-compare-off \
    --adapter-contribution-diagnostics \
    --skip-object-adapter-contribution

  stage2_final="$(find_epoch_checkpoint "$STAGE2_OUTPUT" "$STAGE2_EPOCHS" || true)"
fi

if [[ -z "$stage2_final" ]]; then
  echo "[ERROR] Stage 2 finished without an epoch-$STAGE2_EPOCHS checkpoint." >&2
  exit 1
fi

echo "[Two-stage] Complete."
echo "[Two-stage] Base checkpoint: $stage1_final"
echo "[Two-stage] Final gate checkpoint: $stage2_final"
echo "[Two-stage] ON/OFF JSON: $STAGE2_OUTPUT/scene_gate_diagnostics.json"
echo "[Two-stage] ON/OFF CSV: $STAGE2_OUTPUT/scene_gate_diagnostics.csv"
