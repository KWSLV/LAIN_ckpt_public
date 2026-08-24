#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "a2" && "$1" != "a3" ) ]]; then
  echo "Usage: bash $0 {a2|a3}" >&2
  exit 2
fi

VARIANT="$1"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export PATH="/root/miniconda3/bin:$PATH"
source scripts/server_assets.sh
prepare_lain_server_assets

SEED="${SEED:-66}"
if [[ "$VARIANT" == "a2" ]]; then
  EPOCHS=20
  LR_DROP=10
  LR_HEAD=1e-3
  LR_VIT=1e-3
  LR_TEXT_ADAPTER=5e-4
  LR_OBJ_ADAPTER=5e-4
  RUN_NAME="UO_a2_object_only_last_r4_seed${SEED}"
  MODULE_SUMMARY="Visual ON | Text OFF | Object ON | SceneGate OFF"
  VARIANT_ARGS=(
    --use_obj_cond_adapter
    --obj_cond_rank 4
  )
else
  # Tuned after the A1/A2 pilots: slow the shared Visual/Prompt/Head drift and
  # regularize Text/Object adaptation so unseen performance does not peak early
  # and then collapse toward the seen training distribution.
  # All A-series ablations use the same 20-epoch budget. Existing A3 runs
  # stopped at epoch 15 are accepted below and resumed to epoch 20.
  EPOCHS=20
  LR_DROP=10
  LR_HEAD=5e-4
  LR_VIT=2e-4
  LR_TEXT_ADAPTER=2e-4
  LR_OBJ_ADAPTER=2e-4
  RUN_NAME="UO_a3_tuned_text_object_last_r4a16_seed${SEED}"
  MODULE_SUMMARY="Visual ON | Text ON | Object ON | SceneGate OFF"
  VARIANT_ARGS=(
    --use_text_adapter
    --text_adapter_dim 64
    --lora_rank 4
    --lora_alpha 16
    --adapter_residual_scale 0.05
    --adapter_dropout 0.1
    --use_obj_cond_adapter
    --obj_cond_rank 4
  )
fi

RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/Lain/$RUN_NAME}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/checkpoints}"
WANDB_DIR="${WANDB_DIR:-$RUN_ROOT/wandb}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PRINT_INTERVAL="${PRINT_INTERVAL:-500}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$OUTPUT_DIR" "$WANDB_DIR" "$RUN_ROOT/logs"

exec 9>"/tmp/lain_cuda_${CUDA_VISIBLE_DEVICES}.lock"
if ! flock -n 9; then
  echo "[ERROR] Another managed LAIN job is using CUDA device $CUDA_VISIBLE_DEVICES." >&2
  exit 1
fi

available_kb="$(df -Pk "$RUN_ROOT" | awk 'NR==2 {print $4}')"
minimum_kb=$((4 * 1024 * 1024))
if (( available_kb < minimum_kb )); then
  echo "[ERROR] Less than 4 GiB is available under $RUN_ROOT." >&2
  echo "[ERROR] Archive old checkpoints before starting $VARIANT." >&2
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

  /root/miniconda3/bin/python - "$args_file" "$SEED" "$VARIANT" <<'PY'
import json
import sys

path, seed, variant = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    args = json.load(handle)

expected = {
    "dataset": "hicodet",
    "zs": True,
    "zs_type": "unseen_object",
    "epochs": 20,
    "seed": int(seed),
    "adapter_pos": "last",
    "use_text_adapter": variant == "a3",
    "use_obj_cond_adapter": True,
    "obj_cond_rank": 4,
    "use_scene_gate": False,
    "adapter_contribution_diagnostics": True,
    "keep_last_checkpoints": 1,
    "keep_best_unseen_checkpoint": True,
    "lr_drop": 10,
    "lr_head": 1e-3 if variant == "a2" else 5e-4,
    "lr_vit": 1e-3 if variant == "a2" else 2e-4,
    "lr_text_adapter": 5e-4 if variant == "a2" else 2e-4,
    "lr_obj_cond_adapter": 5e-4 if variant == "a2" else 2e-4,
}
if variant == "a3":
    expected.update({
        "text_adapter_dim": 64,
        "lora_rank": 4,
        "lora_alpha": 16,
    })

mismatches = {
    key: (args.get(key), value)
    for key, value in expected.items()
    if args.get(key) != value
}
# The first A3 launch used a 15-epoch budget. It is the only compatible
# exception: the continuation run restores ckpt_*_15.pt and rewrites args.txt
# with epochs=20 before running epochs 16-20.
if variant == "a3" and mismatches.get("epochs") == (15, 20):
    mismatches.pop("epochs")
if mismatches:
    details = ", ".join(
        f"{key}: found={found!r}, expected={wanted!r}"
        for key, (found, wanted) in mismatches.items()
    )
    raise SystemExit(f"[ERROR] Existing {variant.upper()} directory has incompatible args: {details}")
PY
}

validate_existing_args

final_checkpoint="$(find_epoch_checkpoint "$EPOCHS" || true)"
if [[ -n "$final_checkpoint" ]]; then
  if [[ ! -f "$OUTPUT_DIR/best_unseen.pt" ]]; then
    echo "[ERROR] Final checkpoint exists but best_unseen.pt is missing." >&2
    exit 1
  fi
  echo "[${VARIANT^^}] Complete: $final_checkpoint"
  echo "[${VARIANT^^}] Best unseen: $OUTPUT_DIR/best_unseen.pt"
  exit 0
fi

resume_checkpoint="$(find_latest_checkpoint || true)"
load_args=()
if [[ -n "$resume_checkpoint" ]]; then
  load_args=(--resume "$resume_checkpoint")
  echo "[${VARIANT^^}] Resuming from: $resume_checkpoint"
else
  echo "[${VARIANT^^}] Starting from DETR and CLIP pretrained weights."
fi

PORT="${PORT:-$((RANDOM % 5000 + 15000))}"
RDZV_ID="${RDZV_ID:-$((RANDOM % 5000 + 15000))}"

echo "[${VARIANT^^}] Modules: $MODULE_SUMMARY"
echo "[${VARIANT^^}] UO, seed=$SEED, epochs=$EPOCHS, adapter_pos=last"
echo "[${VARIANT^^}] LR: head=$LR_HEAD, vit=$LR_VIT, text=$LR_TEXT_ADAPTER, object=$LR_OBJ_ADAPTER, drop=$LR_DROP"
echo "[${VARIANT^^}] Adapter contribution diagnostics run after every epoch."
echo "[${VARIANT^^}] Retention: best_unseen.pt plus the latest epoch checkpoint."
echo "[${VARIANT^^}] Output: $OUTPUT_DIR"

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
  --lr-drop "$LR_DROP" \
  --clip-max-norm 0.1 \
  --lr-head "$LR_HEAD" \
  --lr-vit "$LR_VIT" \
  --lr-text-adapter "$LR_TEXT_ADAPTER" \
  --lr-obj-cond-adapter "$LR_OBJ_ADAPTER" \
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
  "${VARIANT_ARGS[@]}" \
  --adapter-contribution-diagnostics \
  --amp \
  --fast-cuda \
  --keep-last-checkpoints 1 \
  --keep-best-unseen-checkpoint \
  --print-interval "$PRINT_INTERVAL"

final_checkpoint="$(find_epoch_checkpoint "$EPOCHS" || true)"
if [[ -z "$final_checkpoint" ]]; then
  echo "[ERROR] ${VARIANT^^} exited without an epoch-$EPOCHS checkpoint." >&2
  exit 1
fi
if [[ ! -f "$OUTPUT_DIR/best_unseen.pt" ]]; then
  echo "[ERROR] ${VARIANT^^} exited without best_unseen.pt." >&2
  exit 1
fi

echo "[${VARIANT^^}] Final checkpoint: $final_checkpoint"
echo "[${VARIANT^^}] Best unseen checkpoint: $OUTPUT_DIR/best_unseen.pt"
echo "[${VARIANT^^}] Best metadata: $OUTPUT_DIR/best_unseen_checkpoint.json"
echo "[${VARIANT^^}] Diagnostics: $OUTPUT_DIR/scene_gate_diagnostics.csv"
