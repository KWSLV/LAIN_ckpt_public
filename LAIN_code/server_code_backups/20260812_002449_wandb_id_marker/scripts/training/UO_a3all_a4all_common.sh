#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "a3all" && "$1" != "a4all" ) ]]; then
  echo "Usage: bash $0 {a3all|a4all}" >&2
  exit 2
fi

VARIANT="$1"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export PATH="/root/miniconda3/bin:$PATH"
source scripts/server_assets.sh
prepare_lain_server_assets

SEED="${SEED:-66}"
EPOCHS=20
RUN_NAME="UO_${VARIANT}_seed${SEED}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/Lain/$RUN_NAME}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/checkpoints}"
WANDB_DIR="${WANDB_DIR:-$RUN_ROOT/wandb}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PRINT_INTERVAL="${PRINT_INTERVAL:-500}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

GATE_ARGS=()
if [[ "$VARIANT" == "a4all" ]]; then
  MODULE_SUMMARY="Visual + Text(r4,a16,dim64) + Object(r4) + SceneGate(v5 pair)"
  GATE_ARGS=(
    --use_scene_gate
    --scene_gate_type pair
    --scene_gate_version v5
    --scene_gate_hidden_dim 128
    --scene_gate_rank 16
    --scene_gate_dropout 0.1
    --scene_gate_alpha 0.02
    --scene_gate_activation standardized_tanh
    --scene_gate_init_std 1e-3
    --scene_gate_start_epoch 1
    --scene-gate-diagnostics
    --scene-gate-compare-off
  )
else
  MODULE_SUMMARY="Visual + Text(r4,a16,dim64) + Object(r4); SceneGate OFF"
fi

mkdir -p "$OUTPUT_DIR" "$WANDB_DIR" "$RUN_ROOT/logs"

exec 9>"/tmp/lain_cuda_${CUDA_VISIBLE_DEVICES}.lock"
if ! flock -n 9; then
  echo "[ERROR] Another managed LAIN job is using CUDA device $CUDA_VISIBLE_DEVICES." >&2
  exit 1
fi

available_kb="$(df -Pk "$RUN_ROOT" | awk 'NR==2 {print $4}')"
minimum_kb=$((8 * 1024 * 1024))
if (( available_kb < minimum_kb )); then
  echo "[ERROR] Less than 8 GiB is available under $RUN_ROOT." >&2
  echo "[ERROR] A 20-epoch keep-every-epoch run is not started." >&2
  exit 1
fi

find_epoch_checkpoint() {
  local epoch_padded
  local matches=()
  epoch_padded="$(printf '%02d' "$1")"
  shopt -s nullglob
  matches=("$OUTPUT_DIR"/ckpt_*_"$epoch_padded".pt)
  shopt -u nullglob
  (( ${#matches[@]} > 0 )) || return 1
  printf '%s\n' "${matches[-1]}"
}

find_latest_checkpoint() {
  local matches=()
  shopt -s nullglob
  matches=("$OUTPUT_DIR"/ckpt_*.pt)
  shopt -u nullglob
  (( ${#matches[@]} > 0 )) || return 1
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
    "adapter_pos": "all",
    "use_text_adapter": True,
    "text_adapter_dim": 64,
    "lora_rank": 4,
    "lora_alpha": 16,
    "adapter_residual_scale": 0.05,
    "adapter_dropout": 0.1,
    "use_obj_cond_adapter": True,
    "obj_cond_rank": 4,
    "use_scene_gate": variant == "a4all",
    "adapter_contribution_diagnostics": True,
    "keep_last_checkpoints": 0,
    "keep_epoch_checkpoints": [],
    "keep_best_unseen_checkpoint": True,
    "lr_drop": 10,
    "lr_head": 5e-4,
    "lr_vit": 2e-4,
    "lr_text_adapter": 2e-4,
    "lr_obj_cond_adapter": 2e-4,
}
if variant == "a4all":
    expected.update({
        "scene_gate_type": "pair",
        "scene_gate_version": "v5",
        "scene_gate_hidden_dim": 128,
        "scene_gate_rank": 16,
        "scene_gate_dropout": 0.1,
        "scene_gate_alpha": 0.02,
        "scene_gate_activation": "standardized_tanh",
        "scene_gate_init_std": 1e-3,
        "scene_gate_start_epoch": 1,
        "scene_gate_diagnostics": True,
        "scene_gate_compare_off": True,
        "lr_scene_gate": 1e-3,
    })

mismatches = {
    key: (args.get(key), value)
    for key, value in expected.items()
    if args.get(key) != value
}
# Compatible one-time migration from the original milestone-only policy. The
# resumed main process rewrites args.txt with the new keep-every-epoch values.
if mismatches.get("keep_last_checkpoints") == (1, 0):
    mismatches.pop("keep_last_checkpoints")
if mismatches.get("keep_epoch_checkpoints") == ([5, 10, 15, 20], []):
    mismatches.pop("keep_epoch_checkpoints")
if mismatches:
    details = ", ".join(
        f"{key}: found={found!r}, expected={wanted!r}"
        for key, (found, wanted) in mismatches.items()
    )
    raise SystemExit(
        f"[ERROR] Existing {variant} directory has incompatible args: {details}"
    )
PY
}

verify_outputs() {
  local epoch
  local retention_start_epoch=1
  local retention_marker="$RUN_ROOT/checkpoint_retention_start_epoch.txt"
  if [[ -f "$retention_marker" ]]; then
    retention_start_epoch="$(tr -d '[:space:]' < "$retention_marker")"
  fi
  if [[ ! "$retention_start_epoch" =~ ^[0-9]+$ ]] || (( retention_start_epoch < 1 || retention_start_epoch > EPOCHS )); then
    echo "[ERROR] Invalid retention start epoch: $retention_start_epoch" >&2
    return 1
  fi
  for ((epoch=retention_start_epoch; epoch<=EPOCHS; epoch++)); do
    if [[ -z "$(find_epoch_checkpoint "$epoch" || true)" ]]; then
      echo "[ERROR] ${VARIANT} is missing the retained epoch-$epoch checkpoint." >&2
      return 1
    fi
  done
  if [[ ! -f "$OUTPUT_DIR/best_unseen.pt" ]]; then
    echo "[ERROR] ${VARIANT} is missing best_unseen.pt." >&2
    return 1
  fi
}

plot_contributions() {
  local json_path="$OUTPUT_DIR/scene_gate_diagnostics.json"
  [[ -f "$json_path" ]] || return 0
  /root/miniconda3/bin/python scripts/diagnostics/plot_module_contributions.py \
    "$json_path" \
    --output "$OUTPUT_DIR/module_contributions.png"
}

validate_existing_args

final_checkpoint="$(find_epoch_checkpoint "$EPOCHS" || true)"
if [[ -n "$final_checkpoint" ]]; then
  verify_outputs
  plot_contributions
  echo "[${VARIANT^^}] Complete: $final_checkpoint"
  echo "[${VARIANT^^}] Best unseen: $OUTPUT_DIR/best_unseen.pt"
  exit 0
fi

resume_checkpoint="$(find_latest_checkpoint || true)"
load_args=()
if [[ -n "$resume_checkpoint" ]]; then
  load_args=(--resume "$resume_checkpoint")
  echo "[${VARIANT^^}] Resuming interrupted run from: $resume_checkpoint"
else
  echo "[${VARIANT^^}] Starting from DETR and CLIP pretrained weights."
fi

# Keeping every epoch needs substantially more data-disk space than the old
# milestone-only policy. Estimate the remaining checkpoint cost from the most
# recent file (or 750 MiB for a fresh run), plus a 2 GiB safety reserve.
latest_epoch=0
estimated_checkpoint_kb=$((750 * 1024))
if [[ -n "$resume_checkpoint" ]]; then
  checkpoint_name="$(basename "$resume_checkpoint")"
  checkpoint_epoch="${checkpoint_name%.pt}"
  checkpoint_epoch="${checkpoint_epoch##*_}"
  latest_epoch=$((10#$checkpoint_epoch))
  estimated_checkpoint_kb=$(( ($(stat -c '%s' "$resume_checkpoint") + 1023) / 1024 ))
fi
remaining_epochs=$((EPOCHS - latest_epoch))
required_kb=$((remaining_epochs * estimated_checkpoint_kb + 2 * 1024 * 1024))
available_kb="$(df -Pk "$RUN_ROOT" | awk 'NR==2 {print $4}')"
if (( available_kb < required_kb )); then
  echo "[ERROR] Not enough data-disk space to retain every remaining epoch." >&2
  echo "[ERROR] Available: $((available_kb / 1024)) MiB; estimated required: $((required_kb / 1024)) MiB." >&2
  exit 1
fi

# Continue the existing online W&B run when resuming. WANDB's local internal
# files may create a new service segment, but the dashboard/run ID remains one.
if [[ -n "$resume_checkpoint" && -z "${WANDB_RUN_ID:-}" ]]; then
  latest_wandb_run="$(readlink -f "$WANDB_DIR/wandb/latest-run" 2>/dev/null || true)"
  if [[ -n "$latest_wandb_run" ]]; then
    latest_wandb_name="$(basename "$latest_wandb_run")"
    detected_wandb_id="${latest_wandb_name##*-}"
    if [[ "$detected_wandb_id" =~ ^[[:alnum:]]+$ ]]; then
      export WANDB_RUN_ID="$detected_wandb_id"
      export WANDB_RESUME=must
      echo "[${VARIANT^^}] Resuming W&B run ID: $WANDB_RUN_ID"
    fi
  fi
fi

PORT="${PORT:-$((RANDOM % 5000 + 15000))}"
RDZV_ID="${RDZV_ID:-$((RANDOM % 5000 + 15000))}"

echo "[${VARIANT^^}] Modules: $MODULE_SUMMARY"
echo "[${VARIANT^^}] UO seed=$SEED, epochs=20, adapter_pos=all"
echo "[${VARIANT^^}] LR: head=5e-4, vit/text/object=2e-4, gate=1e-3, drop=10"
echo "[${VARIANT^^}] Checkpoints: every epoch + best_unseen.pt"
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
  --lr-drop 10 \
  --clip-max-norm 0.1 \
  --lr-head 5e-4 \
  --lr-vit 2e-4 \
  --lr-text-adapter 2e-4 \
  --lr-obj-cond-adapter 2e-4 \
  --lr-scene-gate 1e-3 \
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
  --use_text_adapter \
  --text_adapter_dim 64 \
  --lora_rank 4 \
  --lora_alpha 16 \
  --adapter_residual_scale 0.05 \
  --adapter_dropout 0.1 \
  --use_obj_cond_adapter \
  --obj_cond_rank 4 \
  "${GATE_ARGS[@]}" \
  --adapter-contribution-diagnostics \
  --amp \
  --fast-cuda \
  --keep-last-checkpoints 0 \
  --keep-best-unseen-checkpoint \
  --print-interval "$PRINT_INTERVAL"

final_checkpoint="$(find_epoch_checkpoint "$EPOCHS" || true)"
if [[ -z "$final_checkpoint" ]]; then
  echo "[ERROR] ${VARIANT} exited without an epoch-20 checkpoint." >&2
  exit 1
fi
verify_outputs
plot_contributions

echo "[${VARIANT^^}] Final checkpoint: $final_checkpoint"
echo "[${VARIANT^^}] Best checkpoint: $OUTPUT_DIR/best_unseen.pt"
echo "[${VARIANT^^}] Diagnostics: $OUTPUT_DIR/scene_gate_diagnostics.json"
echo "[${VARIANT^^}] Contribution chart: $OUTPUT_DIR/module_contributions.png"
