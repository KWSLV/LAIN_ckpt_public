#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export PATH="/root/miniconda3/bin:$PATH"
source scripts/server_assets.sh
prepare_lain_server_assets

SEED="${SEED:-66}"
EPOCHS=20
RUN_NAME="UO_a4_text_object_gate_last_r4a16_seed${SEED}"
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

final_checkpoint="$(find_epoch_checkpoint "$EPOCHS" || true)"
if [[ -n "$final_checkpoint" ]]; then
  echo "[A4] Complete: $final_checkpoint"
  /root/miniconda3/bin/python scripts/diagnostics/plot_module_contributions.py \
    "$OUTPUT_DIR/scene_gate_diagnostics.json" \
    --output "$OUTPUT_DIR/module_contributions.png"
  exit 0
fi

resume_checkpoint="$(find_latest_checkpoint || true)"
load_args=()
if [[ -n "$resume_checkpoint" ]]; then
  load_args=(--resume "$resume_checkpoint")
  echo "[A4] Resuming interrupted run from: $resume_checkpoint"
else
  echo "[A4] Starting a fresh 20-epoch joint run from DETR and CLIP weights."
fi

PORT="${PORT:-$((RANDOM % 5000 + 15000))}"
RDZV_ID="${RDZV_ID:-$((RANDOM % 5000 + 15000))}"

echo "[A4] Modules: Visual + Text(r4,a16,dim64) + Object(r4) + SceneGate(v5 pair)"
echo "[A4] Gate: hidden=128 rank=16 dropout=0.1 alpha=0.02 activation=standardized_tanh init_std=1e-3"
echo "[A4] Diagnostics: Gate/Text/Object conditional contribution every epoch; JSON/CSV/PNG + W&B."

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
  --adapter_pos last \
  --use_text_adapter \
  --text_adapter_dim 64 \
  --lora_rank 4 \
  --lora_alpha 16 \
  --adapter_residual_scale 0.05 \
  --adapter_dropout 0.1 \
  --use_obj_cond_adapter \
  --obj_cond_rank 4 \
  --use_scene_gate \
  --scene_gate_type pair \
  --scene_gate_version v5 \
  --scene_gate_hidden_dim 128 \
  --scene_gate_rank 16 \
  --scene_gate_dropout 0.1 \
  --scene_gate_alpha 0.02 \
  --scene_gate_activation standardized_tanh \
  --scene_gate_init_std 1e-3 \
  --scene_gate_start_epoch 1 \
  --scene-gate-diagnostics \
  --scene-gate-compare-off \
  --adapter-contribution-diagnostics \
  --amp \
  --fast-cuda \
  --keep-last-checkpoints 1 \
  --keep-best-unseen-checkpoint \
  --print-interval "$PRINT_INTERVAL"

final_checkpoint="$(find_epoch_checkpoint "$EPOCHS" || true)"
if [[ -z "$final_checkpoint" ]]; then
  echo "[ERROR] A4 exited without an epoch-20 checkpoint." >&2
  exit 1
fi
if [[ ! -f "$OUTPUT_DIR/best_unseen.pt" ]]; then
  echo "[ERROR] A4 exited without best_unseen.pt." >&2
  exit 1
fi

/root/miniconda3/bin/python scripts/diagnostics/plot_module_contributions.py \
  "$OUTPUT_DIR/scene_gate_diagnostics.json" \
  --output "$OUTPUT_DIR/module_contributions.png"

echo "[A4] Final checkpoint: $final_checkpoint"
echo "[A4] Best checkpoint: $OUTPUT_DIR/best_unseen.pt"
echo "[A4] Diagnostics JSON: $OUTPUT_DIR/scene_gate_diagnostics.json"
echo "[A4] Contribution chart: $OUTPUT_DIR/module_contributions.png"
