#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export PATH="/root/miniconda3/bin:$PATH"
source scripts/server_assets.sh
prepare_lain_server_assets

RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/Lain/UO_a4all_seed66}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$RUN_ROOT/checkpoints}"
DIAGNOSTICS_JSON="${DIAGNOSTICS_JSON:-$CHECKPOINT_DIR/scene_gate_diagnostics.json}"
WORK_DIR="${WORK_DIR:-$RUN_ROOT/gate_off_checkpoint_eval}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
FORCE="${FORCE:-0}"

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "[ERROR] Checkpoint directory not found: $CHECKPOINT_DIR" >&2
  exit 1
fi
if [[ ! -f "$DIAGNOSTICS_JSON" ]]; then
  echo "[ERROR] Diagnostics JSON not found: $DIAGNOSTICS_JSON" >&2
  exit 1
fi
if [[ "$FORCE" != "0" && "$FORCE" != "1" ]]; then
  echo "[ERROR] FORCE must be 0 or 1." >&2
  exit 1
fi

mkdir -p "$WORK_DIR/results" "$WORK_DIR/logs"

# Serialize all managed LAIN GPU work. When a3all/a4all training is still
# running, this process waits here and scans checkpoints only after training
# releases the lock, so the final epoch is included and VRAM is not shared.
exec 9>"/tmp/lain_cuda_${CUDA_VISIBLE_DEVICES}.lock"
echo "[Gate-OFF Batch] Waiting for CUDA device $CUDA_VISIBLE_DEVICES..."
flock 9
echo "[Gate-OFF Batch] CUDA device $CUDA_VISIBLE_DEVICES acquired."

timestamp="$(date +%Y%m%d_%H%M%S)"
backup="$CHECKPOINT_DIR/scene_gate_diagnostics.before_gate_off_posthoc_$timestamp.json"
cp -a "$DIAGNOSTICS_JSON" "$backup"
echo "[Gate-OFF Batch] Diagnostics backup: $backup"

already_complete() {
  local epoch="$1"
  local checkpoint_name="$2"
  [[ "$FORCE" == "1" ]] && return 1
  /root/miniconda3/bin/python - "$DIAGNOSTICS_JSON" "$epoch" "$checkpoint_name" <<'PY'
import json
import sys
path, epoch, checkpoint = sys.argv[1], int(sys.argv[2]), sys.argv[3]
payload = json.load(open(path, encoding="utf-8"))
rows = payload.get("rows", []) if isinstance(payload, dict) else payload
row = next((item for item in rows if int(item.get("epoch", -1)) == epoch), {})
ablation = row.get("gate_ablation", {})
complete = (
    ablation.get("checkpoint") == checkpoint
    and ablation.get("scene_gate_enabled_during_evaluation") is False
    and all(key in ablation.get("gate_off", {}) for key in ("full", "unseen", "seen"))
    and all(key in ablation.get("contribution", {}) for key in ("full", "unseen", "seen"))
)
raise SystemExit(0 if complete else 1)
PY
}

mapfile -t checkpoints < <(find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name 'ckpt_*.pt' -printf '%f\n' | sort)
if (( ${#checkpoints[@]} == 0 )); then
  echo "[ERROR] No ckpt_*.pt files found in $CHECKPOINT_DIR." >&2
  exit 1
fi

completed=0
skipped=0
for checkpoint_name in "${checkpoints[@]}"; do
  checkpoint="$CHECKPOINT_DIR/$checkpoint_name"
  stem="${checkpoint_name%.pt}"
  epoch_text="${stem##*_}"
  if [[ ! "$epoch_text" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Cannot parse epoch from checkpoint: $checkpoint_name" >&2
    exit 1
  fi
  epoch=$((10#$epoch_text))
  result_json="$WORK_DIR/results/epoch_$(printf '%02d' "$epoch")_gate_off.json"
  eval_output="$WORK_DIR/eval_epoch_$(printf '%02d' "$epoch")"
  eval_log="$WORK_DIR/logs/epoch_$(printf '%02d' "$epoch").log"

  if already_complete "$epoch" "$checkpoint_name"; then
    echo "[Gate-OFF Batch] Skip completed epoch $epoch: $checkpoint_name"
    skipped=$((skipped + 1))
    continue
  fi

  echo "[Gate-OFF Batch] Evaluate epoch $epoch: $checkpoint_name"
  rm -f "$result_json.tmp"
  mkdir -p "$eval_output"
  PORT="${PORT:-$((RANDOM % 5000 + 15000))}"
  RDZV_ID="${RDZV_ID:-$((RANDOM % 5000 + 15000))}"

  WANDB_MODE=disabled CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  torchrun \
    --rdzv_id "$RDZV_ID" \
    --rdzv_backend c10d \
    --nproc_per_node 1 \
    --rdzv_endpoint "127.0.0.1:$PORT" \
    main.py \
    --pretrained "$DETR_PRETRAINED" \
    --clip_dir_vit "$CLIP_PRETRAINED" \
    --resume "$checkpoint" \
    --output-dir "$eval_output" \
    --eval-result-json "$result_json" \
    --hico-image-root "$HICO_IMAGE_ROOT" \
    --dataset hicodet \
    --partitions train2015 test2015 \
    --zs --zs_type unseen_object \
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
    --adapter_pos all \
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
    --amp \
    --fast-cuda \
    --eval \
    --eval-scene-gate-off-only \
    > >(tee "$eval_log") 2>&1

  if [[ ! -s "$result_json" ]]; then
    echo "[ERROR] Gate-OFF result was not created: $result_json" >&2
    exit 1
  fi

  # Lock + atomic replace protects the shared diagnostics file from partial
  # writes. The batch starts only after training, but the lock also protects
  # manual concurrent post-hoc evaluators.
  exec 8>"$CHECKPOINT_DIR/.scene_gate_diagnostics.lock"
  flock 8
  /root/miniconda3/bin/python \
    scripts/diagnostics/merge_gate_off_checkpoint_eval.py \
    --diagnostics "$DIAGNOSTICS_JSON" \
    --result "$result_json" \
    --expected-checkpoint "$checkpoint"
  flock -u 8

  completed=$((completed + 1))
done

echo "[Gate-OFF Batch] Complete. evaluated=$completed skipped=$skipped total=${#checkpoints[@]}"
echo "[Gate-OFF Batch] Updated JSON: $DIAGNOSTICS_JSON"
