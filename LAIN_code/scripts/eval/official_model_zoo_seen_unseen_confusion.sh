#!/usr/bin/env bash
set -uo pipefail

# Sequentially evaluate the four checkpoints from the official LAIN model zoo.

CODE_ROOT="${CODE_ROOT:-/root/Lain/LAIN-main}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/root/autodl-tmp/Lain/official_model_zoo}"
RESULT_ROOT="${RESULT_ROOT:-/root/autodl-tmp/Lain/seen_unseen_confusion_eval_20260915/official_model_zoo}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$RESULT_ROOT"
STATUS_TSV="$RESULT_ROOT/status.tsv"
printf 'model\tstatus\tstarted_at\tfinished_at\tcheckpoint\tsha256\tlog\n' > "$STATUS_TSV"
cp "$0" "$RESULT_ROOT/$(basename "$0")"

cd "$CODE_ROOT" || exit 1
export PATH="/root/miniconda3/bin:$PATH"
source scripts/server_assets.sh
prepare_lain_server_assets

exec 9>"/tmp/lain_cuda_${CUDA_VISIBLE_DEVICES}.lock"
if ! flock -n 9; then
  echo "[ERROR] CUDA device $CUDA_VISIBLE_DEVICES is already locked." >&2
  exit 1
fi

run_eval() {
  local model="$1"
  local zs_type="$2"
  local n_ctx="$3"
  local use_csc="$4"
  local checkpoint="$CHECKPOINT_ROOT/$model.pt"
  local output_dir="$RESULT_ROOT/$model"
  local started_at finished_at checkpoint_sha status
  local -a csc_args=()

  mkdir -p "$output_dir"
  started_at="$(date --iso-8601=seconds)"
  if [[ ! -s "$checkpoint" ]]; then
    echo "[$model] SKIP: missing checkpoint $checkpoint" | tee "$output_dir/eval.log"
    printf '%s\tSKIPPED_NO_CHECKPOINT\t%s\t%s\t%s\t\t%s\n' \
      "$model" "$started_at" "$(date --iso-8601=seconds)" "$checkpoint" "$output_dir/eval.log" >> "$STATUS_TSV"
    return 0
  fi

  checkpoint_sha="$(sha256sum "$checkpoint" | awk '{print $1}')"
  [[ "$use_csc" == "1" ]] && csc_args+=(--CSC)

  echo "[$model] START zs_type=$zs_type checkpoint=$checkpoint" | tee "$output_dir/eval.log"
  local port=$((RANDOM % 5000 + 21000))
  local rdzv_id=$((RANDOM % 5000 + 21000))
  WANDB_MODE=disabled CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    torchrun \
      --rdzv_id "$rdzv_id" \
      --rdzv_backend c10d \
      --nproc_per_node 1 \
      --rdzv_endpoint "127.0.0.1:$port" \
      main.py \
      --pretrained "$DETR_PRETRAINED" \
      --clip_dir_vit "$CLIP_PRETRAINED" \
      --resume "$checkpoint" \
      --output-dir "$output_dir" \
      --data-root ./hicodet \
      --hico-image-root "$HICO_IMAGE_ROOT" \
      --dataset hicodet \
      --partitions train2015 test2015 \
      --zs --zs_type "$zs_type" \
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
      --N_CTX "$n_ctx" \
      --use_insadapter \
      --adapt_dim 32 \
      --use_prior \
      --adapter_alpha 1.0 \
      --adapter_num_layers 1 \
      --adapter_pos all \
      "${csc_args[@]}" \
      --amp \
      --fast-cuda \
      --seen-unseen-confusion \
      --seen-unseen-confusion-iou 0.5 \
      --eval 2>&1 | tee -a "$output_dir/eval.log"
  status=${PIPESTATUS[0]}
  finished_at="$(date --iso-8601=seconds)"
  if [[ "$status" -eq 0 ]]; then
    status="OK"
  else
    status="FAILED_$status"
  fi

  grep -Ei 'mAP|unseen|seen|full|rare|non.?rare|AP|confusion' "$output_dir/eval.log" \
    | tail -n 200 > "$output_dir/metrics.txt" || true
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$model" "$status" "$started_at" "$finished_at" "$checkpoint" "$checkpoint_sha" "$output_dir/eval.log" >> "$STATUS_TSV"
  echo "[$model] FINISH status=$status"
}

run_eval RF rare_first 24 1
run_eval NF non_rare_first 36 1
run_eval UO unseen_object 36 1
run_eval UV unseen_verb 36 0

date --iso-8601=seconds > "$RESULT_ROOT/finished_at.txt"
tar -C "$(dirname "$RESULT_ROOT")" -czf "$RESULT_ROOT.tar.gz" "$(basename "$RESULT_ROOT")"
echo "OFFICIAL_MODEL_ZOO_EVALS_FINISHED $(date --iso-8601=seconds)"
