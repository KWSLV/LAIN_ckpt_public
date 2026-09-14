#!/usr/bin/env bash
set -uo pipefail

# Evaluate the final epoch-20 checkpoints for reviewer experiments 01..15.
# Existing server checkpoints are deliberately ordered first. Experiments 06
# and 08 are evaluated last because their epoch-20 files may be restored from
# the public GitHub releases while this queue is already running.

CODE_ROOT="${CODE_ROOT:-/root/Lain/LAIN-main}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/Lain/reviewer_revision}"
RESULT_ROOT="${RESULT_ROOT:-/root/autodl-tmp/Lain/seen_unseen_confusion_eval_20260915/reviewer_revision}"
PUBLIC_REPO="${PUBLIC_REPO:-KWSLV/LAIN_ckpt_public}"
PUBLIC_REPO_DIR="${PUBLIC_REPO_DIR:-/root/autodl-tmp/LAIN_ckpt_public_upload_cache}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DO_UPLOAD="${DO_UPLOAD:-1}"
PUBLIC_RESULT_SUBDIR="${PUBLIC_RESULT_SUBDIR:-reviewer_revision/seen_unseen_confusion_eval_20260915}"

mkdir -p "$RESULT_ROOT"
STATUS_TSV="$RESULT_ROOT/status.tsv"
printf 'id\trun_name\tstatus\tstarted_at\tfinished_at\tcheckpoint\tsha256\tlog\n' > "$STATUS_TSV"
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
  local id="$1"
  local run_name="$2"
  local zs_type="$3"
  local n_ctx="$4"
  local use_csc="$5"
  local adapter_pos="$6"
  local variant="$7"
  local checkpoint_dir="$RUN_ROOT/$run_name/checkpoints"
  local output_dir="$RESULT_ROOT/$run_name"
  local checkpoint=""
  local started_at finished_at checkpoint_sha status
  local -a module_args=()
  local -a csc_args=()

  mkdir -p "$output_dir"
  checkpoint="$(find "$checkpoint_dir" -maxdepth 1 -type f -name 'ckpt_*_20.pt' -print 2>/dev/null | sort | head -n 1)"
  started_at="$(date --iso-8601=seconds)"

  if [[ -z "$checkpoint" || ! -f "$checkpoint" ]]; then
    echo "[$id] SKIP: no epoch-20 checkpoint in $checkpoint_dir" | tee "$output_dir/eval.log"
    printf '%s\t%s\tSKIPPED_NO_E20\t%s\t%s\t\t\t%s\n' \
      "$id" "$run_name" "$started_at" "$(date --iso-8601=seconds)" "$output_dir/eval.log" >> "$STATUS_TSV"
    return 0
  fi

  checkpoint_sha="$(sha256sum "$checkpoint" | awk '{print $1}')"
  [[ "$use_csc" == "1" ]] && csc_args+=(--CSC)

  case "$variant" in
    joint_lr1e4|joint_lr3e4|joint_alpha002|joint_alpha005|joint_start5)
      local gate_alpha="0.1"
      [[ "$variant" == "joint_alpha002" ]] && gate_alpha="0.02"
      [[ "$variant" == "joint_alpha005" ]] && gate_alpha="0.05"
      local gate_start_epoch="1"
      [[ "$variant" == "joint_start5" ]] && gate_start_epoch="5"
      module_args+=(
        --use_text_adapter --text_adapter_dim 64 --lora_rank 4 --lora_alpha 16
        --adapter_residual_scale 0.05 --adapter_dropout 0.1
        --use_scene_gate --scene_gate_type pair --scene_gate_version v5_legacy
        --scene_gate_hidden_dim 128 --scene_gate_rank 16 --scene_gate_dropout 0.1
        --scene_gate_alpha "$gate_alpha" --scene_gate_start_epoch "$gate_start_epoch"
        --scene-gate-diagnostics --scene-gate-compare-off
      )
      ;;
    text_rank4|text_rank8)
      local text_rank="4"
      [[ "$variant" == "text_rank8" ]] && text_rank="8"
      module_args+=(
        --use_text_adapter --text_adapter_dim 64 --lora_rank "$text_rank" --lora_alpha 16
        --adapter_residual_scale 0.05 --adapter_dropout 0.1
      )
      ;;
    gate_rank32|gate_rank64)
      local gate_rank="32"
      [[ "$variant" == "gate_rank64" ]] && gate_rank="64"
      module_args+=(
        --use_scene_gate --scene_gate_type pair --scene_gate_version v5_lowrank
        --scene_gate_hidden_dim 128 --scene_gate_rank "$gate_rank" --scene_gate_dropout 0.1
        --scene_gate_alpha 0.1 --scene_gate_start_epoch 1
        --scene_gate_activation tanh --scene_gate_init_std 0
        --scene-gate-diagnostics --scene-gate-compare-off
      )
      ;;
    gate_no_centering)
      module_args+=(
        --use_scene_gate --scene_gate_type pair --scene_gate_version v5_legacy
        --scene_gate_hidden_dim 128 --scene_gate_rank 16 --scene_gate_dropout 0.1
        --scene_gate_alpha 0.1 --scene_gate_start_epoch 1
        --scene-gate-disable-centering
        --scene-gate-diagnostics --scene-gate-compare-off
      )
      ;;
    baseline)
      ;;
    *)
      echo "[$id] ERROR: unknown variant $variant" | tee "$output_dir/eval.log"
      return 2
      ;;
  esac

  echo "[$id] START run=$run_name checkpoint=$checkpoint" | tee "$output_dir/eval.log"
  local port=$((RANDOM % 5000 + 15000))
  local rdzv_id=$((RANDOM % 5000 + 15000))
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
      --adapter_pos "$adapter_pos" \
      "${csc_args[@]}" \
      "${module_args[@]}" \
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

  grep -Ei 'mAP|unseen|seen|full|rare|non.?rare|AP' "$output_dir/eval.log" | tail -n 160 > "$output_dir/metrics.txt" || true
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$id" "$run_name" "$status" "$started_at" "$finished_at" "$checkpoint" "$checkpoint_sha" "$output_dir/eval.log" >> "$STATUS_TSV"
  echo "[$id] FINISH status=$status"
}

# All reviewer experiments are recorded in numerical order. Missing e20 files
# are explicit SKIPPED_NO_E20 rows rather than silently disappearing.
run_eval 01 01_uo_joint_gate_lr1e4_seed66 unseen_object 36 1 last joint_lr1e4
run_eval 02 02_uo_joint_gate_lr3e4_seed66 unseen_object 36 1 last joint_lr3e4
run_eval 03 03_uo_joint_gate_alpha002_seed66 unseen_object 36 1 last joint_alpha002
run_eval 04 04_uo_joint_gate_alpha005_seed66 unseen_object 36 1 last joint_alpha005
run_eval 05 05_uo_joint_gate_start5_seed66 unseen_object 36 1 last joint_start5
run_eval 06 06_uo_text_rank8_seed66 unseen_object 36 1 last text_rank8
run_eval 07 07_uo_paper_gate_rank32_seed66 unseen_object 36 1 all gate_rank32
run_eval 08 08_uo_paper_gate_rank64_seed66 unseen_object 36 1 all gate_rank64
run_eval 09 09_uo_paper_gate_no_centering_seed66 unseen_object 36 1 all gate_no_centering
run_eval 10 10_rfuc_lain_matched_seed66 rare_first 24 1 all baseline
run_eval 11 11_rfuc_text_rank4_seed66 rare_first 24 1 all text_rank4
run_eval 12 12_nfuc_lain_matched_seed66 non_rare_first 36 1 all baseline
run_eval 13 13_nfuc_text_rank4_seed66 non_rare_first 36 1 all text_rank4
run_eval 14 14_uv_lain_matched_seed66 unseen_verb 36 0 all baseline
run_eval 15 15_uv_text_rank4_seed66 unseen_verb 36 0 all text_rank4

date --iso-8601=seconds > "$RESULT_ROOT/finished_at.txt"
tar -C "$(dirname "$RESULT_ROOT")" -czf "$RESULT_ROOT.tar.gz" "$(basename "$RESULT_ROOT")"

if [[ "$DO_UPLOAD" != "1" ]]; then
  echo "REVIEWER_EVALS_FINISHED_UPLOAD_DEFERRED $(date --iso-8601=seconds)"
  exit 0
fi

# Ensure every evaluated e20 checkpoint is represented as a release asset.
while IFS=$'\t' read -r id run_name status _started _finished checkpoint _sha _log; do
  [[ "$id" == "id" || "$status" != "OK" || ! -f "$checkpoint" ]] && continue
  if ! gh release view "$run_name" --repo "$PUBLIC_REPO" >/dev/null 2>&1; then
    gh release create "$run_name" --repo "$PUBLIC_REPO" --title "$run_name" \
      --notes "Reviewer-revision experiment $id checkpoint archive and epoch-20 evaluation."
  fi
  asset_name="$(basename "$checkpoint")"
  if ! gh release view "$run_name" --repo "$PUBLIC_REPO" --json assets \
      --jq ".assets[] | select(.name == \"$asset_name\") | .name" | grep -qx "$asset_name"; then
    gh release upload "$run_name" "$checkpoint" --repo "$PUBLIC_REPO"
  fi
done < "$STATUS_TSV"

# Commit the small evaluation artifacts and logs to the public repository.
if [[ -d "$PUBLIC_REPO_DIR/.git" ]]; then
  for attempt in {1..20}; do
    git -C "$PUBLIC_REPO_DIR" pull --ff-only && break
    sleep $((attempt * 15))
  done
  destination="$PUBLIC_REPO_DIR/$PUBLIC_RESULT_SUBDIR"
  mkdir -p "$destination"
  cp -a "$RESULT_ROOT/." "$destination/"
  git -C "$PUBLIC_REPO_DIR" add "$PUBLIC_RESULT_SUBDIR"
  if ! git -C "$PUBLIC_REPO_DIR" diff --cached --quiet; then
    git -C "$PUBLIC_REPO_DIR" commit -m "Add reviewer experiments epoch-20 eval results"
    for attempt in {1..20}; do
      git -C "$PUBLIC_REPO_DIR" push origin main && break
      sleep $((attempt * 15))
    done
  fi
fi

echo "ALL_EVALS_AND_UPLOADS_FINISHED $(date --iso-8601=seconds)"
