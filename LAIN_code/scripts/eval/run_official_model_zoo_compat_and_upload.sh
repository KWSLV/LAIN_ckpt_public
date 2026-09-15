#!/usr/bin/env bash
set -uo pipefail

CODE_ROOT="${CODE_ROOT:-/root/Lain/LAIN-main}"
RESULT_ROOT="${RESULT_ROOT:-/root/autodl-tmp/Lain/official_model_zoo_compat_eval_20260915}"
PUBLIC_REPO="${PUBLIC_REPO:-KWSLV/LAIN_ckpt_public}"
PUBLIC_REPO_DIR="${PUBLIC_REPO_DIR:-/root/autodl-tmp/LAIN_ckpt_public_upload_cache}"
PUBLIC_SUBDIR="${PUBLIC_SUBDIR:-evaluations/official_model_zoo_compat_eval_20260915}"
LOG="$RESULT_ROOT/master.log"

mkdir -p "$RESULT_ROOT"
exec > >(tee -a "$LOG") 2>&1
echo "OFFICIAL_ONLY_START $(date --iso-8601=seconds)"
cd "$CODE_ROOT" || exit 1

CODE_ROOT="$CODE_ROOT" \
CHECKPOINT_ROOT=/root/autodl-tmp/Lain/official_model_zoo \
RESULT_ROOT="$RESULT_ROOT/results" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
bash scripts/eval/official_model_zoo_seen_unseen_confusion.sh
eval_status=$?
echo "OFFICIAL_ONLY_EVAL_END status=$eval_status $(date --iso-8601=seconds)"

retry_command() {
  local attempt
  for attempt in {1..20}; do
    "$@" && return 0
    sleep "$((attempt < 10 ? attempt * 15 : 150))"
  done
  return 1
}

printf 'eval_status=%s\nfinished_at=%s\n' \
  "$eval_status" "$(date --iso-8601=seconds)" > "$RESULT_ROOT/final_status.txt"
archive="$RESULT_ROOT.tar.gz"
tar -C "$(dirname "$RESULT_ROOT")" -czf "$archive" "$(basename "$RESULT_ROOT")"

tag=official_model_zoo_compat_eval_20260915
if ! gh release view "$tag" --repo "$PUBLIC_REPO" >/dev/null 2>&1; then
  retry_command gh release create "$tag" --repo "$PUBLIC_REPO" \
    --title "Official LAIN model-zoo compatible confusion evaluation" \
    --notes "RF/NF/UO/UV official-public eval semantics with seen/unseen confusion outputs."
fi
retry_command gh release upload "$tag" "$archive" --repo "$PUBLIC_REPO"

if [[ -d "$PUBLIC_REPO_DIR/.git" ]]; then
  retry_command git -C "$PUBLIC_REPO_DIR" pull --ff-only
  destination="$PUBLIC_REPO_DIR/$PUBLIC_SUBDIR"
  mkdir -p "$destination" "$PUBLIC_REPO_DIR/LAIN_code/scripts/eval" \
    "$PUBLIC_REPO_DIR/LAIN_code/utils"
  cp -a "$RESULT_ROOT/." "$destination/"
  cp "$CODE_ROOT/main.py" "$PUBLIC_REPO_DIR/LAIN_code/main.py"
  cp "$CODE_ROOT/analysis.py" "$PUBLIC_REPO_DIR/LAIN_code/analysis.py"
  cp "$CODE_ROOT/engine.py" "$PUBLIC_REPO_DIR/LAIN_code/engine.py"
  cp "$CODE_ROOT/utils/args.py" "$PUBLIC_REPO_DIR/LAIN_code/utils/args.py"
  cp "$CODE_ROOT/scripts/eval/official_model_zoo_seen_unseen_confusion.sh" \
    "$PUBLIC_REPO_DIR/LAIN_code/scripts/eval/"
  cp "$CODE_ROOT/scripts/eval/run_official_model_zoo_compat_and_upload.sh" \
    "$PUBLIC_REPO_DIR/LAIN_code/scripts/eval/"
  git -C "$PUBLIC_REPO_DIR" add "$PUBLIC_SUBDIR" LAIN_code
  if ! git -C "$PUBLIC_REPO_DIR" diff --cached --quiet; then
    git -C "$PUBLIC_REPO_DIR" commit -m "Add official model-zoo compatible confusion eval"
    retry_command git -C "$PUBLIC_REPO_DIR" push origin main
  fi
fi

echo "OFFICIAL_ONLY_AND_UPLOAD_FINISHED $(date --iso-8601=seconds)"
