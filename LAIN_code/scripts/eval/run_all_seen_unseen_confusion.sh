#!/usr/bin/env bash
set -uo pipefail

# One unattended pipeline:
#   1) verify the four official checkpoints already uploaded from the workstation;
#   2) restore the reviewer 06/08 e20 release assets and verify their checksums;
#   3) evaluate reviewer experiments 01..15 in order (missing e20 => explicit skip);
#   4) evaluate official RF/NF/UO/UV checkpoints in order;
#   5) publish checkpoints as release assets and commit all small eval artifacts.

CODE_ROOT="${CODE_ROOT:-/root/Lain/LAIN-main}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/Lain/reviewer_revision}"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-/root/autodl-tmp/Lain/official_model_zoo}"
RESULT_BASE="${RESULT_BASE:-/root/autodl-tmp/Lain/seen_unseen_confusion_eval_20260915}"
PUBLIC_REPO="${PUBLIC_REPO:-KWSLV/LAIN_ckpt_public}"
PUBLIC_REPO_DIR="${PUBLIC_REPO_DIR:-/root/autodl-tmp/LAIN_ckpt_public_upload_cache}"
PUBLIC_RESULT_SUBDIR="${PUBLIC_RESULT_SUBDIR:-evaluations/seen_unseen_confusion_20260915}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MASTER_LOG="$RESULT_BASE/master.log"

mkdir -p "$RESULT_BASE"
exec > >(tee -a "$MASTER_LOG") 2>&1
echo "PIPELINE_START $(date --iso-8601=seconds)"
echo "CODE_ROOT=$CODE_ROOT"
echo "RESULT_BASE=$RESULT_BASE"

cd "$CODE_ROOT" || exit 1
export PATH="/root/miniconda3/bin:$PATH"

verify_size() {
  local path="$1"
  local expected="$2"
  local actual
  if [[ ! -f "$path" ]]; then
    echo "[FATAL] Missing required checkpoint: $path"
    return 1
  fi
  actual="$(stat -c %s "$path")"
  if [[ "$actual" != "$expected" ]]; then
    echo "[FATAL] Size mismatch: $path actual=$actual expected=$expected"
    return 1
  fi
}

# These are the byte sizes of the four extracted files from the official archive.
verify_size "$OFFICIAL_ROOT/RF.pt" 730057574 || exit 20
verify_size "$OFFICIAL_ROOT/NF.pt" 735808358 || exit 20
verify_size "$OFFICIAL_ROOT/UO.pt" 735808358 || exit 20
verify_size "$OFFICIAL_ROOT/UV.pt" 710151014 || exit 20
(
  cd "$OFFICIAL_ROOT" || exit 1
  sha256sum RF.pt NF.pt UO.pt UV.pt > SHA256SUMS
)
echo "OFFICIAL_CHECKPOINTS_VERIFIED $(date --iso-8601=seconds)"

# 06 and 08 are the only reviewer e20 checkpoints found in GitHub but not as
# valid full files on the server. The downloader is resumable and checksum-aware.
download_ok=0
for attempt in {1..20}; do
  echo "REVIEWER_DOWNLOAD_ATTEMPT=$attempt $(date --iso-8601=seconds)"
  if python scripts/eval/reviewer_revision_epoch20_download.py; then
    download_ok=1
    break
  fi
  sleep "$((attempt < 10 ? attempt * 15 : 150))"
done
if [[ "$download_ok" != "1" ]]; then
  echo "[ERROR] Reviewer 06/08 download did not pass checksums after all retries; their eval rows will be skipped."
fi

find "$RUN_ROOT" -mindepth 3 -maxdepth 3 -type f -path '*/checkpoints/ckpt_*_20.pt' \
  -printf '%p\t%s\n' | sort > "$RESULT_BASE/reviewer_e20_inventory.tsv"

echo "REVIEWER_EVAL_BEGIN $(date --iso-8601=seconds)"
CODE_ROOT="$CODE_ROOT" RUN_ROOT="$RUN_ROOT" \
RESULT_ROOT="$RESULT_BASE/reviewer_revision" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
DO_UPLOAD=0 bash scripts/eval/reviewer_revision_epoch20_all.sh
reviewer_status=$?
echo "REVIEWER_EVAL_END status=$reviewer_status $(date --iso-8601=seconds)"

echo "OFFICIAL_EVAL_BEGIN $(date --iso-8601=seconds)"
CODE_ROOT="$CODE_ROOT" CHECKPOINT_ROOT="$OFFICIAL_ROOT" \
RESULT_ROOT="$RESULT_BASE/official_model_zoo" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
bash scripts/eval/official_model_zoo_seen_unseen_confusion.sh
official_status=$?
echo "OFFICIAL_EVAL_END status=$official_status $(date --iso-8601=seconds)"

retry_command() {
  local attempt
  for attempt in {1..20}; do
    "$@" && return 0
    sleep "$((attempt < 10 ? attempt * 15 : 150))"
  done
  return 1
}

ensure_release() {
  local tag="$1"
  local title="$2"
  local notes="$3"
  gh release view "$tag" --repo "$PUBLIC_REPO" >/dev/null 2>&1 && return 0
  retry_command gh release create "$tag" --repo "$PUBLIC_REPO" --title "$title" --notes "$notes"
}

upload_asset_if_missing() {
  local tag="$1"
  local path="$2"
  local asset_name
  [[ -f "$path" ]] || return 1
  asset_name="$(basename "$path")"
  if gh release view "$tag" --repo "$PUBLIC_REPO" --json assets \
      --jq ".assets[] | select(.name == \"$asset_name\") | .name" | grep -qx "$asset_name"; then
    echo "RELEASE_ASSET_EXISTS tag=$tag asset=$asset_name"
    return 0
  fi
  retry_command gh release upload "$tag" "$path" --repo "$PUBLIC_REPO"
}

# Mirror every successfully evaluated reviewer e20 checkpoint as a release asset.
reviewer_status_file="$RESULT_BASE/reviewer_revision/status.tsv"
if [[ -f "$reviewer_status_file" ]]; then
  while IFS=$'\t' read -r id run_name status _started _finished checkpoint _sha _log; do
    [[ "$id" == "id" || "$status" != "OK" || ! -f "$checkpoint" ]] && continue
    ensure_release "$run_name" "$run_name" \
      "Reviewer-revision experiment $id epoch-20 checkpoint and seen/unseen confusion evaluation."
    upload_asset_if_missing "$run_name" "$checkpoint"
  done < "$reviewer_status_file"
fi

# Mirror the official four-file LAIN model zoo in one public release.
official_tag="official_lain_model_zoo"
ensure_release "$official_tag" "Official LAIN model zoo mirror" \
  "RF, NF, UO and UV checkpoints mirrored from the official OreoChocolate/LAIN model-zoo archive."
for model in RF NF UO UV; do
  upload_asset_if_missing "$official_tag" "$OFFICIAL_ROOT/$model.pt"
done

# Add a downloadable result archive as well as the browsable files committed below.
tar -C "$(dirname "$RESULT_BASE")" -czf "$RESULT_BASE.tar.gz" "$(basename "$RESULT_BASE")"
result_tag="seen_unseen_confusion_eval_20260915"
ensure_release "$result_tag" "Seen/unseen confusion evaluations (2026-09-15)" \
  "Evaluation logs, metrics, counts, rates, CSV/JSON summaries and confusion-matrix figures."
upload_asset_if_missing "$result_tag" "$RESULT_BASE.tar.gz"

# Publish code and browsable results to the public repository.
if [[ ! -d "$PUBLIC_REPO_DIR/.git" ]]; then
  retry_command gh repo clone "$PUBLIC_REPO" "$PUBLIC_REPO_DIR"
fi
if [[ -d "$PUBLIC_REPO_DIR/.git" ]]; then
  retry_command git -C "$PUBLIC_REPO_DIR" pull --ff-only
  destination="$PUBLIC_REPO_DIR/$PUBLIC_RESULT_SUBDIR"
  mkdir -p "$destination" "$PUBLIC_REPO_DIR/LAIN_code/scripts/eval" "$PUBLIC_REPO_DIR/LAIN_code/utils"
  cp -a "$RESULT_BASE/." "$destination/"
  cp "$CODE_ROOT/analysis.py" "$PUBLIC_REPO_DIR/LAIN_code/analysis.py"
  cp "$CODE_ROOT/engine.py" "$PUBLIC_REPO_DIR/LAIN_code/engine.py"
  cp "$CODE_ROOT/utils/args.py" "$PUBLIC_REPO_DIR/LAIN_code/utils/args.py"
  cp "$CODE_ROOT/scripts/eval/reviewer_revision_epoch20_download.py" "$PUBLIC_REPO_DIR/LAIN_code/scripts/eval/"
  cp "$CODE_ROOT/scripts/eval/reviewer_revision_epoch20_all.sh" "$PUBLIC_REPO_DIR/LAIN_code/scripts/eval/"
  cp "$CODE_ROOT/scripts/eval/official_model_zoo_seen_unseen_confusion.sh" "$PUBLIC_REPO_DIR/LAIN_code/scripts/eval/"
  cp "$CODE_ROOT/scripts/eval/run_all_seen_unseen_confusion.sh" "$PUBLIC_REPO_DIR/LAIN_code/scripts/eval/"
  git -C "$PUBLIC_REPO_DIR" add "$PUBLIC_RESULT_SUBDIR" LAIN_code
  if ! git -C "$PUBLIC_REPO_DIR" diff --cached --quiet; then
    git -C "$PUBLIC_REPO_DIR" commit -m "Add seen/unseen confusion analysis and evaluation results"
    retry_command git -C "$PUBLIC_REPO_DIR" push origin main
  fi
fi

printf 'reviewer_status=%s\nofficial_status=%s\nfinished_at=%s\n' \
  "$reviewer_status" "$official_status" "$(date --iso-8601=seconds)" > "$RESULT_BASE/final_status.txt"
echo "ALL_EVALS_AND_UPLOADS_FINISHED $(date --iso-8601=seconds)"
