#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export PATH="/root/miniconda3/bin:$PATH"

SEED="${SEED:-66}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
INITIAL_DELAY_SECONDS="${INITIAL_DELAY_SECONDS:-18000}"
POLL_SECONDS="${POLL_SECONDS:-600}"
STABILITY_SECONDS="${STABILITY_SECONDS:-10}"
OBJECT_RUN_ROOT="${OBJECT_RUN_ROOT:-/root/autodl-tmp/Lain/UO_a4_ep9_objreset_objectonly_r4_seed66}"
OBJECT_CHECKPOINT_DIR="$OBJECT_RUN_ROOT/checkpoints"
DIAGNOSTICS_JSON="$OBJECT_CHECKPOINT_DIR/scene_gate_diagnostics.json"
TRAIN_SCRIPT="$ROOT/scripts/training/UO_objectbest_gateonly_v5h128_a004.sh"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/Lain/UO_objectbest_gateonly_v5h128_a004_seed${SEED}}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

server_has_gpu_work() {
  exec 8>"/tmp/lain_cuda_${CUDA_VISIBLE_DEVICES}.lock"
  if ! flock -n 8; then
    return 0
  fi
  flock -u 8

  if pgrep -f '(^|/)(torchrun|python)([[:space:]]|.*[[:space:]])[^[:cntrl:]]*main\.py' >/dev/null 2>&1; then
    return 0
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '^[[:space:]]*[0-9]+'; then
      return 0
    fi
  fi
  return 1
}

select_best_object_checkpoint() {
  python - "$DIAGNOSTICS_JSON" "$OBJECT_CHECKPOINT_DIR" <<'PY'
import glob
import json
import math
import os
import sys

diagnostics_path, checkpoint_dir = sys.argv[1:]
with open(diagnostics_path, "r", encoding="utf-8") as handle:
    records = json.load(handle)

valid = []
for record in records:
    try:
        epoch = int(record["epoch"])
        unseen = float(record["on"]["unseen"])
    except (KeyError, TypeError, ValueError):
        continue
    if math.isfinite(unseen):
        valid.append((unseen, epoch))

if not valid:
    raise SystemExit("no completed Object-only epoch with finite on.unseen")

unseen, epoch = max(valid)
candidates = sorted(glob.glob(os.path.join(checkpoint_dir, f"ckpt_*_{epoch:02d}.pt")))
if len(candidates) != 1:
    raise SystemExit(
        f"expected one checkpoint for best epoch {epoch}, found {len(candidates)}"
    )

checkpoint = os.path.realpath(candidates[0])
if not checkpoint.startswith(os.path.realpath(checkpoint_dir) + os.sep):
    raise SystemExit("selected checkpoint escaped the Object-only checkpoint directory")
print(f"{checkpoint}\t{epoch}\t{unseen:.12f}")
PY
}

echo "[$(timestamp)] Deferred Gate launcher started."
echo "[$(timestamp)] Initial wait: ${INITIAL_DELAY_SECONDS}s; poll interval: ${POLL_SECONDS}s."
echo "[$(timestamp)] Object diagnostics: $DIAGNOSTICS_JSON"
sleep "$INITIAL_DELAY_SECONDS"

while true; do
  if server_has_gpu_work; then
    echo "[$(timestamp)] Server still has training/evaluation or another GPU process; retrying in ${POLL_SECONDS}s."
    sleep "$POLL_SECONDS"
    continue
  fi

  if [[ ! -s "$DIAGNOSTICS_JSON" ]]; then
    echo "[$(timestamp)] Diagnostics JSON is missing or empty; retrying in ${POLL_SECONDS}s."
    sleep "$POLL_SECONDS"
    continue
  fi

  if ! selection="$(select_best_object_checkpoint 2>&1)"; then
    echo "[$(timestamp)] Cannot select a complete Object checkpoint: $selection"
    sleep "$POLL_SECONDS"
    continue
  fi

  IFS=$'\t' read -r source_checkpoint source_epoch source_unseen <<< "$selection"
  if [[ ! -f "$source_checkpoint" ]]; then
    echo "[$(timestamp)] Selected checkpoint disappeared: $source_checkpoint"
    sleep "$POLL_SECONDS"
    continue
  fi

  size_before="$(stat -c '%s' "$source_checkpoint")"
  mtime_before="$(stat -c '%Y' "$source_checkpoint")"
  sleep "$STABILITY_SECONDS"
  size_after="$(stat -c '%s' "$source_checkpoint")"
  mtime_after="$(stat -c '%Y' "$source_checkpoint")"
  if [[ "$size_before" != "$size_after" || "$mtime_before" != "$mtime_after" || "$size_after" -lt 104857600 ]]; then
    echo "[$(timestamp)] Selected checkpoint is not stable yet; retrying in ${POLL_SECONDS}s."
    sleep "$POLL_SECONDS"
    continue
  fi

  if server_has_gpu_work; then
    echo "[$(timestamp)] GPU became busy during checkpoint validation; retrying in ${POLL_SECONDS}s."
    sleep "$POLL_SECONDS"
    continue
  fi

  mkdir -p "$RUN_ROOT"
  printf '%s\n' "$source_checkpoint" > "$RUN_ROOT/selected_source_checkpoint.txt"
  printf '{\n  "source_checkpoint": "%s",\n  "source_epoch": %s,\n  "source_unseen_mAP": %s,\n  "selected_at": "%s"\n}\n' \
    "$source_checkpoint" "$source_epoch" "$source_unseen" "$(timestamp)" \
    > "$RUN_ROOT/selected_source_checkpoint.json"

  echo "[$(timestamp)] Server is idle. Selected Object-only epoch=$source_epoch unseen=$source_unseen."
  echo "[$(timestamp)] Starting Gate-only training from: $source_checkpoint"
  export SEED CUDA_VISIBLE_DEVICES RUN_ROOT
  export SOURCE_CHECKPOINT="$source_checkpoint"
  export SOURCE_EPOCH="$source_epoch"
  export SOURCE_UNSEEN="$source_unseen"
  exec bash "$TRAIN_SCRIPT"
done
