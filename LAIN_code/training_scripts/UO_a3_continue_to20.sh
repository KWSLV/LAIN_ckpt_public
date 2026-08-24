#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SEED="${SEED:-66}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/Lain/UO_a3_tuned_text_object_last_r4a16_seed${SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/checkpoints}"

shopt -s nullglob
epoch15=("$OUTPUT_DIR"/ckpt_*_15.pt)
epoch20=("$OUTPUT_DIR"/ckpt_*_20.pt)
shopt -u nullglob

if (( ${#epoch20[@]} > 0 )); then
  echo "[A3 Continue] Already complete: ${epoch20[-1]}"
  exit 0
fi
if (( ${#epoch15[@]} == 0 )); then
  echo "[ERROR] A3 epoch-15 checkpoint not found under $OUTPUT_DIR" >&2
  exit 1
fi

echo "[A3 Continue] Restoring model/optimizer/scheduler/AMP state from ${epoch15[-1]}"
echo "[A3 Continue] Continuing epochs 16-20 in the original output directory."
SEED="$SEED" RUN_ROOT="$RUN_ROOT" OUTPUT_DIR="$OUTPUT_DIR" \
  exec bash "$ROOT/scripts/training/UO_a2_a3_common.sh" a3
