#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/root/autodl-tmp/Lain/UO_scene_text_obj_run/0721checkpoints/ckpt_58425_15.pt}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/Lain/UO_scene_gate_refine_from_ep15}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/checkpoints}"

if [[ ! -f "$BASE_CHECKPOINT" ]]; then
  echo "[ERROR] Initial checkpoint does not exist: $BASE_CHECKPOINT" >&2
  exit 1
fi

if compgen -G "$OUTPUT_DIR/ckpt_*.pt" >/dev/null; then
  echo "[ERROR] Refusing to reuse a non-empty checkpoint directory: $OUTPUT_DIR" >&2
  exit 1
fi

# Delegate to the original refinement script so every training parameter stays
# identical. Only the initial checkpoint and output directory are different.
export RUN_ROOT OUTPUT_DIR
exec bash "$ROOT/scripts/training/UO_scene_gate_refine.sh" "$BASE_CHECKPOINT"
