#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EVAL_ROOT="${EVAL_ROOT:-/root/autodl-tmp/Lain/UO_checkpoint_module_eval_$RUN_ID}"
mkdir -p "$EVAL_ROOT"

run_eval() {
  local mode="$1"
  local checkpoint="$2"
  local tag="$3"
  local output="$EVAL_ROOT/$tag"
  mkdir -p "$output"
  EVAL_ROOT="$EVAL_ROOT" OUTPUT_DIR="$output" \
    bash scripts/eval/UO_checkpoint_module_eval.sh "$mode" "$checkpoint" "$tag" \
    2>&1 | tee "$output/eval.log"
}

run_eval \
  object_only \
  /root/autodl-tmp/Lain/UO_a4_ep9_objinherit_objectonly_r4_seed66/checkpoints/ckpt_03895_01.pt \
  object_inherit_epoch01

run_eval \
  object_only \
  /root/autodl-tmp/Lain/UO_a4_ep9_objinherit_objectonly_r4_seed66/checkpoints/ckpt_19475_05.pt \
  object_inherit_epoch05

run_eval \
  object_gate \
  /root/autodl-tmp/Lain/UO_objectbest_gateonly_v5h128_a004_seed66/checkpoints/ckpt_15580_04.pt \
  objectbest_gate_epoch04

echo "[UO Batch Eval] complete: $EVAL_ROOT"
