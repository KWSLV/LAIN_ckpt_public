#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/UO_scene_text_obj_joint_diag}" \
bash scripts/training/UO.sh
