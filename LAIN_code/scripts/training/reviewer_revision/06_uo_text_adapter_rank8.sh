#!/usr/bin/env bash
set -euo pipefail

# 06: Extend the existing Text Adapter rank-2/rank-4 ablation to rank 8.
export EXPERIMENT_ID="06"
export EXPERIMENT_NAME="uo_text_rank8"
export EXPERIMENT_DESCRIPTION="UO Text Adapter rank=8, SceneGate off"
export ZS_TYPE="unseen_object"
export N_CTX="36"
export USE_CSC="1"
export ADAPTER_POS="last"
export USE_TEXT_ADAPTER="1"
export LORA_RANK="8"
export USE_SCENE_GATE="0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_common.sh"

