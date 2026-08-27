#!/usr/bin/env bash
set -euo pipefail

# 15: UV matched LAIN plus the paper Text Adapter rank 4.
export EXPERIMENT_ID="15"
export EXPERIMENT_NAME="uv_text_rank4"
export EXPERIMENT_DESCRIPTION="UV LAIN+Text Adapter rank=4"
export ZS_TYPE="unseen_verb"
export N_CTX="36"
export USE_CSC="0"
export ADAPTER_POS="all"
export USE_TEXT_ADAPTER="1"
export LORA_RANK="4"
export USE_SCENE_GATE="0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_common.sh"

