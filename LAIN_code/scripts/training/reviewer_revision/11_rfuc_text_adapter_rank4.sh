#!/usr/bin/env bash
set -euo pipefail

# 11: RF-UC matched LAIN plus the paper Text Adapter rank 4.
export EXPERIMENT_ID="11"
export EXPERIMENT_NAME="rfuc_text_rank4"
export EXPERIMENT_DESCRIPTION="RF-UC LAIN+Text Adapter rank=4"
export ZS_TYPE="rare_first"
export N_CTX="24"
export USE_CSC="1"
export ADAPTER_POS="all"
export USE_TEXT_ADAPTER="1"
export LORA_RANK="4"
export USE_SCENE_GATE="0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_common.sh"

