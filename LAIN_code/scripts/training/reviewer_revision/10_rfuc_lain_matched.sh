#!/usr/bin/env bash
set -euo pipefail

# 10: Matched LAIN baseline for Rare-First Unseen Combination.
export EXPERIMENT_ID="10"
export EXPERIMENT_NAME="rfuc_lain_matched"
export EXPERIMENT_DESCRIPTION="RF-UC matched LAIN baseline"
export ZS_TYPE="rare_first"
export N_CTX="24"
export USE_CSC="1"
export ADAPTER_POS="all"
export USE_TEXT_ADAPTER="0"
export USE_SCENE_GATE="0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_common.sh"

