#!/usr/bin/env bash
set -euo pipefail

# 12: Matched LAIN baseline for Non-Rare-First Unseen Combination.
export EXPERIMENT_ID="12"
export EXPERIMENT_NAME="nfuc_lain_matched"
export EXPERIMENT_DESCRIPTION="NF-UC matched LAIN baseline"
export ZS_TYPE="non_rare_first"
export N_CTX="36"
export USE_CSC="1"
export ADAPTER_POS="all"
export USE_TEXT_ADAPTER="0"
export USE_SCENE_GATE="0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_common.sh"

