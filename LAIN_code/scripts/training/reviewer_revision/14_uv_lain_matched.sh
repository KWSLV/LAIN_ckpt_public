#!/usr/bin/env bash
set -euo pipefail

# 14: Matched LAIN baseline for Unseen Verb.
export EXPERIMENT_ID="14"
export EXPERIMENT_NAME="uv_lain_matched"
export EXPERIMENT_DESCRIPTION="UV matched LAIN baseline"
export ZS_TYPE="unseen_verb"
export N_CTX="36"
export USE_CSC="0"
export ADAPTER_POS="all"
export USE_TEXT_ADAPTER="0"
export USE_SCENE_GATE="0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_common.sh"

