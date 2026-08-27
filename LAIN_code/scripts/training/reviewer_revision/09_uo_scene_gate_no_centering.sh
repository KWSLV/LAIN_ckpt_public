#!/usr/bin/env bash
set -euo pipefail

# 09: Strict same-code centering-off control for the original paper V5 Gate.
export EXPERIMENT_ID="09"
export EXPERIMENT_NAME="uo_paper_gate_no_centering"
export EXPERIMENT_DESCRIPTION="UO paper SceneGate with pair centering disabled"
export ZS_TYPE="unseen_object"
export N_CTX="36"
export USE_CSC="1"
export ADAPTER_POS="all"
export USE_TEXT_ADAPTER="0"
export USE_SCENE_GATE="1"
export SCENE_GATE_VERSION="v5_legacy"
export SCENE_GATE_ALPHA="0.1"
export SCENE_GATE_START_EPOCH="1"
export SCENE_GATE_DISABLE_CENTERING="1"
export LR_SCENE_GATE="1e-3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_common.sh"

