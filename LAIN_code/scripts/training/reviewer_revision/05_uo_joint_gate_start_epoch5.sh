#!/usr/bin/env bash
set -euo pipefail

# 05: Let the Text Adapter establish alignment before enabling SceneGate.
export EXPERIMENT_ID="05"
export EXPERIMENT_NAME="uo_joint_gate_start5"
export EXPERIMENT_DESCRIPTION="UO Adapter+paper-SceneGate, Gate starts at epoch 5"
export ZS_TYPE="unseen_object"
export N_CTX="36"
export USE_CSC="1"
export ADAPTER_POS="last"
export USE_TEXT_ADAPTER="1"
export LORA_RANK="4"
export USE_SCENE_GATE="1"
export SCENE_GATE_VERSION="v5_legacy"
export SCENE_GATE_ALPHA="0.1"
export SCENE_GATE_START_EPOCH="5"
export LR_SCENE_GATE="1e-3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_common.sh"

