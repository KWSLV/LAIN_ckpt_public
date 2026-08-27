#!/usr/bin/env bash
set -euo pipefail

# 01: Test whether the integrated model is harmed by an overly large Gate LR.
export EXPERIMENT_ID="01"
export EXPERIMENT_NAME="uo_joint_gate_lr1e4"
export EXPERIMENT_DESCRIPTION="UO Adapter+paper-SceneGate, gate LR=1e-4"
export ZS_TYPE="unseen_object"
export N_CTX="36"
export USE_CSC="1"
export ADAPTER_POS="last"
export USE_TEXT_ADAPTER="1"
export LORA_RANK="4"
export USE_SCENE_GATE="1"
export SCENE_GATE_VERSION="v5_legacy"
export SCENE_GATE_ALPHA="0.1"
export SCENE_GATE_START_EPOCH="1"
export LR_SCENE_GATE="1e-4"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_common.sh"

