#!/usr/bin/env bash
set -euo pipefail

# 07: Rank-controlled version of the paper V5 formula. Text Adapter is off.
export EXPERIMENT_ID="07"
export EXPERIMENT_NAME="uo_paper_gate_rank32"
export EXPERIMENT_DESCRIPTION="UO paper-formula low-rank SceneGate rank=32"
export ZS_TYPE="unseen_object"
export N_CTX="36"
export USE_CSC="1"
export ADAPTER_POS="all"
export USE_TEXT_ADAPTER="0"
export USE_SCENE_GATE="1"
export SCENE_GATE_VERSION="v5_lowrank"
export SCENE_GATE_RANK="32"
export SCENE_GATE_ALPHA="0.1"
export SCENE_GATE_START_EPOCH="1"
export LR_SCENE_GATE="1e-3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_common.sh"

