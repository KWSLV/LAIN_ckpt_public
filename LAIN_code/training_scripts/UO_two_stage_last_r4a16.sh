#!/usr/bin/env bash
set -euo pipefail

# Two-stage ablation with the visual LAIN Adapter only in ViT block 12.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec bash "$ROOT/scripts/training/UO_two_stage_r4a16_common.sh" last
