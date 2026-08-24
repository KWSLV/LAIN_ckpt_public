#!/usr/bin/env bash
set -euo pipefail

# Two-stage paper-aligned run with visual LAIN Adapters in all 12 ViT blocks.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec bash "$ROOT/scripts/training/UO_two_stage_r4a16_common.sh" all
