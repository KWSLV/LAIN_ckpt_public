#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python -m pip install --upgrade pip setuptools wheel

if ! python -c 'import torch, torchvision' >/dev/null 2>&1; then
  python -m pip install \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128
fi

python -m pip install -r requirements-server.txt
python -m pip install --no-deps -e ./pocket

python - <<'PY'
import cv2
import ftfy
import numpy
import scipy
import torch
import torchvision
import wandb
from pocket.core import DistributedLearningEngine

print("[OK] Python dependencies are installed")
print(f"[OK] torch={torch.__version__}, torchvision={torchvision.__version__}")
print(f"[OK] CUDA available={torch.cuda.is_available()}, CUDA={torch.version.cuda}")
PY
