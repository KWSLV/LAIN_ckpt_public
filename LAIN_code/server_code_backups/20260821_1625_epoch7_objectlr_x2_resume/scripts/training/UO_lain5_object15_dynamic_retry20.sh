#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export PATH="/root/miniconda3/bin:$PATH"
source scripts/server_assets.sh
prepare_lain_server_assets

SCRIPT_NAME="$(basename "$0" .sh)"
SEED="${SEED:-66}"
EPOCHS=20
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/Lain/${SCRIPT_NAME}_seed${SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/checkpoints}"
WANDB_DIR="${WANDB_DIR:-$RUN_ROOT/wandb}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PORT="${PORT:-$((RANDOM % 5000 + 15000))}"
RDZV_ID="${RDZV_ID:-$((RANDOM % 5000 + 15000))}"

if compgen -G "$OUTPUT_DIR/ckpt_*.pt" >/dev/null \
  || [[ -f "$OUTPUT_DIR/staged_retry_attempts.jsonl" ]]; then
  echo "[ERROR] Refusing to mix a new staged run with existing output: $OUTPUT_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$WANDB_DIR" "$RUN_ROOT/logs"

exec 9>"/tmp/lain_cuda_${CUDA_VISIBLE_DEVICES}.lock"
if ! flock -n 9; then
  echo "[ERROR] Another managed LAIN job is using CUDA device $CUDA_VISIBLE_DEVICES." >&2
  exit 1
fi

echo "[$SCRIPT_NAME] Fresh UO run from DETR + CLIP pretrained weights"
echo "[$SCRIPT_NAME] Logical epochs 1-5: LAIN only; Object forward BYPASS"
echo "[$SCRIPT_NAME] Logical epochs 6-20: LAIN + Object"
echo "[$SCRIPT_NAME] Text=OFF, Gate=OFF for the full run"
echo "[$SCRIPT_NAME] Initial LR: head=5e-4 vit=2e-4 object=2e-4"
echo "[$SCRIPT_NAME] StepLR/fixed scheduler: DISABLED (lr_drop=0 is inert)"
echo "[$SCRIPT_NAME] First accepted Unseen >=37.5: one-time initial LR / 10"
echo "[$SCRIPT_NAME] Rollback limit: pre-trigger=0.15, post-trigger=0.05; current LR x0.5 per retry"
echo "[$SCRIPT_NAME] W&B run name: $SCRIPT_NAME"
echo "[$SCRIPT_NAME] Output: $OUTPUT_DIR"

WANDB_DIR="$WANDB_DIR" \
WANDB_NAME="$SCRIPT_NAME" \
WANDB_RUN_GROUP="LAIN5_Object15_dynamic_retry20" \
WANDB__SERVICE_WAIT=300 \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
torchrun \
  --rdzv_id "$RDZV_ID" \
  --rdzv_backend c10d \
  --nproc_per_node 1 \
  --rdzv_endpoint "127.0.0.1:$PORT" \
  main.py \
  --pretrained "$DETR_PRETRAINED" \
  --clip_dir_vit "$CLIP_PRETRAINED" \
  --output-dir "$OUTPUT_DIR" \
  --hico-image-root "$HICO_IMAGE_ROOT" \
  --dataset hicodet \
  --partitions train2015 test2015 \
  --zs --zs_type unseen_object \
  --num_classes 117 \
  --epochs "$EPOCHS" \
  --batch-size 8 \
  --test-batch-size 8 \
  --num-workers 8 \
  --prefetch-factor 4 \
  --seed "$SEED" \
  --weight-decay 1e-4 \
  --lr-drop 0 \
  --disable-lr-scheduler \
  --clip-max-norm 0.1 \
  --lr-head 5e-4 \
  --lr-vit 2e-4 \
  --lr-obj-cond-adapter 2e-4 \
  --lain-object-staged-retry-policy \
  --staged-object-start-epoch 6 \
  --staged-lr-trigger-unseen 37.5 \
  --staged-pre-trigger-max-drop 0.15 \
  --staged-post-trigger-max-drop 0.05 \
  --staged-rollback-lr-factor 0.5 \
  --alpha 0.5 \
  --gamma 0.2 \
  --hyper_lambda 2.8 \
  --box-score-thresh 0.2 \
  --fg-iou-thresh 0.5 \
  --min-instances 3 \
  --max-instances 15 \
  --use_hotoken \
  --use_prompt \
  --use_exp \
  --CSC \
  --N_CTX 36 \
  --use_insadapter \
  --adapt_dim 32 \
  --use_prior \
  --adapter_alpha 1.0 \
  --adapter_num_layers 1 \
  --adapter_pos all \
  --use_obj_cond_adapter \
  --obj_cond_rank 4 \
  --adapter-contribution-diagnostics \
  --amp \
  --fast-cuda \
  --keep-last-checkpoints 0 \
  --keep-best-unseen-checkpoint \
  --print-interval 500

final_checkpoint="$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name 'ckpt_*_20.pt' -print -quit)"
if [[ -z "$final_checkpoint" ]]; then
  echo "[ERROR] Training exited without an accepted logical-epoch-20 checkpoint." >&2
  exit 1
fi

python - "$OUTPUT_DIR" <<'PY'
import json
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
rows = json.loads((output / 'scene_gate_diagnostics.json').read_text(encoding='utf-8'))
if [row['epoch'] for row in rows] != list(range(1, 21)):
    raise SystemExit('[ERROR] Accepted diagnostics are not exactly logical epochs 1..20.')
for row in rows:
    staged = row['staged_training_metrics']
    expected_object = row['epoch'] >= 6
    if bool(staged['object_forward_active']) != expected_object:
        raise SystemExit(f"[ERROR] Object stage mismatch at epoch {row['epoch']}.")
attempts = [
    json.loads(line)
    for line in (output / 'staged_retry_attempts.jsonl').read_text(encoding='utf-8').splitlines()
    if line.strip()
]
accepted = [row for row in attempts if row['accepted']]
if [row['logical_epoch'] for row in accepted] != list(range(1, 21)):
    raise SystemExit('[ERROR] Attempt log does not contain exactly 20 accepted logical epochs.')
if sum(bool(row['one_time_lr_drop_applied']) for row in attempts) > 1:
    raise SystemExit('[ERROR] The Unseen>=37.5 LR trigger executed more than once.')
print('[VERIFY] 20 accepted logical epochs; staged Object state and one-time LR trigger are valid.')
PY

echo "[$SCRIPT_NAME] Complete: $final_checkpoint"
echo "[$SCRIPT_NAME] Best: $OUTPUT_DIR/best_unseen.pt"
echo "[$SCRIPT_NAME] Accepted diagnostics: $OUTPUT_DIR/scene_gate_diagnostics.json"
echo "[$SCRIPT_NAME] All attempts: $OUTPUT_DIR/staged_retry_attempts.jsonl"
echo "[$SCRIPT_NAME] Failed retries: $OUTPUT_DIR/staged_retry_failures.jsonl"

