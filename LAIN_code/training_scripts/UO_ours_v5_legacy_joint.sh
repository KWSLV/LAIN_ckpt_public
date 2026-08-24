#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

# =============================================================================
# Paths and experiment identity
# =============================================================================
DETR_PRETRAINED="${DETR_PRETRAINED:-/root/Lain/checkpoints/detr-r50-hicodet.pth}"
CLIP_PRETRAINED="${CLIP_PRETRAINED:-/root/Lain/checkpoints/ViT-B-16.pt}"
DATA_ROOT="${DATA_ROOT:-./hicodet}"
HICO_IMAGE_ROOT="${HICO_IMAGE_ROOT:-/root/Lain/hico_20160224_det}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/Lain/UO_ours_v5_legacy_joint}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/checkpoints}"
WANDB_DIR="${WANDB_DIR:-$RUN_ROOT/wandb}"
RESUME="${RESUME:-}"

# =============================================================================
# Reproducibility, runtime, and data loading
# Original joint V5 used seed=66, batch=8, workers=4, AMP off.
# AMP/TF32 can improve speed but may produce slightly different numerics.
# =============================================================================
SEED="${SEED:-66}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
PRINT_INTERVAL="${PRINT_INTERVAL:-100}"
USE_AMP="${USE_AMP:-0}"
USE_FAST_CUDA="${USE_FAST_CUDA:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# =============================================================================
# Optimizer and schedule
# LR_HEAD: LAIN projection heads; LR_VIT: CLIP visual/prompt parameters.
# Adapter and Gate learning rates use independent optimizer groups.
# =============================================================================
LR_HEAD="${LR_HEAD:-1e-3}"
LR_VIT="${LR_VIT:-1e-3}"
LR_TEXT_ADAPTER="${LR_TEXT_ADAPTER:-5e-4}"
LR_OBJ_COND_ADAPTER="${LR_OBJ_COND_ADAPTER:-5e-4}"
LR_SCENE_GATE="${LR_SCENE_GATE:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
LR_DROP="${LR_DROP:-10}"
CLIP_MAX_NORM="${CLIP_MAX_NORM:-0.1}"

# =============================================================================
# Frozen DETR architecture and proposal settings
# These values must match the pretrained DETR checkpoint.
# =============================================================================
BACKBONE="${BACKBONE:-resnet50}"
POSITION_EMBEDDING="${POSITION_EMBEDDING:-sine}"
REPR_DIM="${REPR_DIM:-512}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
ENC_LAYERS="${ENC_LAYERS:-6}"
DEC_LAYERS="${DEC_LAYERS:-6}"
DIM_FEEDFORWARD="${DIM_FEEDFORWARD:-2048}"
TRANSFORMER_DROPOUT="${TRANSFORMER_DROPOUT:-0.1}"
NHEADS="${NHEADS:-8}"
NUM_QUERIES="${NUM_QUERIES:-100}"
BOX_SCORE_THRESH="${BOX_SCORE_THRESH:-0.2}"
FG_IOU_THRESH="${FG_IOU_THRESH:-0.5}"
MIN_INSTANCES="${MIN_INSTANCES:-3}"
MAX_INSTANCES="${MAX_INSTANCES:-15}"

# =============================================================================
# HOI focal loss and inference prior
# ALPHA/GAMMA control focal loss; HYPER_LAMBDA sharpens detector priors at test.
# =============================================================================
FOCAL_ALPHA="${FOCAL_ALPHA:-0.5}"
FOCAL_GAMMA="${FOCAL_GAMMA:-0.2}"
HYPER_LAMBDA="${HYPER_LAMBDA:-2.8}"

# =============================================================================
# Original LAIN modules
# ADAPTER_POS=all is the original LAIN default: visual adapters in all 12
# ViT-B/16 visual transformer layers.
# =============================================================================
N_CTX="${N_CTX:-36}"
CTX_INIT="${CTX_INIT:-}"
CLASS_TOKEN_POSITION="${CLASS_TOKEN_POSITION:-end}"
ADAPT_DIM="${ADAPT_DIM:-32}"
ADAPTER_NUM_LAYERS="${ADAPTER_NUM_LAYERS:-1}"
ADAPTER_ALPHA="${ADAPTER_ALPHA:-1.0}"
ADAPTER_POS="${ADAPTER_POS:-all}"
ADAPTER_SCALAR="${ADAPTER_SCALAR:-learnable_scalar}"
FEAT_MASK_TYPE="${FEAT_MASK_TYPE:-0}"
REPEAT_FACTOR_SAMPLING="${REPEAT_FACTOR_SAMPLING:-false}"

# =============================================================================
# Added adapter module from lain_adapter
# Text Adapter adapts CLIP text prototypes. ObjCond Adapter conditions those
# prototypes on the detected object class.
# =============================================================================
TEXT_ADAPTER_DIM="${TEXT_ADAPTER_DIM:-64}"
LORA_RANK="${LORA_RANK:-2}"
LORA_ALPHA="${LORA_ALPHA:-8}"
ADAPTER_RESIDUAL_SCALE="${ADAPTER_RESIDUAL_SCALE:-0.05}"
ADAPTER_DROPOUT="${ADAPTER_DROPOUT:-0.1}"
OBJ_COND_RANK="${OBJ_COND_RANK:-4}"

# =============================================================================
# Original collapsing SceneGate V5 from Scenegateresult-master
# Formula:
#   u=tanh(MLP([LN(detach(ho)), LN(detach(cls))]))
#   u_rel=u-mean_pair(u)
#   fused=ho+SCENE_GATE_ALPHA*u_rel*detach(cls)
# The final MLP layer is always zero-initialized in v5_legacy.
# =============================================================================
SCENE_GATE_HIDDEN_DIM="${SCENE_GATE_HIDDEN_DIM:-128}"
SCENE_GATE_DROPOUT="${SCENE_GATE_DROPOUT:-0.1}"
SCENE_GATE_ALPHA="${SCENE_GATE_ALPHA:-0.1}"
SCENE_GATE_START_EPOCH="${SCENE_GATE_START_EPOCH:-1}"

# Diagnostics do not change training weights, but they increase evaluation cost.
# Original joint diagnostics used both values=1. Set COMPARE_GATE_OFF=0 to keep
# one evaluation per epoch while still collecting collapse statistics.
GATE_DIAGNOSTICS="${GATE_DIAGNOSTICS:-1}"
COMPARE_GATE_OFF="${COMPARE_GATE_OFF:-1}"
HOI_ERROR_ANALYSIS="${HOI_ERROR_ANALYSIS:-0}"

PORT="${PORT:-$((RANDOM % 5000 + 5000))}"
RDZV_ID="${RDZV_ID:-$((RANDOM % 5000 + 5000))}"

for required_file in "$DETR_PRETRAINED" "$CLIP_PRETRAINED"; do
  if [[ ! -f "$required_file" ]]; then
    echo "[ERROR] Required pretrained model not found: $required_file" >&2
    exit 1
  fi
done
if [[ ! -d "$HICO_IMAGE_ROOT" ]]; then
  echo "[ERROR] HICO image root not found: $HICO_IMAGE_ROOT" >&2
  exit 1
fi
if [[ -n "$RESUME" && ! -f "$RESUME" ]]; then
  echo "[ERROR] Resume checkpoint not found: $RESUME" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$WANDB_DIR"
if [[ -z "$RESUME" ]] && compgen -G "$OUTPUT_DIR/ckpt_*.pt" >/dev/null; then
  echo "[ERROR] Refusing to overwrite an existing run: $OUTPUT_DIR" >&2
  echo "[ERROR] Change RUN_ROOT for a new from-scratch experiment." >&2
  exit 1
fi

MAIN_ARGS=(
  main.py
  --pretrained "$DETR_PRETRAINED"
  --clip_dir_vit "$CLIP_PRETRAINED"
  --output-dir "$OUTPUT_DIR"
  --data-root "$DATA_ROOT"
  --hico-image-root "$HICO_IMAGE_ROOT"
  --dataset hicodet
  --partitions train2015 test2015
  --zs
  --zs_type unseen_object
  --num_classes 117
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --test-batch-size "$TEST_BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --prefetch-factor "$PREFETCH_FACTOR"
  --seed "$SEED"
  --device cuda
  --print-interval "$PRINT_INTERVAL"
  --backbone "$BACKBONE"
  --position-embedding "$POSITION_EMBEDDING"
  --repr-dim "$REPR_DIM"
  --hidden-dim "$HIDDEN_DIM"
  --enc-layers "$ENC_LAYERS"
  --dec-layers "$DEC_LAYERS"
  --dim-feedforward "$DIM_FEEDFORWARD"
  --dropout "$TRANSFORMER_DROPOUT"
  --nheads "$NHEADS"
  --num-queries "$NUM_QUERIES"
  --box-score-thresh "$BOX_SCORE_THRESH"
  --fg-iou-thresh "$FG_IOU_THRESH"
  --min-instances "$MIN_INSTANCES"
  --max-instances "$MAX_INSTANCES"
  --weight-decay "$WEIGHT_DECAY"
  --lr-drop "$LR_DROP"
  --clip-max-norm "$CLIP_MAX_NORM"
  --lr-head "$LR_HEAD"
  --lr-vit "$LR_VIT"
  --lr-text-adapter "$LR_TEXT_ADAPTER"
  --lr-obj-cond-adapter "$LR_OBJ_COND_ADAPTER"
  --lr-scene-gate "$LR_SCENE_GATE"
  --alpha "$FOCAL_ALPHA"
  --gamma "$FOCAL_GAMMA"
  --hyper_lambda "$HYPER_LAMBDA"
  --feat_mask_type "$FEAT_MASK_TYPE"
  --repeat_factor_sampling "$REPEAT_FACTOR_SAMPLING"
  --use_hotoken
  --use_prompt
  --use_exp
  --CSC
  --N_CTX "$N_CTX"
  --CTX_INIT "$CTX_INIT"
  --CLASS_TOKEN_POSITION "$CLASS_TOKEN_POSITION"
  --use_insadapter
  --use_prior
  --adapt_dim "$ADAPT_DIM"
  --adapter_num_layers "$ADAPTER_NUM_LAYERS"
  --adapter_alpha "$ADAPTER_ALPHA"
  --adapter_pos "$ADAPTER_POS"
  --adapter_scalar "$ADAPTER_SCALAR"
  --use_text_adapter
  --text_adapter_dim "$TEXT_ADAPTER_DIM"
  --lora_rank "$LORA_RANK"
  --lora_alpha "$LORA_ALPHA"
  --adapter_residual_scale "$ADAPTER_RESIDUAL_SCALE"
  --adapter_dropout "$ADAPTER_DROPOUT"
  --use_obj_cond_adapter
  --obj_cond_rank "$OBJ_COND_RANK"
  --use_scene_gate
  --scene_gate_type pair
  --scene_gate_version v5_legacy
  --scene_gate_hidden_dim "$SCENE_GATE_HIDDEN_DIM"
  --scene_gate_dropout "$SCENE_GATE_DROPOUT"
  --scene_gate_alpha "$SCENE_GATE_ALPHA"
  --scene_gate_start_epoch "$SCENE_GATE_START_EPOCH"
  --scene-gate-diagnostics
  --scene-gate-compare-off
)

if [[ -n "$RESUME" ]]; then
  MAIN_ARGS+=(--resume "$RESUME")
fi
if [[ "$USE_AMP" == "1" ]]; then
  MAIN_ARGS+=(--amp)
fi
if [[ "$USE_FAST_CUDA" == "1" ]]; then
  MAIN_ARGS+=(--fast-cuda)
fi
if [[ "$GATE_DIAGNOSTICS" == "1" ]]; then
  MAIN_ARGS+=(--scene-gate-diagnostics)
fi
if [[ "$COMPARE_GATE_OFF" == "1" ]]; then
  MAIN_ARGS+=(--scene-gate-compare-off)
fi
if [[ "$HOI_ERROR_ANALYSIS" == "1" ]]; then
  MAIN_ARGS+=(--hoi-error-analysis)
fi

echo "[INFO] SceneGate implementation: v5_legacy (original collapsing V5)"
echo "[INFO] Joint training starts at epoch: $SCENE_GATE_START_EPOCH"
echo "[INFO] Output directory: $OUTPUT_DIR"

WANDB_DIR="$WANDB_DIR" WANDB__SERVICE_WAIT=300 \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
torchrun \
  --rdzv_id "$RDZV_ID" \
  --rdzv_backend c10d \
  --nproc_per_node 1 \
  --rdzv_endpoint "127.0.0.1:$PORT" \
  "${MAIN_ARGS[@]}"
