#!/usr/bin/env bash

# Shared server-side asset discovery for training and evaluation scripts.

LAIN_PRETRAINED_ROOT="${LAIN_PRETRAINED_ROOT:-/root/Lain/checkpoints}"
HICO_IMAGE_ROOT="${HICO_IMAGE_ROOT:-/root/Lain/hico_20160224_det}"

_find_pretrained_file() {
  local current="$1"
  local filename="$2"

  if [[ -n "$current" && -f "$current" ]]; then
    printf '%s\n' "$current"
    return 0
  fi

  find "$LAIN_PRETRAINED_ROOT" -type f -name "$filename" -print -quit 2>/dev/null || true
}

_valid_hico_root() {
  [[ -d "$1/images/train2015" && -d "$1/images/test2015" ]]
}

prepare_lain_server_assets() {
  DETR_PRETRAINED="$(_find_pretrained_file \
    "${DETR_PRETRAINED:-}" "detr-r50-hicodet.pth")"
  CLIP_PRETRAINED="$(_find_pretrained_file \
    "${CLIP_PRETRAINED:-}" "ViT-B-16.pt")"

  if [[ -z "$DETR_PRETRAINED" ]]; then
    echo "[ERROR] Cannot find detr-r50-hicodet.pth under $LAIN_PRETRAINED_ROOT" >&2
    return 1
  fi
  if [[ -z "$CLIP_PRETRAINED" ]]; then
    echo "[ERROR] Cannot find ViT-B-16.pt under $LAIN_PRETRAINED_ROOT" >&2
    return 1
  fi

  if ! _valid_hico_root "$HICO_IMAGE_ROOT"; then
    echo "[ERROR] Invalid or unextracted HICO root: $HICO_IMAGE_ROOT" >&2
    echo "[ERROR] Expected images/train2015 and images/test2015 below it." >&2
    return 1
  fi

  export DETR_PRETRAINED CLIP_PRETRAINED HICO_IMAGE_ROOT
  echo "[INFO] DETR weights: $DETR_PRETRAINED"
  echo "[INFO] CLIP weights: $CLIP_PRETRAINED"
  echo "[INFO] HICO images: $HICO_IMAGE_ROOT"
}
