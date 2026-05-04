#!/usr/bin/env bash
# Reproduce the training run for
# `dongyoonkim/grootn17-lora-so101-eraser-tier1`
# on `dongyoonkim/so101-eraser-90ep-wrist`.
#
# Required env vars:
#   BASE_MODEL  local path to nvidia/GR00T-N1.7-3B
#   DATASET     local path to so101-eraser-90ep-wrist (LeRobot v3.0)
#   OUTPUT      output checkpoint directory
#   MODALITY    path to examples/so101_wrist_only_config.py
#
# Optional:
#   WANDB_PROJECT   if set, --use-wandb is enabled with this project
#
# Hardware: tested on a single NVIDIA RTX 3090 Ti (24 GB).
# Wall time: ~2.3 h for 15 000 steps at the default batch size.

set -euo pipefail

: "${BASE_MODEL:?path to nvidia/GR00T-N1.7-3B}"
: "${DATASET:?path to so101-eraser-90ep-wrist}"
: "${OUTPUT:?output checkpoint directory}"
: "${MODALITY:?path to so101_wrist_only_config.py}"

WANDB_FLAGS=(--no-use-wandb)
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  WANDB_FLAGS=(--use-wandb --wandb-project "${WANDB_PROJECT}")
fi

python launch_finetune_grootn17_lora.py \
  --dataset-path "${DATASET}" \
  --base-model-path "${BASE_MODEL}" \
  --output-dir "${OUTPUT}" \
  --modality-config-path "${MODALITY}" \
  --embodiment-tag NEW_EMBODIMENT \
  --max-steps 15000 \
  --save-steps 3000 \
  --global-batch-size 16 \
  --learning-rate 1e-4 \
  --warmup-ratio 0.05 \
  --lora_rank 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --lora_include_mlp \
  --lora_modules_to_save_action_head \
  --no-tune_diffusion_model \
  "${WANDB_FLAGS[@]}"
