#!/usr/bin/env bash
# Hyper‑parameter sweep for LoRA rank on 2×H100

set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

# ---------- user‑editable ----------
MODEL="stabilityai/stable-diffusion-xl-base-1.0"
DATA_DIR="/data/itay"           # 15–20 DreamBooth photos
PROMPT="photo of <itay> man"          # your token prompt
RESOLUTION=1024
EPOCHS=600
BATCH=4                             # per‑GPU
VAL_PROMPT="photo of <itay> man in Central Park, cinematic lighting"
SAVE_EVERY=10                       # epochs
# -----------------------------------

RANKS=(8 16 32 64)

for RANK in "${RANKS[@]}"; do
  OUT="outputs_dreambooth/rank_${RANK}"
  mkdir -p "${OUT}"

  accelerate launch diffuser/dreambooth_lora.py \
      --pretrained_model_name_or_path "${MODEL}" \
      --instance_data_dir "${DATA_DIR}" \
      --instance_prompt "${PROMPT}" \
      --validation_prompt "${VAL_PROMPT}" \
      --num_validation_images 4 \
      --resolution "${RESOLUTION}" \
      --train_batch_size "${BATCH}" \
      --num_train_epochs "${EPOCHS}" \
      --checkpointing_epochs "${SAVE_EVERY}" \
      --rank "${RANK}" \
      --output_dir "${OUT}" \
      --mixed_precision bf16 \
      --gradient_checkpointing \
      --enable_xformers_memory_efficient_attention \
      --allow_tf32 \
      --dataloader_num_workers 8 \
      --report_to tensorboard \
      
done
