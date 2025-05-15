#!/usr/bin/env bash
set -euo pipefail

################################################################################
# Hyper‑parameter sweep over LoRA rank for SDXL fine‑tuning
#
# • Uses one H100 80 GB ‑‑> batch_size 8 @ 1024² fits comfortably (bf16)
# • 300 epochs, checkpoint every 10 epochs (–save_every_epochs 10)
# • WandB project "sdxl‑lora‑rank‑sweep" – each run is tagged with its RANK
# • Requires: accelerate>=0.25, diffusers>=0.25, peft>=0.10, torch>=2.3, \
#             xformers, bitsandbytes(optional)
################################################################################

export WANDB_PROJECT="sdxl-lora-rank-sweep"
export WANDB__SERVICE_WAIT=300                # give wandb time to finish uploads
export CUDA_VISIBLE_DEVICES=1

# DATA / MODEL paths – edit to match your setup
MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
VAE_NAME="madebyollin/sdxl-vae-fp16-fix"      # numerically stable VAE
DATA_DIR="avatar/data/irit"  

# Training‑time performance flags common to all runs
# --- define once ---
COMMON_ARGS=(
  --pretrained_model_name_or_path "$MODEL_NAME"
  --pretrained_vae_model_name_or_path "$VAE_NAME"
  --train_data_dir "$DATA_DIR"
  --resolution 1024 --random_flip
  --train_batch_size 8
  --num_train_epochs 600
  --save_every_epochs 10
  --checkpointing_steps 0
  --learning_rate 1e-4
  --lr_scheduler cosine_with_restarts --lr_warmup_steps 0
  --mixed_precision bf16
  --enable_xformers_memory_efficient_attention
  --allow_tf32
  --gradient_checkpointing
  --dataloader_num_workers 8
  --report_to wandb
  --seed 42
)


# If you enabled torch.compile in accelerate config, leave it; otherwise uncomment:
# export TORCH_COMPILE=1

RANKS=(8)
for RANK in "${RANKS[@]}"; do
  OUTPUT_DIR="irit_output/rank${RANK}"
  accelerate launch diffuser/sdxl_lora.py \
  "${COMMON_ARGS[@]}" \
  --rank "$RANK" \
  --output_dir "$OUTPUT_DIR" \
  --logging_dir "$OUTPUT_DIR/logs"
done
