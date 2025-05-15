CUDA_VISIBLE_DEVICES=4 python infer/db_infer.py \
  --pretrained_model="stabilityai/stable-diffusion-xl-base-1.0" \
  --db_lora_root="output_dreambooth" \
  --ranks 64 \
  --prompts_module="itay_prompt" \
  --output_dir="results/db_lora"
