CUDA_VISIBLE_DEVICES=6 python infer/lora_infer.py \
  --pretrained_model="stabilityai/stable-diffusion-xl-base-1.0" \
  --lora_root="irit_output" \
  --ranks 8 \
  --prompts_module="iris_prompt" \
  --output_dir="results/iris"
