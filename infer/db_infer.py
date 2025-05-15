#!/usr/bin/env python3
# coding=utf-8
"""
DreamBooth LoRA inference script (single GPU) for SDXL with hyperparameter-rank support.
Iterates through rank_{r}/checkpoint-* folders for each specified LoRA rank,
loads each adapter, and generates images for the defined prompts.
"""
import argparse
import importlib
import os
import sys
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler


def main():
    parser = argparse.ArgumentParser(
        description="DreamBooth LoRA inference for SDXL (single GPU, hyperparam ranks)"
    )
    parser.add_argument(
        "--pretrained_model", type=str, required=True,
        help="Path or HF ID of the base SDXL model"
    )
    parser.add_argument(
        "--db_lora_root", type=str, default="lora-dreambooth-model",
        help="Root directory containing rank_{r}/checkpoint-* subfolders"
    )
    parser.add_argument(
        "--ranks", type=int, nargs="+", default=[4,8,16,32,64],
        help="List of LoRA rank values to process (e.g., 4 8 16 32 64)"
    )
    parser.add_argument(
        "--prompts_module", type=str, default="itay_prompt",
        help="Python module name containing prompts (itay_prompt.py)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="generated_images/db_lora",
        help="Root folder for saving generated images"
    )
    parser.add_argument(
        "--seed_base", type=int, default=6969,
        help="Base seed; images use seed_base + j"
    )
    parser.add_argument(
        "--num_images", type=int, default=2,
        help="Number of images per prompt"
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=50,
        help="Number of inference steps"
    )
    args = parser.parse_args()

    # Import prompts
    sys.path.append(os.getcwd())
    prompt_module = importlib.import_module(args.prompts_module)
    positive_prompts = [
        getattr(prompt_module, a)
        for a in dir(prompt_module)
        if a.startswith("positive_prompt_")
    ]
    negative_prompt = getattr(prompt_module, "negative_prompt", "")

    # Iterate over each LoRA rank
    for r in args.ranks:
        rank_dir = os.path.join(args.db_lora_root, f"rank_{r}")
        if not os.path.isdir(rank_dir):
            print(f"[Warning] Rank folder missing: {rank_dir}")
            continue

        # Iterate through checkpoints within this rank
        for ckpt in sorted(os.listdir(rank_dir)):
            ckpt_path = os.path.join(rank_dir, ckpt)
            if not os.path.isdir(ckpt_path) or not ckpt.startswith("checkpoint-epoch"):
                continue

            # Load pipeline on single GPU
            pipe = StableDiffusionXLPipeline.from_pretrained(
                args.pretrained_model,
                safety_checker=None,
                torch_dtype=torch.float16
            )
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe = pipe.to("cuda")

            # Load the LoRA weights for this checkpoint
            pipe.load_lora_weights(ckpt_path)
            pipe.set_progress_bar_config(disable=True)

            # Prepare output directory
            out_base = os.path.join(args.output_dir, f"rank_{r}", ckpt)
            os.makedirs(out_base, exist_ok=True)

            # Generate images for each prompt
            for idx, prompt in enumerate(positive_prompts):
                prompt_dir = os.path.join(out_base, f"prompt_{idx}")
                os.makedirs(prompt_dir, exist_ok=True)
                for j in range(args.num_images):
                    seed = args.seed_base + j
                    gen = torch.Generator(device=pipe.device).manual_seed(seed)
                    result = pipe(
                        prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=args.num_inference_steps,
                        generator=gen
                    )
                    img = result.images[0]
                    save_path = os.path.join(prompt_dir, f"image_{j}.png")
                    img.save(save_path)
                    print(f"Saved: {save_path}")

            # Clean up GPU memory
            del pipe
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
