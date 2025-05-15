#!/usr/bin/env python3
# coding=utf-8
"""
LoRA inference script for Stable Diffusion XL.
Generates images using LoRA adapters saved across hyperparameter search ranks and epochs.
"""
import argparse
import importlib
import os
import sys
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

def main():
    parser = argparse.ArgumentParser(description="LoRA Inference for SDXL with hyperparameter search")
    parser.add_argument(
        "--pretrained_model", type=str, required=True,
        help="Path or HF ID of the base SDXL model"
    )
    parser.add_argument(
        "--lora_root", type=str, default="output_lora",
        help="Root directory containing rank_{r}/epoch_{e} subfolders"
    )
    parser.add_argument(
        "--ranks", type=int, nargs="+", default=[4],
        help="List of LoRA rank values to process (e.g., 1 2 4 8)"
    )
    parser.add_argument(
        "--prompts_module", type=str, default="itay_prompt",
        help="Module name containing prompts (itay_prompt.py)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="generated_images/lora",
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

    # Add CWD to path for prompts
    sys.path.append(os.getcwd())
    prompt_module = importlib.import_module(args.prompts_module)
    positive_prompts = [getattr(prompt_module, a) for a in dir(prompt_module) if a.startswith("positive_prompt_")]
    negative_prompt = getattr(prompt_module, "negative_prompt", "")

    for r in args.ranks:
        rank_dir = os.path.join(args.lora_root, f"rank{r}")
        if not os.path.isdir(rank_dir):
            print(f"[Warning] Rank folder missing: {rank_dir}")
            continue
        # Collect epoch subdirs
        epochs = sorted(
            [d for d in os.listdir(rank_dir) if d.startswith("epoch_")],
            key=lambda x: int(x.split('_')[1])
        )
        if not epochs:
            print(f"[Warning] No epochs in {rank_dir}")
            continue

        for epoch in epochs[10:]:
            epoch_path = os.path.join(rank_dir, epoch)
            if not os.path.isdir(epoch_path):
                continue

            # Load base pipeline
            pipe = StableDiffusionXLPipeline.from_pretrained(
                args.pretrained_model,
                safety_checker=None,
                torch_dtype=torch.float16
            )
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe = pipe.to("cuda")

            # Load LoRA weights
            pipe.load_lora_weights(epoch_path)
            pipe.set_progress_bar_config(disable=True)

            out_base = os.path.join(args.output_dir, f"rank{r}", epoch)
            os.makedirs(out_base, exist_ok=True)

            # Generate images
            for idx, prompt in enumerate(positive_prompts):
                prompt_dir = os.path.join(out_base, f"prompt_{idx}")
                os.makedirs(prompt_dir, exist_ok=True)
                for j in range(args.num_images):
                    seed = args.seed_base + j
                    generator = torch.Generator(device=pipe.device).manual_seed(seed)
                    result = pipe(
                        prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=args.num_inference_steps,
                        generator=generator
                    )
                    img = result.images[0]
                    save_path = os.path.join(prompt_dir, f"image_{j}.png")
                    img.save(save_path)
                    print(f"Saved: {save_path}")

            # Cleanup
            del pipe
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
