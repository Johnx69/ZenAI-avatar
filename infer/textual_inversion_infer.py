import os
import torch
from diffusers import StableDiffusionXLPipeline
import importlib.util
import argparse

# --- Load prompts from full path ---
prompt_path = "/egr/research-actionlab/anhdao/ZenAI/avatar/itay_prompt.py"
spec = importlib.util.spec_from_file_location("itay_prompt", prompt_path)
prompts_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prompts_module)

positive_prompts = [getattr(prompts_module, f"positive_prompt_{i}") for i in range(10)]
negative_prompt = prompts_module.negative_prompt.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nv", type=int, required=True, help="Number of vectors (e.g., 1, 2, 4, 8, 16)")
    args = parser.parse_args()
    nv = args.nv

    base_model = "stabilityai/stable-diffusion-xl-base-1.0"
    ti_root = f"/egr/research-actionlab/anhdao/ZenAI/avatar/output_textual_inversion/nv_{nv}"

    # Load base pipeline once
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")

    for epoch_folder in sorted(os.listdir(ti_root)):
        epoch_path = os.path.join(ti_root, epoch_folder)
        if not os.path.isdir(epoch_path): continue
        if not os.path.exists(os.path.join(epoch_path, "learned_embeds.safetensors")):
            print(f"[⚠️] Skipping {epoch_path}: no learned_embeds.bin found")
            continue

        # --- Remove <itay> if already in tokenizer ---
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder

        if "<itay>" in tokenizer.get_vocab():
            print(f"🔄 Removing <itay> from tokenizer and text encoder before loading from {epoch_path}")
            token_id = tokenizer.convert_tokens_to_ids("<itay>")
            tokenizer.vocab.pop("<itay>", None)
            tokenizer.token_to_id.pop("<itay>", None)
            tokenizer.encoder.pop("<itay>", None)
            tokenizer.decoder.pop(token_id, None)
            with torch.no_grad():
                text_encoder.get_input_embeddings().weight[token_id] = torch.zeros_like(
                    text_encoder.get_input_embeddings().weight[token_id]
                )

        # --- Load learned embedding ---
        try:
            pipe.load_textual_inversion(epoch_path)
        except Exception as e:
            print(f"[❌] Failed to load from {epoch_path}: {e}")
            continue

        # --- Inference ---
        for i, prompt in enumerate(positive_prompts):
            for j in range(2):
                seed = 6969 + j
                generator = torch.Generator(device="cuda").manual_seed(seed)

                try:
                    image = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=50,
                        guidance_scale=7.5,
                        generator=generator,
                        output_type="pil",
                        added_cond_kwargs={"text_embeds": None, "time_ids": None}
                    ).images[0]

                    save_dir = f"/egr/research-actionlab/anhdao/ZenAI/avatar/results/ti/nv_{nv}/{epoch_folder}/prompt_{i}"
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"image_{j}.png")
                    image.save(save_path)
                    print(f"[✅] Saved: {save_path}")

                except Exception as gen_err:
                    print(f"[❌] Generation failed: nv={nv}, epoch={epoch_folder}, prompt={i}, seed={seed}: {gen_err}")
        
        del pipe
        torch.cuda.empty_cache()
        # ⚠️ Re-initialize pipeline now to get a fresh tokenizer/text_encoder
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to("cuda")

if __name__ == "__main__":
    main()
