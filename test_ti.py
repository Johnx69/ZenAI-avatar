from diffusers import StableDiffusionXLPipeline
import torch

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    use_safetensors=True,
).to("cuda")

prompt = "(wide shot) analog modelshoot 8k close-up LinkedIn profile picture of <itay>, frontal face, looking at camera, professional suit, upper body, blurred glass building background, crisp details, neutral expression, photorealistic, high-resolution, sharp focus on eyes, ambient cinematic lighting, hyperrealistic, masterpiece, best quality, ultra-detailed"
pipe.load_textual_inversion("/egr/research-actionlab/anhdao/ZenAI/avatar/output_textual_inversion/nv_1/epoch_100")
# SDXL requires both `prompt` and `negative_prompt`, and works best with additional keyword arguments
image = pipe(
    prompt=prompt,
    negative_prompt="asian, back view, multiple heads, 2 heads, elongated body, double image, 2 faces, multiple people, double head, , (nsfw), nsfw, nsfw, nsfw, nude, nude, nude, porn, porn, porn, naked, naked, nude, porn, frilly, frilled, lacy, ruffled, victorian, (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
    num_inference_steps=50,
    guidance_scale=7.5,
    output_type="pil",
    added_cond_kwargs={"text_embeds": None, "time_ids": None}  # <-- Prevents NoneType error
).images[0]

image.save("itay.png")
