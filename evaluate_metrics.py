import os
import numpy as np
import pandas as pd
import importlib.util
import torch
from torchmetrics.functional.multimodal import clip_score
from insightface.app import FaceAnalysis
from PIL import Image
from functools import partial
import argparse

# ---- Utility Functions ----

def load_prompts(prompt_file):
    spec = importlib.util.spec_from_file_location("itay_prompt", prompt_file)
    itay_prompt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(itay_prompt)
    return [getattr(itay_prompt, f"positive_prompt_{i}") for i in range(10)]


def init_clip_score_fn():
    return partial(clip_score, model_name_or_path="openai/clip-vit-large-patch14")


def init_face_analyzer():
    fa = FaceAnalysis()
    fa.prepare(ctx_id=0, det_size=(640, 640))
    return fa


def get_arcface_embedding(fa, img_np):
    faces = fa.get(img_np)
    if len(faces) == 0:
        return None
    emb = faces[0].embedding
    return emb / np.linalg.norm(emb)


def compute_clip_score_fn(clip_fn, img_path, prompt):
    img = Image.open(img_path).convert("RGB")
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float().to("cuda")
    return float(clip_fn(img_tensor, [prompt]).item())

# ---- Core Evaluation for a Single Parameter ----
def evaluate_param(method, cfg, param, prompts, clip_fn, fa, train_embs):
    records = []
    root = cfg["root"]
    for epoch in cfg["epochs"]:
        print(f"[{method}] {cfg['param_name']}={param}, Epoch={epoch}")
        clip_scores, arc_scores = [], []
        for p_idx, prompt in enumerate(prompts):
            prompt_num = p_idx + cfg.get("prompt_offset", 0)
            # Determine files to process
            if "seeds" in cfg and "image_indices" not in cfg:
                targets = cfg["seeds"]
                for seed in targets:
                    rel = cfg["pattern"].format(param=param, epoch=epoch, prompt=prompt_num, seed=seed)
                    path = os.path.join(root, rel)
                    if not os.path.exists(path):
                        continue
                    print(f"  Processing: {path}")
                    try:
                        cs = compute_clip_score_fn(clip_fn, path, prompt)
                    except Exception as e:
                        print(f"    [CLIP ERROR] {e}")
                        cs = None
                    emb = get_arcface_embedding(fa, np.array(Image.open(path).convert("RGB")))
                    arc = float(np.max(train_embs.dot(emb))) if emb is not None else None
                    if cs is not None: clip_scores.append(cs)
                    if arc is not None: arc_scores.append(arc)
            else:
                for idx, seed in zip(cfg.get("image_indices", []), cfg.get("seeds", [])):
                    rel = cfg["pattern"].format(param=param, epoch=epoch, prompt=prompt_num, i=idx, seed=seed)
                    path = os.path.join(root, rel)
                    if not os.path.exists(path):
                        continue
                    print(f"  Processing: {path}")
                    try:
                        cs = compute_clip_score_fn(clip_fn, path, prompt)
                    except Exception as e:
                        print(f"    [CLIP ERROR] {e}")
                        cs = None
                    emb = get_arcface_embedding(fa, np.array(Image.open(path).convert("RGB")))
                    arc = float(np.max(train_embs.dot(emb))) if emb is not None else None
                    if cs is not None: clip_scores.append(cs)
                    if arc is not None: arc_scores.append(arc)
        epoch_clip = float(np.mean(clip_scores)) if clip_scores else None
        epoch_arc = float(np.mean(arc_scores)) if arc_scores else None
        print(f"  -> Epoch {epoch}: CLIP={epoch_clip}, ArcFace={epoch_arc}\n")
        records.append({cfg["param_name"]: param, "epoch": epoch,
                        "clip_score": epoch_clip, "arcface_score": epoch_arc})
    return pd.DataFrame(records)

# ---- Main ----
def main():
    parser = argparse.ArgumentParser(description="Evaluate metrics by method and parameter.")
    parser.add_argument("-m", "--methods", nargs='+', choices=["lora","db_lora","ti","itay_loras","cog_sdxl"],
                        required=True, help="Methods to evaluate")
    args = parser.parse_args()

    methods_cfg = {
        "lora": {"root":"avatar/results/lora",
                 "pattern":"rank{param}/epoch_{epoch}/prompt_{prompt}/image_{i}.png",
                 "params":[64], "epochs":list(range(10,601,10)),
                 "param_name":"rank","prompt_offset":0,"image_indices":[0,1],"seeds":[None,None]},
        "db_lora": {"root":"avatar/results/db_lora",
                     "pattern":"rank_{param}/checkpoint-epoch-{epoch}/prompt_{prompt}/image_{i}.png",
                     "params":[64], "epochs":list(range(10,600,10)),
                     "param_name":"rank","prompt_offset":0,"image_indices":[0,1],"seeds":[None,None]},
        "ti": {"root":"avatar/results/ti",
                "pattern":"nv_{param}/epoch_{epoch}/prompt_{prompt}/image_{i}.png",
                "params":[1,2,4,8,16], "epochs":list(range(10,301,10)),
                "param_name":"nv","prompt_offset":0,"image_indices":[0,1],"seeds":[None,None]},
        "itay_loras": {"root":"sd-scripts/results/itay_loras",
                        "pattern":"rank_{param}/epoch_{epoch}/prompt_{prompt}/image_{i}_seed{seed}.png",
                        "params":[4,8,16,32,64], "epochs":list(range(10,300,10)),
                        "param_name":"rank","prompt_offset":1,
                        "image_indices":[1,2],"seeds":[6969,6970]},
        "cog_sdxl": {"root":"cog-sdxl/results",
                       "pattern":"rank_{param}/epoch_{epoch}/prompt_{prompt}/{seed}.png",
                       "params":[4,8,16,32,64], "epochs":list(range(9,600,10)),
                       "param_name":"rank","prompt_offset":0,"seeds":[6969,6970]}
    }

    prompt_file = "avatar/itay_prompt.py"
    prompts = load_prompts(prompt_file)
    clip_fn = init_clip_score_fn()
    fa = init_face_analyzer()

    print("Loading training embeddings...")
    train_embs = []
    for fname in os.listdir("avatar/data/itay"):
        img = Image.open(os.path.join("avatar/data/itay", fname)).convert("RGB")
        emb = get_arcface_embedding(fa, np.array(img))
        if emb is not None: train_embs.append(emb)
    train_embs = np.stack(train_embs)
    train_embs /= np.linalg.norm(train_embs,axis=1,keepdims=True)
    print(f"Loaded {len(train_embs)} embeddings.\n")

    for method in args.methods:
        cfg = methods_cfg[method]
        for param in cfg["params"]:
            print(f"\nEvaluating {method} with {cfg['param_name']}={param}")
            df = evaluate_param(method, cfg, param, prompts, clip_fn, fa, train_embs)
            outfile = f"evaluation_{method}_{cfg['param_name']}{param}.xlsx"
            df.to_excel(outfile, index=False)
            print(f"Saved results to {outfile}\n")

if __name__ == "__main__":
    main()
