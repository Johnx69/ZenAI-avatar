from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# External libs (≈ pip install …):
#   • torchmetrics>=1.3.0   – CLIP‑T & FID
#   • open_clip_torch       – CLIP back‑end for CLIP‑I
#   • timm>=0.9.12          – DINO‑V2 backbone
#   • insightface           – ArcFace  i‑preservation
#   • adaface‑pytorch       – AdaFace  i‑preservation
# ---------------------------------------------------------------------------
from torchmetrics.functional.multimodal.clip_score import clip_score as _clip_score
from torchmetrics.image.fid import FrechetInceptionDistance
from open_clip import create_model_from_pretrained, get_tokenizer
import timm
import insightface
from adaface_pytorch import load_pretrained_adaface

__all__ = [
    "evaluate_epoch",
    "save_metrics",
]


# --------------------------- model singletons --------------------------------
@torch.no_grad()
def _load_backbones(device: str = "cpu"):
    """Lazily create heavy back‑bones once per process."""
    global _MODELS  # pylint: disable=global‑at‑module‑level
    if "_MODELS" in globals():
        return _MODELS

    # CLIP ViT‑B/32 for text‑image alignment (CLIP‑T)
    clip_model, _, _ = create_model_from_pretrained("laion2b_s34b_b79k")
    clip_model = clip_model.to(device).eval()
    clip_tokenizer = get_tokenizer("laion2b_s34b_b79k")

    # OpenCLIP ViT‑G/14 for CLIP‑I (image‑image)
    clip_img_model, _, _ = create_model_from_pretrained("hf_hub:", pretrained_cfg="openai/clip‑vit‑large‑patch14")
    clip_img_model = clip_img_model.to(device).eval()

    # DINO‑V2 – ViT‑S/8 (timm id: "dino_vits8")
    dino = timm.create_model("dino_vits8", pretrained=True).to(device).eval()

    # ArcFace (InsightFace – iresnet100)
    arcface = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    arcface.prepare(ctx_id=int(device.split(":")[-1]))

    # AdaFace
    adaface = load_pretrained_adaface("ir_101", fp16=("cuda" in device)).to(device).eval()

    _MODELS = {
        "clip_model": clip_model,
        "clip_tokenizer": clip_tokenizer,
        "clip_img_model": clip_img_model,
        "dino": dino,
        "arc": arcface,
        "ada": adaface,
    }
    return _MODELS


# ------------------------------- helpers -------------------------------------
@torch.no_grad()
def _img_to_tensor(img: Image.Image, size: int = 224) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    ])
    return tfm(img)


@torch.no_grad()
def _stack_images(imgs: List[Image.Image], size: int = 224, device: str = "cpu") -> torch.Tensor:
    return torch.stack([_img_to_tensor(im, size) for im in imgs]).to(device)


# --------------------------- metric functions --------------------------------
@torch.no_grad()
def _clip_t_score(imgs: torch.Tensor, prompts: List[str], clip_model, tokenizer) -> float:
    """Text‑image alignment (higher = better)."""
    imgs_int = (imgs * 255).clamp(0, 255).type(torch.uint8)
    score = _clip_score(imgs_int, prompts, model_name_or_path="openai/clip‑vit‑base‑patch16").item()
    return score


@torch.no_grad()
def _clip_i_score(gen_imgs: torch.Tensor, real_imgs: torch.Tensor, clip_img_model) -> float:
    feats_fake = clip_img_model.encode_image(gen_imgs).float()
    feats_real = clip_img_model.encode_image(real_imgs).float()
    feats_fake = F.normalize(feats_fake, dim=1)
    feats_real = F.normalize(feats_real, dim=1)
    sim = (feats_fake @ feats_real.T).mean().item()
    return sim


@torch.no_grad()
def _dino_v2_score(gen_imgs: torch.Tensor, real_imgs: torch.Tensor, dino) -> float:
    feats_fake = F.normalize(dino(gen_imgs), dim=1)
    feats_real = F.normalize(dino(real_imgs), dim=1)
    return (feats_fake @ feats_real.T).mean().item()


@torch.no_grad()
def _fid_score(gen_imgs: torch.Tensor, real_imgs: torch.Tensor, fid_metric: FrechetInceptionDistance) -> float:
    fid_metric.reset()
    fid_metric.update(real_imgs, real=True)
    fid_metric.update(gen_imgs, real=False)
    return float(fid_metric.compute())


@torch.no_grad()
def _identity_scores(gen_imgs: List[Image.Image], ref_imgs: List[Image.Image], arc, ada, device: str) -> Tuple[float, float]:
    # ArcFace via insightface's FaceAnalysis ⇒ produces 512‑D embeddings, already L2‑normed.
    def _get_arc_emb(im):
        face = arc.get(im)[0] if arc.get(im) else None
        return torch.from_numpy(face.normed_embedding) if face else None

    ref_arc = [e for im in ref_imgs if (e := _get_arc_emb(np.array(im)))]
    ref_arc = torch.stack(ref_arc).to(device)  # (R,512)

    sims_arc = []
    sims_ada = []
    for im in gen_imgs:
        arr = np.array(im)[..., ::-1]  # BGR for insightface
        emb_arc = _get_arc_emb(arr)
        if emb_arc is not None:
            sims_arc.append((emb_arc.to(device) @ ref_arc.T).max().item())
        # AdaFace – expects 112×112 RGB tensor
        tf = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])
        emb_ada = ada(tf(im).unsqueeze(0).to(device))
        emb_ada = F.normalize(emb_ada, dim=1)
        ref_ada = ada(torch.stack([tf(r) for r in ref_imgs]).to(device))
        ref_ada = F.normalize(ref_ada, dim=1)
        sims_ada.append((emb_ada @ ref_ada.T).max().item())

    return float(np.mean(sims_arc)), float(np.mean(sims_ada))


# ------------------------------- main API ------------------------------------
@torch.no_grad()
def evaluate_epoch(
    images: List[Image.Image],
    prompts: List[str],
    fid_real_dir: str | Path,
    ref_face_dir: str | Path,
    device: str = "cuda:0",
) -> Dict[str, float]:
    """Compute all metrics for *one epoch*.

    Parameters
    ----------
    images : List[PIL.Image]
        Images generated *this epoch* – **one per prompt**.
    prompts : List[str]
        Prompts used to generate `images` (same order).
    fid_real_dir : Path
        Folder of JPEG/PNG images (high‑quality real faces) for FID + other
        image‑image metrics.
    ref_face_dir : Path
        Folder of subject's reference images – used for identity scores.
    device : str
        CUDA device (or "cpu").
    """
    if len(images) != len(prompts):
        raise ValueError("images and prompts length mismatch")

    models = _load_backbones(device)
    # ------------------- stack tensors @ 224×224 -------------------
    gen_t = _stack_images(images, 224, device)

    # sample up to 256 real images for speed
    real_paths = list(Path(fid_real_dir).glob("*.jpg"))[:256]
    real_imgs = [Image.open(p).convert("RGB") for p in real_paths]
    real_t = _stack_images(real_imgs, 224, device)

    # -------------------------- metrics ----------------------------
    fid_metric = FrechetInceptionDistance(normalize=True).to(device)

    scores = {
        "CLIP-T": _clip_t_score(gen_t, prompts, models["clip_model"], models["clip_tokenizer"]),
        "FID": _fid_score(gen_t, real_t, fid_metric),
        "CLIP-I": _clip_i_score(gen_t, real_t, models["clip_img_model"]),
        "DINO-V2": _dino_v2_score(gen_t, real_t, models["dino"]),
    }

    # Identity metrics
    ref_imgs = [Image.open(p).convert("RGB") for p in Path(ref_face_dir).glob("*.jpg")]
    arc_sim, ada_sim = _identity_scores(images, ref_imgs, models["arc"], models["ada"], device)
    scores.update({
        "ArcFace-sim": arc_sim,
        "AdaFace-sim": ada_sim,
    })

    return scores


# --------------------------- persistence -------------------------------------
def save_metrics(out_path: str | Path, epoch: int, score_dict: Dict[str, float]):
    out_path = Path(out_path)
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}
    data[str(epoch)] = score_dict
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
