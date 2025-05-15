# Avatar Personalization Project

A comprehensive pipeline for personalizing Stable Diffusion XL to generate consistent, high-quality avatar images for a subject (e.g., `<itay>` or `<iris>`). Supports three adaptation methods:

- **Textual Inversion (TI)**
- **LoRA Fine‑Tuning (SDXL LORA)**
- **DreamBooth with LoRA (DB LORA)**

This README explains project structure, setup, training, inference, evaluation, and examples of usage.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Training](#training)

   - [Textual Inversion (TI)](#textual-inversion-ti)
   - [LoRA Fine‑Tuning (SDXL LORA)](#lora-fine-tuning-sdxl-lora)
   - [DreamBooth LoRA (DB LORA)](#dreambooth-lora-db-lora)

6. [Inference](#inference)

   - [TI Inference](#ti-inference)
   - [LORA & DB Inference](#lora--db-inference)

7. [Evaluation](#evaluation)
8. [Prompts & Metadata](#prompts--metadata)
9. [Scripts Overview](#scripts-overview)
10. [License](#license)

---

## Project Structure

```
avatar/
├── data/irit/metadata.jsonl         # IRIT dataset metadata
├── data/itay/                       # Reference images for `<itay>` (23 .jpg files)
├── diffuser/                       # Fine‑tuning scripts
│   ├── sdxl_lora.py                # LoRA training for SDXL
│   ├── textual_inversion.py        # TI training
│   ├── dreambooth_lora.py          # DreamBooth + LoRA training
│   └── ...                         # utility modules & data loaders
├── infer/                          # Inference entrypoints
│   ├── textual_inversion_infer.py  # Apply TI embeddings
│   ├── lora_infer.py               # Run LoRA-adapted SDXL
│   └── db_infer.py                 # Run DB LoRA-adapted SDXL
├── scripts/                        # Bash wrappers
│   ├── diffuser_ti.sh              # TI hyperparam sweep
│   ├── diffuser_lora.sh            # LORA hyperparam sweep
│   ├── diffuser_dreambooth.sh      # DB LORA hyperparam sweep
│   ├── infer_ti.sh                 # TI inference
│   ├── infer_db.sh                 # DB inference
│   ├── eval.sh                     # Unified evaluation script
│   └── ...                         # other wrappers
├── metrics.py                      # Epoch-level metric computation
├── evaluate_metrics.py             # Aggregate CI    metrics & Excel export
├── itay_prompt.py                  # Positive/negative prompts for `<itay>`
├── iris_prompt.py                  # Positive/negative prompts for `<iris>`
├── metadata.jsonl                  # repeated? double-check
├── requirements.txt                # Python deps
├── accelerate_config.yaml          # DeepSpeed & bf16 config
└── test_ti.py                      # Quick smoke test for TI
```

## Installation

```bash
# Create & activate virtual env
conda create -n avatar python==3.10 -y
conda activate avatar

# Install dependencies
pip install --upgrade pip
pip install -r avatar/requirements.txt
# Additional for metrics:
pip install torchmetrics open_clip_torch timm insightface adaface-pytorch
```

## Configuration

Customize `avatar/accelerate_config.yaml`:

- `mixed_precision: bf16`
- `distributed_type: DEEPSPEED`
- `gpu_ids: 0` etc.

Ensure `$CUDA_VISIBLE_DEVICES` matches your GPU.

---

## Training

### Textual Inversion (TI)

1. **Hyperparameter sweep over `--num_vectors`**
2. Save embeddings every block of epochs.

```bash
# wrapper: scripts/diffuser_ti.sh (adjust NVECTORS and NV block)
bash avatar/scripts/diffuser_ti.sh
```

**Direct CLI**:

```bash
accelerate launch avatar/diffuser/textual_inversion.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --train_data_dir avatar/data/itay \
  --placeholder_token <itay> --initializer_token man \
  --resolution 1024 --train_batch_size 4 --gradient_accumulation_steps 4 \
  --num_vectors 16 --num_train_epochs 300 \
  --mixed_precision bf16 --scale_lr --lr_scheduler cosine_with_restarts \
  --output_dir output_textual_inversion/nv_16
```

### LoRA Fine‑Tuning (SDXL LORA)

Sweep over LoRA rank:

```bash
bash avatar/scripts/diffuser_lora.sh
```

**Direct CLI**:

```bash
accelerate launch avatar/diffuser/sdxl_lora.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --train_data_dir avatar/data/irit \
  --rank 8 --resolution 1024 --train_batch_size 8 --num_train_epochs 600 \
  --mixed_precision bf16 --gradient_checkpointing --enable_xformers_memory_efficient_attention \
  --output_dir irit_output/rank8
```

### DreamBooth LoRA (DB LORA)

Sweep over LoRA rank for instance images in `/data/itay`:

```bash
bash avatar/scripts/diffuser_dreambooth.sh
```

**Direct CLI**:

```bash
accelerate launch avatar/diffuser/dreambooth_lora.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --instance_data_dir avatar/data/itay --instance_prompt "photo of <itay> man" \
  --validation_prompt "photo of <itay> man in Central Park" \
  --rank 64 --resolution 1024 --train_batch_size 4 --num_train_epochs 600 \
  --checkpointing_epochs 10 --mixed_precision bf16 \
  --gradient_checkpointing --enable_xformers_memory_efficient_attention \
  --output_dir output_dreambooth/rank_64
```

---

## Inference

### TI Inference

```bash
# Apply learned textual inversion
bash avatar/scripts/infer_ti.sh
```

or:

```bash
python avatar/infer/textual_inversion_infer.py --nv 16
```

### LoRA & DB Inference

```bash
# DreamBooth
bash avatar/scripts/infer_db.sh
# LoRA inference module
python avatar/infer/lora_infer.py \
  --pretrained_model=stabilityai/stable-diffusion-xl-base-1.0 \
  --lora_root outputs_dreambooth/rank_64 --ranks 64 \
  --prompts_module itay_prompt --output_dir results/db_lora
```

---

## Evaluation

Compute CLIP‑T, FID, CLIP‑I, DINO‑V2, ArcFace & AdaFace similarities.

```bash
# Unified evaluation:
bash avatar/scripts/eval.sh lora db_lora ti
```

or:

```bash
python avatar/evaluate_metrics.py --methods lora db_lora ti
```

Results are exported to `evaluation_<method>_<param>.xlsx`.

---

## Prompts & Metadata

- **`avatar/itay_prompt.py`**: Positive and negative prompt templates for `<itay>`
- **`avatar/iris_prompt.py`**: Templates for `<iris>`
- **`avatar/metadata.jsonl`**: JSONL metadata for training datasets

---

## Scripts Overview

| Script                 | Purpose                             |
| ---------------------- | ----------------------------------- |
| `sdxl_lora.py`         | LoRA fine‑tuning                    |
| `textual_inversion.py` | Textual inversion training          |
| `dreambooth_lora.py`   | DreamBooth + LoRA adaptation        |
| `infer/*.py`           | Inference entrypoints               |
| `evaluate_metrics.py`  | Aggregated metric logging & export  |
| `metrics.py`           | Single‐epoch metric computation     |
| `scripts/*.sh`         | Bash wrappers for hyperparam sweeps |

---
