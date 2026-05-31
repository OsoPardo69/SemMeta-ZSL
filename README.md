# SMETA-ZSL: Semantic Meta-Alignment for Zero-Shot Threat Classification

Official implementation of **SMETA-ZSL**, a framework for generalized zero-shot learning (GZSL) on tabular cybersecurity data using LLM-derived semantic prototypes and episodic meta-learning.

> Paper under review at COLM 2026.

---

## Overview

Cyber defense systems struggle to recognize new threat families the moment they emerge, before labeled samples are available. SMETA-ZSL addresses this by using **Cyber Threat Intelligence (CTI) reports** — the natural language descriptions published by analysts — as the only supervision signal for unseen classes.

The framework has three components:

1. **Discriminative Semantic Prototype Learning** — A large language model (LLaMA-3.1-8B) is contrastively fine-tuned with a Supervised Contrastive loss and an isotropy regularizer to produce class-discriminative embeddings from overlapping CTI descriptions.

2. **Cross-Modal Alignment via Meta Knowledge Distillation** — A lightweight Student MLP learns to project behavioral tabular features (API calls, network traces, etc.) into the LLM semantic space. Training is structured as episodic meta-learning that explicitly simulates the zero-shot condition at every step, with knowledge distillation providing soft relational supervision.

3. **Adaptive Confidence Gating** — At inference, a parameter-free Z-score gate routes each sample to the seen or unseen class space based on how statistically dominant its best seen-class match is.

SMETA-ZSL is the only method that simultaneously supports:
- Zero-shot classification (no labeled unseen samples at inference)
- Training without unseen prototypes or unlabeled unseen instances
- Open-set recognition (unseen class identities unknown at training time)

### Results

Across 7 benchmarks (4 cybersecurity + 3 general-domain tabular datasets), SMETA-ZSL improves over the strongest baseline by **10.8 points on average** (harmonic mean), with gains up to **18.1 points**.

| Dataset | SMETA-ZSL (H) | Best Baseline (H) | Gain |
|---|---|---|---|
| CIC-AndMal-2020 | **57.78** | 39.70 (MZSL) | +18.1 |
| BODMAS | **50.20** | 48.12 (FL-ZSL) | +2.1 |
| APIGRAPH | **50.19** | 30.80 (TZSL) | +19.4 |
| AVASTCTU | **57.00** | 44.46 (FL-ZSL) | +12.5 |
| GOODREADS | **35.92** | 24.34 (ProtoLLM) | +11.6 |
| PETFINDER | **33.36** | 31.41 (FL-ZSL) | +1.9 |
| FAKEDDIT | 44.45 | **47.80** (ProtoLLM) | -3.4 |

---

## Repository Structure

```
smeta-zsl/
├── smeta_zsl/               # Core package
│   ├── model.py             # StudentMLP
│   ├── losses.py            # SupCon, isotropy, KD losses
│   ├── dataset.py           # MalwareTabularDataset
│   └── utils.py             # Shared data loading utilities
│
├── fine-tuning/             # Stage 1 — LLM fine-tuning & embedding generation
│   ├── train_supcon_debias_AndMAL.py
│   ├── generate_embeddings.py
│   ├── evaluate_embeddings.py
│   └── configs/             # Per-dataset YAML configs
│       ├── andmal.yaml
│       ├── bodmas.yaml
│       ├── avast.yaml
│       ├── apigraph.yaml
│       ├── goodreads.yaml
│       ├── petfinder.yaml
│       └── fakeddit.yaml
│
├── train/                   # Stage 2 — Meta-alignment training & evaluation
│   ├── train.py
│   ├── evaluate.py
│   └── configs/             # Per-dataset YAML configs (same structure)
│
├── baselines/               # All baseline implementations
│
├── data/                    # Pre-computed embeddings and CTI reports
│   ├── andmal/
│   │   ├── cti_reports.json
│   │   └── embeddings/
│   │       ├── llama8b/     # canonical (LLaMA-3.1-8B, used in paper)
│   │       ├── llama3b/
│   │       ├── gemma/
│   │       ├── qwen/
│   │       ├── qwen_coder/
│   │       └── mistral/
│   ├── bodmas/              # same per-model structure as andmal/
│   ├── apigraph/
│   │   ├── cti_reports.json
│   │   └── embeddings/      # LLaMA-3.1-8B embeddings
│   ├── avast/
│   ├── goodreads/
│   ├── petfinder/
│   └── fakeddit/
│
├── run.py                   # Single-command train + evaluate entry point
├── requirements.txt
└── setup.py
```

Each `embeddings/` directory contains:
- `train_embeddings.npy`, `test_seen_embeddings.npy`, `unseen_embeddings.npy`
- `train_data.pkl`, `test_seen_data.pkl`, `unseen_data.pkl`, `unseen_classes.pkl`

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/smeta-zsl.git
cd smeta-zsl
pip install -e .
```

Requirements: Python ≥ 3.10, PyTorch ≥ 2.1. Stage 1 (LLM fine-tuning) requires a CUDA GPU with ≥ 24 GB VRAM. Stage 2 runs on a single GPU with ≥ 8 GB VRAM.

---

## Quick Start

### Train + Evaluate with a Single Command

```bash
# Default model (llama3b) — available for all datasets
python run.py --andmal
python run.py --bodmas
python run.py --apigraph

# Select a specific embedding model
python run.py --andmal --llama8b
python run.py --bodmas --qwen
python run.py --avast  --gemma

# Multi-seed replication (5 seeds, reports mean ± std)
python run.py --andmal --llama8b --runs 5

# Fix a specific seed
python run.py --bodmas --qwen --seed 123

# Override the Z-score gate threshold
python run.py --andmal --llama8b --z_threshold 2.0
```

Available dataset flags: `--andmal`, `--bodmas`, `--apigraph`, `--avast`, `--goodreads`, `--petfinder`, `--fakeddit`

Available model flags: `--llama3b` (default), `--llama8b`, `--gemma`, `--qwen`, `--qwen_coder`, `--mistral`

---

## Step-by-Step Usage

### Stage 1 — Fine-tune the LLM and Generate Embeddings

Fine-tune the LLM with the SupCon + isotropy objective:

```bash
# Edit fine-tuning/configs/andmal.yaml to set adapter_dir
accelerate launch fine-tuning/train_supcon_debias_AndMAL.py --model llama8b
```

Then generate embeddings for each dataset:

```bash
python fine-tuning/generate_embeddings.py --config fine-tuning/configs/andmal.yaml
```

This writes `train_embeddings.npy`, `test_seen_embeddings.npy`, `unseen_embeddings.npy` and the corresponding `.pkl` metadata files to `data/andmal/embeddings/llama8b/`.

Optionally evaluate embedding quality:

```bash
python fine-tuning/evaluate_embeddings.py --config fine-tuning/configs/andmal.yaml
```

### Stage 2 — Train the Student MLP (advanced)

```bash
python train/train.py --config train/configs/andmal.yaml
```

To replicate the paper's 5-seed evaluation:

```bash
python train/train.py --config train/configs/andmal.yaml --runs 5
```

### Stage 2 — Evaluate a Checkpoint

```bash
python train/evaluate.py \
    --config train/configs/andmal.yaml \
    --checkpoint outputs/andmal/student_mlp_seed42.pt

# Sweep Z-score thresholds
python train/evaluate.py \
    --config train/configs/andmal.yaml \
    --checkpoint outputs/andmal/student_mlp_seed42.pt \
    --sweep_z
```

### Baselines

Each baseline in `baselines/` can be run independently. To run all baselines on a dataset:

```bash
python baselines/run_all.py
```

---

## Pre-computed Semantic Prototypes

To avoid non-determinism from LLM generation, we release the fixed `.npy` prototype files used in the paper. Download them from the [anonymous repository](https://anonymous.4open.science/r/SemMeta-ZSL-BE16/) and place them under `data/<dataset>/embeddings/llama8b/`.

---

## Hyperparameters

All hyperparameters are stored in the per-dataset YAML configs under `train/configs/`. Key values from the paper:

| Dataset | Support | Query | Episodes | λ_qry | T_KD | α |
|---|---|---|---|---|---|---|
| CIC-AndMal-2020 | 13 | 5 | 8000 | 2.0 | 3.0 | 0.7 |
| BODMAS | 15 | 20 | 4000 | 0.8 | 1.0 | 0.7 |
| APIGRAPH | 14 | 30 | 1000 | 2.5 | 4.0 | 0.0 |
| AVASTCTU | 2 | 4 | 5000 | 2.9 | 1.5 | 0.7 |
| GOODREADS | 4 | 2 | 8000 | 2.0 | 3.0 | 0.7 |
| PETFINDER | 2 | 1 | 8000 | 2.0 | 3.0 | 0.7 |
| FAKEDDIT | 3 | 1 | 8000 | 2.0 | 3.0 | 0.7 |

Sensitivity analysis across all hyperparameters is provided in Appendix 9 of the paper.

---

## Citation

```bibtex
@inproceedings{smeta-zsl-2026,
  title     = {SMETA-ZSL: Semantic Meta-Alignment for Zero-Shot Threat Classification},
  author    = {Anonymous},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026},
  note      = {Under review}
}
```

---

## License

This repository will be released under the MIT License upon paper acceptance.
