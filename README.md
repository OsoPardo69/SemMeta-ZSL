# SMETA-ZSL: Semantic Meta-Alignment for Zero-Shot Threat Classification

Official implementation of **SMETA-ZSL**, a framework for generalized zero-shot learning (GZSL) on tabular cybersecurity data using LLM-derived semantic prototypes and episodic meta-learning.

> Paper under review at COLM 2026.

---

## Overview

Cyber defense systems struggle to recognize new threat families the moment they emerge, before labeled samples are available. SMETA-ZSL addresses this by using **Cyber Threat Intelligence (CTI) reports** — the natural language descriptions published by analysts — as the only supervision signal for unseen classes.
<img width="1507" height="287" alt="Framework_Overview" src="https://github.com/user-attachments/assets/ed30749c-46b9-4752-a038-4763b9049483" />

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
├── stage1/                  # LLM fine-tuning & embedding generation
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
├── stage2/                  # Meta-alignment training & evaluation
│   ├── train.py
│   ├── evaluate.py
│   └── configs/             # Per-dataset YAML configs (same structure)
│
├── baselines/               # All baseline implementations
├── data/                    # Dataset placeholder (see data/README.md)
├── requirements.txt
└── setup.py
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/smeta-zsl.git
cd smeta-zsl
pip install -e .
```

Requirements: Python ≥ 3.10, PyTorch ≥ 2.1, a CUDA-capable GPU with ≥ 24 GB VRAM for Stage 1 (LLM inference). Stage 2 runs on a single GPU with ≥ 8 GB VRAM.

---

## Data Setup

Place dataset files under `data/` following the structure described in [data/README.md](data/README.md). Each dataset requires:
- A CTI reports JSON file: `[{label, description}, ...]`
- A tabular feature CSV: rows are samples, last column is the class label

Dataset download links are listed in [data/README.md](data/README.md).

---

## Usage

### Stage 1 — Generate LLM Embeddings

Fine-tune the LLM with the SupCon + isotropy objective (training script from the paper's training run), then generate embeddings for each dataset:

```bash
# Edit stage1/configs/andmal.yaml to set your paths and adapter checkpoint
python stage1/generate_embeddings.py --config stage1/configs/andmal.yaml
```

This writes `train_embeddings.npy`, `test_seen_embeddings.npy`, `unseen_embeddings.npy` and the corresponding `.pkl` metadata files to `data/andmal/embeddings/`.

Optionally evaluate embedding quality:

```bash
python stage1/evaluate_embeddings.py --config stage1/configs/andmal.yaml
```

### Stage 2 — Train the Student MLP

```bash
python stage2/train.py --config stage2/configs/andmal.yaml
```

To replicate the paper's 5-seed evaluation:

```bash
python stage2/train.py --config stage2/configs/andmal.yaml --runs 5
```

### Stage 2 — Evaluate

```bash
python stage2/evaluate.py \
    --config stage2/configs/andmal.yaml \
    --checkpoint outputs/andmal/student_mlp_seed42.pt

# Sweep Z-score thresholds to find the optimal value
python stage2/evaluate.py \
    --config stage2/configs/andmal.yaml \
    --checkpoint outputs/andmal/student_mlp_seed42.pt \
    --sweep_z
```

### Baselines

Each baseline in `baselines/` can be run independently. See comments at the top of each file for usage. To run all baselines on a dataset:

```bash
python baselines/run_all.py
```

---

## Hyperparameters

All hyperparameters are stored in the per-dataset YAML configs under `stage2/configs/`. Key values from the paper:

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
