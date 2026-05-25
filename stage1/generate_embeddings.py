"""
Stage 1 — Embedding Generation
================================
Loads a contrastively fine-tuned LLM (with LoRA adapters) and generates
L2-normalized last-token embeddings for all CTI report splits.

Usage:
    python stage1/generate_embeddings.py --config stage1/configs/andmal.yaml
    python stage1/generate_embeddings.py --config stage1/configs/bodmas.yaml
"""

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_and_split(json_path: Path, unseen_classes: list, test_size: float = 0.2, seed: int = 42):
    """Loads CTI JSON and returns strict GZSL splits (no leakage)."""
    with open(json_path) as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    unseen_mask = df["label"].isin(unseen_classes)
    df_unseen = df[unseen_mask].copy()
    df_seen = df[~unseen_mask].copy()

    assert df_seen["label"].isin(unseen_classes).sum() == 0, "Leakage detected in split!"

    df_train, df_test_seen = train_test_split(
        df_seen, test_size=test_size, stratify=df_seen["label"], random_state=seed
    )
    return df_train, df_test_seen, df_unseen


def get_base_model_path(adapter_dir: Path) -> str:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing adapter_config.json in {adapter_dir}")
    with open(config_path) as f:
        return json.load(f)["base_model_name_or_path"]


def get_last_token_embeddings(model, tokenizer, texts: list, batch_size: int = 8, device: str = "cuda") -> np.ndarray:
    """Extracts last non-padding token embeddings from the LLM."""
    model.eval()
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings", leave=False):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            seq_lengths = inputs["attention_mask"].sum(dim=1) - 1
            embeddings = last_hidden[torch.arange(last_hidden.shape[0], device=device), seq_lengths]
            all_embeddings.append(embeddings.cpu().float().numpy())

    embeddings = np.vstack(all_embeddings)
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-8)


def main():
    parser = argparse.ArgumentParser(description="Generate LLM embeddings for SMETA-ZSL Stage 1")
    parser.add_argument("--config", required=True, help="Path to dataset YAML config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_name = cfg["dataset_name"]
    unseen_classes = cfg["unseen_classes"]
    adapter_dir = Path(cfg["adapter_dir"])
    output_dir = Path(cfg["output_dir"])
    data_path = Path(cfg["cti_reports_path"])

    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{dataset_name}] Device: {device}")

    # Load and split CTI data
    df_train, df_test_seen, df_unseen = load_and_split(data_path, unseen_classes, seed=args.seed)
    df_train.to_pickle(output_dir / "train_data.pkl")
    df_test_seen.to_pickle(output_dir / "test_seen_data.pkl")
    df_unseen.to_pickle(output_dir / "unseen_data.pkl")
    print(f"[{dataset_name}] Splits saved — train: {len(df_train)}, seen-test: {len(df_test_seen)}, unseen: {len(df_unseen)}")

    # Load model
    base_model_id = get_base_model_path(adapter_dir)
    print(f"[{dataset_name}] Loading base model: {base_model_id}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    # Fix PEFT/Accelerate compatibility
    if hasattr(base_model, "_no_split_modules") and base_model._no_split_modules is not None:
        clean = []
        for m in base_model._no_split_modules:
            clean.extend(list(m) if isinstance(m, (set, list, tuple)) else [m])
        base_model._no_split_modules = clean

    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    # Generate and save embeddings
    for split_name, df in [("train", df_train), ("test_seen", df_test_seen), ("unseen", df_unseen)]:
        print(f"[{dataset_name}] Generating {split_name} embeddings ({len(df)} samples)...")
        emb = get_last_token_embeddings(model, tokenizer, df["description"].tolist(), args.batch_size, device)
        np.save(output_dir / f"{split_name}_embeddings.npy", emb)
        print(f"[{dataset_name}] Saved {split_name}_embeddings.npy — shape: {emb.shape}")

    del model, base_model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    print(f"[{dataset_name}] Done.")


if __name__ == "__main__":
    main()
