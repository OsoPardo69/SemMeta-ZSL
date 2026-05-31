"""
Stage 1 — Embedding Quality Evaluation
========================================
Loads pre-generated embeddings and reports quality metrics:
  - KNN classification accuracy
  - Silhouette score (cosine)
  - Intra/inter-class cosine similarity gap
  - t-SNE visualization (saved as PNG)

Usage:
    python stage1/evaluate_embeddings.py --config stage1/configs/andmal.yaml
    python stage1/evaluate_embeddings.py --config stage1/configs/andmal.yaml --output_dir /path/to/figures
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_embeddings(emb_dir: Path):
    train_emb = np.load(emb_dir / "train_embeddings.npy")
    test_seen_emb = np.load(emb_dir / "test_seen_embeddings.npy")
    unseen_emb = np.load(emb_dir / "unseen_embeddings.npy")
    train_meta = pd.read_pickle(emb_dir / "train_data.pkl")
    test_seen_meta = pd.read_pickle(emb_dir / "test_seen_data.pkl")
    unseen_meta = pd.read_pickle(emb_dir / "unseen_data.pkl")
    return train_emb, test_seen_emb, unseen_emb, train_meta, test_seen_meta, unseen_meta


def similarity_metrics(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    emb = F.normalize(torch.tensor(embeddings), p=2, dim=1).numpy()
    intra, inter = [], []
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(np.dot(emb[i], emb[j]))
            (intra if labels[i] == labels[j] else inter).append(sim)
    return {
        "mean_intra_sim": float(np.mean(intra)) if intra else 0.0,
        "mean_inter_sim": float(np.mean(inter)) if inter else 0.0,
        "separation_gap": float(np.mean(intra) - np.mean(inter)) if intra and inter else 0.0,
    }


def knn_accuracy(embeddings: np.ndarray, labels: np.ndarray, test_size: float = 0.3, k: int = 5) -> float:
    n_classes = len(np.unique(labels))
    test_size = max(test_size, (n_classes + 1) / len(labels))
    X_tr, X_te, y_tr, y_te = train_test_split(
        embeddings, labels, test_size=test_size, random_state=42, stratify=labels
    )
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    knn.fit(X_tr, y_tr)
    return float(accuracy_score(y_te, knn.predict(X_te)))


def plot_tsne(embeddings: np.ndarray, labels: np.ndarray, label_encoder: LabelEncoder, title: str, save_path: Path):
    tsne = TSNE(n_components=2, perplexity=30, max_iter=300, random_state=42, metric="cosine")
    proj = tsne.fit_transform(embeddings)
    unique = np.unique(labels)
    cmap = plt.cm.get_cmap("tab20", len(unique))
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, lid in enumerate(unique):
        idx = np.where(labels == lid)[0]
        ax.scatter(proj[idx, 0], proj[idx, 1], color=cmap(i), label=label_encoder.inverse_transform([lid])[0],
                   alpha=0.7, s=30)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"t-SNE saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 1 embedding quality")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default=None, help="Directory for figures (defaults to embedding dir)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    emb_dir = Path(cfg["output_dir"])
    dataset_name = cfg["dataset_name"]
    fig_dir = Path(args.output_dir) if args.output_dir else emb_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    train_emb, test_seen_emb, unseen_emb, train_meta, test_seen_meta, unseen_meta = load_embeddings(emb_dir)

    # Use validation split (seen) for evaluation
    all_emb = np.concatenate([train_emb, test_seen_emb], axis=0)
    all_meta = pd.concat([train_meta, test_seen_meta], ignore_index=True)

    le = LabelEncoder()
    labels_enc = le.fit_transform(all_meta["label"].values)

    print(f"\n=== {dataset_name} — Stage 1 Embedding Quality ===")
    print(f"Seen embeddings: {all_emb.shape}")

    knn_acc = knn_accuracy(all_emb, labels_enc)
    print(f"KNN Accuracy (k=5, cosine, test=30%): {knn_acc:.4f}")

    sil = silhouette_score(all_emb, labels_enc, metric="cosine")
    print(f"Silhouette Score (cosine):             {sil:.4f}")

    sim = similarity_metrics(all_emb[:500], labels_enc[:500])  # subsample for speed
    print(f"Mean Intra-class Similarity:           {sim['mean_intra_sim']:.4f}")
    print(f"Mean Inter-class Similarity:           {sim['mean_inter_sim']:.4f}")
    print(f"Separation Gap:                        {sim['separation_gap']:.4f}")

    plot_tsne(all_emb, labels_enc, le, f"{dataset_name} — Finetuned LLM Embeddings",
              fig_dir / f"{dataset_name}_tsne.png")


if __name__ == "__main__":
    main()
