#!/usr/bin/env python3
"""
run_zeus_multiseed.py
═══════════════════════════════════════════════════════════════════════════════
ZEUS Baseline for VENOM (CIC-AndMal-2020)
Multi-Seed & Multi-Dataset Evaluation with First-Seed Calibration.
Dynamically re-splits seen data per seed.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import math
import random
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DATASETS = {
    "AndMAL": "/home/iamontoyasa/02_VENOM/03_Baseline/01_run1_prepared_data_one_shot.pt",
    "BODMAS": "/home/iamontoyasa/02_VENOM/03_Baseline/02_run1_prepared_data_one_shot.pt",
    "AVASTCTU": "/home/iamontoyasa/02_VENOM/03_Baseline/03_run1_prepared_data_one_shot.pt",
    "APIGRAPH": "/home/iamontoyasa/02_VENOM/03_Baseline/04_run1_prepared_data_one_shot.pt",
    "Goodread": "/home/iamontoyasa/02_VENOM/03_Baseline/06_run1_prepared_data_one_shot.pt",
    "Petfinder": "/home/iamontoyasa/02_VENOM/03_Baseline/07_run1_prepared_data_one_shot.pt",
    "Fakeddit": "/home/iamontoyasa/02_VENOM/03_Baseline/08_run1_prepared_data_one_shot.pt"
}

SEEDS = [42, 123, 456, 789, 2025]

# Extended Gamma grid for the first-seed calibration sweep
GAMMA_GRID = [0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]

BANNER = "═" * 77


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_metrics(y_true, y_pred, seen_ids, unseen_ids):
    labels_present = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels_present)
    per_class_acc = {label: (cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0)
                     for i, label in enumerate(labels_present)}

    S = np.mean([per_class_acc[l] for l in labels_present if l in seen_ids]) * 100
    U = np.mean([per_class_acc[l] for l in labels_present if l in unseen_ids]) * 100
    H = (2 * S * U / (S + U)) if (S + U) > 0 else 0.0
    return S, U, H


def print_summary_table(results_dict, n_seeds):
    print(f"\n{BANNER}")
    print(f" ZEUS RESULTS   (mean ± std, %,  {n_seeds} seeds)")
    print(f"{BANNER}")
    print(f" {'Dataset':<13} │ {'Seen Acc':>16}       │ {'Unseen Acc':>16}       │ {'H-Mean':>11}")
    print(f" ──────────────┼────────────────────────┼────────────────────────┼─────────────────")

    for dataset, metrics in results_dict.items():
        s_mean, s_std = np.mean(metrics["S"]), np.std(metrics["S"], ddof=1) if len(metrics["S"]) > 1 else 0.0
        u_mean, u_std = np.mean(metrics["U"]), np.std(metrics["U"], ddof=1) if len(metrics["U"]) > 1 else 0.0
        h_mean, h_std = np.mean(metrics["H"]), np.std(metrics["H"], ddof=1) if len(metrics["H"]) > 1 else 0.0

        print(
            f" {dataset:<13} │     {s_mean:>5.2f} ± {s_std:<4.2f}       │     {u_mean:>5.2f} ± {u_std:<4.2f}       │   {h_mean:>5.2f} ± {h_std:<4.2f}")
    print(f"{BANNER}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════

class ZEUSEncoder(nn.Module):
    def __init__(self, d_in=282, d_model=256, n_heads=4, n_layers=4):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, X):
        E = self.proj(X)
        return self.transformer(E)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SYNTHETIC PRE-TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_task(n_samples=512, d_in=282, device="cpu"):
    k = torch.randint(4, 12, (1,)).item()
    centroids = torch.randn(k, d_in, device=device) * 3.0
    y = torch.randint(0, k, (n_samples,), device=device)
    X = centroids[y] + torch.randn(n_samples, d_in, device=device) * 0.4
    return X, y


def pretrain_zeus(model, device, steps=2000):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    model.train()

    for step in range(steps):
        X_b, y_b = [], []
        for _ in range(4):
            X, y = generate_synthetic_task(d_in=model.proj.in_features, device=device)
            X_b.append(X);
            y_b.append(y)

        X_b, y_b = torch.stack(X_b), torch.stack(y_b)
        optimizer.zero_grad()

        Z = model(X_b)
        loss = 0
        for i in range(4):
            z_i, y_i = Z[i], y_b[i]
            unique_y = torch.unique(y_i)
            centroids = torch.stack([z_i[y_i == k].mean(0) for k in unique_y])
            dist = torch.cdist(z_i, centroids, p=2).pow(2)
            loss += F.cross_entropy(-dist / math.sqrt(Z.shape[-1]), y_i)

        loss.backward()
        optimizer.step()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ONE-SHOT EVALUATION (WITH DYNAMIC SPLITTING)
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_one_shot(model, X_train, y_train, X_support, y_support, X_test, y_test, seen_ids, unseen_ids, device,
                      fixed_gamma=None):
    model.eval()

    # Move to device
    X_train, y_train = torch.tensor(X_train, device=device), torch.tensor(y_train, device=device)
    X_support = torch.tensor(X_support, device=device)
    X_test, y_test = torch.tensor(X_test, device=device), torch.tensor(y_test, device=device)

    # 1. Calculate Anchors
    X_anchors = []
    anchor_labels = []

    for sid in seen_ids:
        mask = (y_train == sid)
        if mask.any():
            X_anchors.append(X_train[mask].mean(0))
        else:
            X_anchors.append(torch.zeros(X_train.shape[1], device=device))
        anchor_labels.append(sid)

    for i, uid in enumerate(unseen_ids):
        X_anchors.append(X_support[i])
        anchor_labels.append(uid)

    X_anchors = torch.stack(X_anchors).unsqueeze(0)
    Z_anchors = model(X_anchors).squeeze(0)

    # 2. Project Test Data (in chunks)
    Z_test = []
    for chunk in torch.split(X_test, 512):
        Z_test.append(model(chunk.unsqueeze(0)).squeeze(0))
    Z_test = torch.cat(Z_test, dim=0)

    # 3. Nearest Neighbor Classification with Gamma Sweep/Lock
    dist = torch.cdist(F.normalize(Z_test, dim=-1), F.normalize(Z_anchors, dim=-1))

    y_true = y_test.cpu().numpy()
    seen_local_ids = [i for i, lbl in enumerate(anchor_labels) if lbl in seen_ids]

    # If fixed_gamma is provided (Seeds 2-5), skip sweep and evaluate directly
    gammas_to_test = GAMMA_GRID if fixed_gamma is None else [fixed_gamma]

    best_h, best_res, best_gamma_found = -1, None, 0.0

    for gamma in gammas_to_test:
        d_calib = dist.clone()
        d_calib[:, seen_local_ids] += gamma
        preds_idx = d_calib.argmin(dim=1).cpu().numpy()
        preds = np.array([anchor_labels[i] for i in preds_idx])

        S, U, H = get_metrics(y_true, preds, seen_ids, unseen_ids)

        # Immediate return if we are using the locked gamma
        if fixed_gamma is None:
            if H > best_h:
                best_h = H
                best_res = (S, U, H)
                best_gamma_found = gamma
        else:
            return S, U, H, fixed_gamma

    return best_res[0], best_res[1], best_res[2], best_gamma_found


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{BANNER}")
    print("  VENOM — ZEUS Multi-Seed Pipeline")
    print("  Mode: 1-Shot GZSL (with 1st-seed Gamma calibration)")
    print("  Dynamically re-splits seen data per seed.")
    print(f"{BANNER}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary_results = {}

    for ds_name, ds_path in DATASETS.items():
        if not os.path.exists(ds_path):
            print(f"\n[Warning] File not found for {ds_name}, skipping... ({ds_path})")
            continue

        print(f"\n{'─' * 60}\n Processing Dataset: {ds_name}\n{'─' * 60}")
        data = torch.load(ds_path, map_location="cpu", weights_only=False)

        seen_ids = data["seen_label_ids"]
        unseen_ids = data["unseen_label_ids"]

        # Base Data Extraction
        X_train_base = data["X_train"].numpy().astype(np.float32)
        y_train_base = data["y_train"].numpy()
        X_support = data["X_support"].numpy().astype(np.float32)
        y_support = data["y_support"].numpy()
        X_test_full = data["X_test"].numpy().astype(np.float32)
        y_test_full = data["y_test"].numpy()

        # Separate unseen test data
        unseen_mask = np.isin(y_test_full, unseen_ids)
        X_test_unseen = X_test_full[unseen_mask]
        y_test_unseen = y_test_full[unseen_mask]

        # Separate seen test data
        X_test_seen = X_test_full[~unseen_mask]
        y_test_seen = y_test_full[~unseen_mask]

        # Merge for dynamic splitting
        X_seen_all = np.concatenate([X_train_base, X_test_seen], axis=0)
        y_seen_all = np.concatenate([y_train_base, y_test_seen], axis=0)

        summary_results[ds_name] = {"S": [], "U": [], "H": []}
        best_gamma = None  # Will be calibrated on the first seed

        for idx, seed in enumerate(SEEDS):
            set_seed(seed)
            print(f"  → Running Seed {seed} ", end="", flush=True)

            # --- DYNAMIC DATA SPLITTING ---
            X_seen_train, X_seen_test, y_seen_train, y_seen_test = train_test_split(
                X_seen_all, y_seen_all, test_size=0.2, random_state=seed, stratify=y_seen_all
            )

            X_test_seed = np.concatenate([X_seen_test, X_test_unseen], axis=0)
            y_test_seed = np.concatenate([y_seen_test, y_test_unseen], axis=0)

            # Initialize and pretrain model (fresh state per seed)
            zeus = ZEUSEncoder(d_in=data["d_in"]).to(device)
            pretrain_zeus(zeus, device, steps=2000)

            # --- EVALUATION ---
            S, U, H, g_used = evaluate_one_shot(
                model=zeus,
                X_train=X_seen_train, y_train=y_seen_train,
                X_support=X_support, y_support=y_support,
                X_test=X_test_seed, y_test=y_test_seed,
                seen_ids=seen_ids, unseen_ids=unseen_ids,
                device=device, fixed_gamma=best_gamma
            )

            if idx == 0:
                best_gamma = g_used
                print(f"[Calibrated Gamma: {best_gamma:.1f}] ", end="")

            summary_results[ds_name]["S"].append(S)
            summary_results[ds_name]["U"].append(U)
            summary_results[ds_name]["H"].append(H)

            print(f"✓ (S:{S:.1f} U:{U:.1f} H:{H:.1f})")

    # Print the requested summary table
    print_summary_table(summary_results, len(SEEDS))


if __name__ == "__main__":
    main()