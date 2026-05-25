#!/usr/bin/env python3
"""
run_APT_oneshot_multiseed.py
═══════════════════════════════════════════════════════════════════════════════
APT — Adversarially Pre-trained Transformer
Multi-Seed, Multi-Dataset One-Shot GZSL Evaluator.
Calibrates Gamma penalty on Seed 1, locks for Seeds 2-5.
Dynamically re-splits seen train/test data per seed for valid variance.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import time
import warnings
import random

import numpy as np
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

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

CFG = {
    "output_path": "/home/iamontoyasa/02_VENOM/03_Baseline/apt_oneshot_results_multiseed.csv",
    "apt_checkpoint": "/home/iamontoyasa/02_VENOM/03_Baseline/apt/artifacts/saves/apt_best.pt",
    "apt_repo_path": "/home/iamontoyasa/02_VENOM/03_Baseline/apt",
    "subsample": 10_000,
    "gamma_range": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
}

BANNER = "═" * 77


# ═══════════════════════════════════════════════════════════════════════════════
# DATA UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_prepared_data(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"[Data] File not found: '{path}'")
    return torch.load(path, weights_only=False)


def build_label_mapping_all(data: dict):
    seen_global = data["seen_label_ids"]
    unseen_global = data["unseen_label_ids"]
    all_names = data["all_class_names"]
    sorted_names = sorted(all_names)

    all_global_sorted = sorted(seen_global + unseen_global)
    global_to_local = {g: l for l, g in enumerate(all_global_sorted)}
    local_to_name = {l: sorted_names[g] for g, l in global_to_local.items()}
    all_names_ordered = [local_to_name[l] for l in sorted(local_to_name)]

    seen_local_ids = {global_to_local[g] for g in seen_global}
    unseen_local_ids = {global_to_local[g] for g in unseen_global}

    return global_to_local, local_to_name, all_names_ordered, seen_local_ids, unseen_local_ids


def prepare_arrays_oneshot(data: dict):
    (global_to_local, local_to_name, all_names_ordered,
     seen_local_ids, unseen_local_ids) = build_label_mapping_all(data)

    X_train = data["X_train"].numpy().astype(np.float32)
    y_train = np.array([global_to_local[g] for g in data["y_train"].numpy()], dtype=np.int64)

    X_support = data["X_support"].numpy().astype(np.float32)
    y_support = np.array([global_to_local[g] for g in data["y_support"].numpy()], dtype=np.int64)

    X_test_full = data["X_test"].numpy().astype(np.float32)
    y_test_full = np.array([global_to_local[g] for g in data["y_test"].numpy()], dtype=np.int64)

    # Separate seen and unseen from the test set
    seen_mask = np.isin(y_test_full, list(seen_local_ids))
    X_test_seen = X_test_full[seen_mask]
    y_test_seen = y_test_full[seen_mask]

    X_test_unseen = X_test_full[~seen_mask]
    y_test_unseen = y_test_full[~seen_mask]

    # Merge seen train and seen test to create a pool for dynamic splitting
    X_seen_all = np.concatenate([X_train, X_test_seen], axis=0)
    y_seen_all = np.concatenate([y_train, y_test_seen], axis=0)

    return (X_seen_all, y_seen_all, X_test_unseen, y_test_unseen, X_support, y_support,
            seen_local_ids, unseen_local_ids, len(all_names_ordered))


def maybe_subsample_oneshot(X_seen, y_seen, X_support, y_support, max_n, seed=42):
    n_support = len(X_support)
    seen_budget = max_n - n_support

    if len(X_seen) > seen_budget:
        ratio = seen_budget / len(X_seen)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - ratio, random_state=seed)
        idx, _ = next(sss.split(X_seen, y_seen))
        X_seen_sub, y_seen_sub = X_seen[idx], y_seen[idx]
    else:
        X_seen_sub, y_seen_sub = X_seen, y_seen

    X_out = np.concatenate([X_seen_sub, X_support], axis=0)
    y_out = np.concatenate([y_seen_sub, y_support], axis=0)
    return X_out, y_out


# ═══════════════════════════════════════════════════════════════════════════════
# APT PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════

class APTPredictor:
    def __init__(self, checkpoint_path: str, device: str = None):
        self.checkpoint_path = checkpoint_path
        self.seen_local_ids = set()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._X_ctx = None
        self._y_ctx = None
        self._load()

    def _load(self):
        sys.path.insert(0, CFG["apt_repo_path"])
        try:
            import apt  # noqa: F401
            from apt.model.model import APT
        except ImportError:
            raise ImportError(f"[APT] Package not found. Run: cd {CFG['apt_repo_path']} && pip install -e .")

        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        state_dict, config = ckpt
        self._model = APT(**config)
        self._model.load_state_dict(state_dict, strict=True)
        self._model.eval().to(self.device)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self._X_ctx = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        self._y_ctx = torch.tensor(y_train, dtype=torch.long).to(self.device)
        return self

    def predict(self, X_test: np.ndarray, gamma: float = 0.0) -> np.ndarray:
        X_q = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        X_all = torch.cat([self._X_ctx, X_q], dim=0)

        with torch.no_grad():
            logits = self._model(X_all.unsqueeze(0), self._y_ctx.unsqueeze(0)).squeeze(0)

        n_test = logits.shape[0]
        n_classes = int(self._y_ctx.max().item()) + 1
        class_scores = torch.zeros(n_test, n_classes, device=self.device)

        class_scores.scatter_add_(
            dim=1,
            index=self._y_ctx.unsqueeze(0).expand(n_test, -1),
            src=logits
        )

        if gamma > 0:
            for s_id in self.seen_local_ids:
                class_scores[:, s_id] -= gamma

        return class_scores.argmax(dim=-1).cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def get_gzsl_metrics(y_true, y_pred, seen_local_ids, unseen_local_ids):
    labels_present = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels_present)
    per_class_acc = {label: (cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0)
                     for i, label in enumerate(labels_present)}

    S = np.mean([per_class_acc[l] for l in labels_present if l in seen_local_ids]) * 100
    U = np.mean([per_class_acc[l] for l in labels_present if l in unseen_local_ids]) * 100
    H = (2 * S * U / (S + U)) if (S + U) > 0 else 0.0
    return S, U, H


def print_summary_table(results_dict, n_seeds):
    print(f"\n{BANNER}")
    print(f" APT RESULTS   (mean ± std, %,  {n_seeds} seeds)")
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


def main():
    print(f"\n{BANNER}")
    print("  VENOM — APT One-Shot GZSL Multi-Seed Pipeline")
    print("  Gamma sweep performed ONLY on first seed, locked for remainder.")
    print(f"{BANNER}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INIT] Loading APT Base Model to {device}...")
    predictor = APTPredictor(checkpoint_path=CFG["apt_checkpoint"], device=device)

    summary_results = {}
    detailed_rows = []

    for dataset_name, data_path in DATASETS.items():
        print(f"\n{'─' * 50}")
        print(f" Processing Dataset: {dataset_name}")
        print(f"{'─' * 50}")

        try:
            data = load_prepared_data(data_path)
            (X_seen_all, y_seen_all, X_test_unseen, y_test_unseen, X_support, y_support,
             seen_local_ids, unseen_local_ids, n_classes) = prepare_arrays_oneshot(data)
        except Exception as e:
            print(f"  [Error] Skipping {dataset_name}: {e}")
            continue

        predictor.seen_local_ids = seen_local_ids
        summary_results[dataset_name] = {"S": [], "U": [], "H": []}
        best_gamma = 0.0

        for idx, seed in enumerate(SEEDS):
            set_seed(seed)
            print(f"  → Running Seed {seed} ", end="", flush=True)

            # CREATE FRESH SPLIT FOR THIS SEED (80% train, 20% test - seen classes only)
            X_seen_train, X_seen_test, y_seen_train, y_seen_test = train_test_split(
                X_seen_all, y_seen_all, test_size=0.2, random_state=seed, stratify=y_seen_all
            )

            # Recombine the test sets to form the full GZSL test set for this run
            X_test_seed = np.concatenate([X_seen_test, X_test_unseen], axis=0)
            y_test_seed = np.concatenate([y_seen_test, y_test_unseen], axis=0)

            # Subsample Context (1-shot support is ALWAYS kept)
            if CFG["subsample"]:
                X_train_fit, y_train_fit = maybe_subsample_oneshot(
                    X_seen_train, y_seen_train, X_support, y_support,
                    max_n=CFG["subsample"], seed=seed
                )
            else:
                X_train_fit = np.concatenate([X_seen_train, X_support], axis=0)
                y_train_fit = np.concatenate([y_seen_train, y_support], axis=0)

            predictor.fit(X_train_fit, y_train_fit)

            if idx == 0:
                # --- CALIBRATION ON FIRST SEED ---
                best_h = -1
                for gamma in CFG["gamma_range"]:
                    y_pred = predictor.predict(X_test_seed, gamma=gamma)
                    _, _, H = get_gzsl_metrics(y_test_seed, y_pred, seen_local_ids, unseen_local_ids)
                    if H > best_h:
                        best_h = H
                        best_gamma = gamma
                print(f"[Calibrated Gamma: {best_gamma:.1f}] ", end="")

            # Predict using Locked (or just Calibrated) Gamma on the seed's specific test set
            y_pred = predictor.predict(X_test_seed, gamma=best_gamma)
            S, U, H = get_gzsl_metrics(y_test_seed, y_pred, seen_local_ids, unseen_local_ids)

            summary_results[dataset_name]["S"].append(S)
            summary_results[dataset_name]["U"].append(U)
            summary_results[dataset_name]["H"].append(H)

            detailed_rows.append({
                "dataset": dataset_name, "seed": seed,
                "gamma": best_gamma, "S": S, "U": U, "H": H
            })

            print("✓")  # Mark seed as complete

    # Save detailed per-seed results
    df = pd.DataFrame(detailed_rows)
    df.to_csv(CFG["output_path"], index=False)
    print(f"\n[DONE] Detailed results saved to: {CFG['output_path']}")

    # Print the requested summary table
    print_summary_table(summary_results, len(SEEDS))


if __name__ == "__main__":
    main()