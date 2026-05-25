#!/usr/bin/env python3
"""
run_APT_multiseed.py
═══════════════════════════════════════════════════════════════════════════════
APT — Adversarially Pre-trained Transformer (Wu & Bergman, ICML 2025)
Setting : FULLY SUPERVISED CEILING — seen classes only
          Upper-bound reference for GZSL comparisons. No H-mean reported.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import time
import random
import warnings

import numpy as np
import torch
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DATASETS = {
    "AndMAL": "/home/iamontoyasa/02_VENOM/03_Baseline/01_run1_prepared_data_zero_shot.pt",
    "BODMAS": "/home/iamontoyasa/02_VENOM/03_Baseline/02_run1_prepared_data_zero_shot.pt",
    "AVASTCTU": "/home/iamontoyasa/02_VENOM/03_Baseline/03_run1_prepared_data_zero_shot.pt",
    "APIGRAPH": "/home/iamontoyasa/02_VENOM/03_Baseline/04_run1_prepared_data_zero_shot.pt",
    "Goodread": "/home/iamontoyasa/02_VENOM/03_Baseline/06_run1_prepared_data_zero_shot.pt",
    "Petfinder": "/home/iamontoyasa/02_VENOM/03_Baseline/07_run1_prepared_data_zero_shot.pt",
    "Fakeddit": "/home/iamontoyasa/02_VENOM/03_Baseline/08_run1_prepared_data_zero_shot.pt"
}

SEEDS = [42, 123, 456, 789, 2025]

CFG = {
    "output_path": "/home/iamontoyasa/02_VENOM/03_Baseline/apt_ceiling_results_multiseed.csv",
    "apt_checkpoint": "/home/iamontoyasa/02_VENOM/03_Baseline/apt/artifacts/saves/apt_best.pt",
    "apt_repo_path": "/home/iamontoyasa/02_VENOM/03_Baseline/apt",
    "subsample": 10_000,
}

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


def load_prepared_data(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"[Data] File not found: '{path}'")
    return torch.load(path, weights_only=False)


def build_label_mapping(seen_label_ids: list, all_class_names: list):
    sorted_names = sorted(all_class_names)
    sorted_seen = sorted(seen_label_ids)
    global_to_local = {g: l for l, g in enumerate(sorted_seen)}
    local_to_name = {l: sorted_names[g] for g, l in global_to_local.items()}
    return global_to_local, local_to_name


def filter_seen_test(X_test, y_test, unseen_label_ids):
    unseen_set = set(unseen_label_ids)
    mask = torch.tensor([y.item() not in unseen_set for y in y_test], dtype=torch.bool)
    return X_test[mask], y_test[mask]


def maybe_subsample(X, y, max_n, seed=42):
    if len(X) <= max_n:
        return X, y
    ratio = max_n / len(X)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - ratio, random_state=seed)
    idx, _ = next(sss.split(X, y))
    return X[idx], y[idx]


def prepare_arrays(data: dict):
    seen_label_ids = data["seen_label_ids"]
    unseen_label_ids = data["unseen_label_ids"]
    all_class_names = data["all_class_names"]

    global_to_local, local_to_name = build_label_mapping(seen_label_ids, all_class_names)

    X_train = data["X_train"].numpy().astype(np.float32)
    y_train_global = data["y_train"].numpy()
    y_train_local = np.array([global_to_local[g] for g in y_train_global], dtype=np.int64)

    X_test_seen, y_test_seen_global = filter_seen_test(data["X_test"], data["y_test"], unseen_label_ids)
    X_test_seen = X_test_seen.numpy().astype(np.float32)
    y_test_local = np.array([global_to_local[g] for g in y_test_seen_global.numpy()], dtype=np.int64)

    # MERGE train and test for seen classes so we can dynamically re-split them per seed
    X_seen_all = np.concatenate([X_train, X_test_seen], axis=0)
    y_seen_all = np.concatenate([y_train_local, y_test_local], axis=0)

    n_classes = len(seen_label_ids)
    return X_seen_all, y_seen_all, n_classes


# ═══════════════════════════════════════════════════════════════════════════════
# APT PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════

class APTPredictor:
    def __init__(self, checkpoint_path: str, device: str = None):
        self.checkpoint_path = checkpoint_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._X_ctx = None
        self._y_ctx = None
        self._load()

    def _load(self):
        sys.path.insert(0, CFG["apt_repo_path"])
        try:
            import apt  # noqa: F401
        except ImportError:
            raise ImportError(f"[APT] Package not found. Run: cd {CFG['apt_repo_path']} && pip install -e .")

        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self._model = self._build_model(ckpt)
        self._model.eval()
        self._model.to(self.device)

    def _build_model(self, ckpt):
        state_dict, config = ckpt
        from apt.model.model import APT
        model = APT(**config)
        model.load_state_dict(state_dict, strict=True)
        return model

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self._X_ctx = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        self._y_ctx = torch.tensor(y_train, dtype=torch.long).to(self.device)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
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
        return class_scores.argmax(dim=-1).cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary_table(results_dict, n_seeds):
    print(f"\n{BANNER}")
    print(f" APT RESULTS   (mean ± std, %,  {n_seeds} seeds)")
    print(f"{BANNER}")
    print(f" {'Dataset':<13} │ {'Seen Acc':>16}       │ {'Unseen Acc':>16}       │ {'H-Mean':>11}")
    print(f" ──────────────┼────────────────────────┼────────────────────────┼─────────────────")

    for dataset, metrics in results_dict.items():
        mean_seen = np.mean(metrics["seen_accs"])
        std_seen = np.std(metrics["seen_accs"], ddof=1) if len(metrics["seen_accs"]) > 1 else 0.0

        # Unseen and H-Mean are N/A because this is a Fully Supervised Ceiling on seen classes
        print(f" {dataset:<13} │     {mean_seen:>5.2f} ± {std_seen:<4.2f}       │          N/A           │       N/A")
    print(f"{BANNER}\n")


def main():
    print(f"\n{BANNER}")
    print("  VENOM — APT Ceiling Baseline (Fully Supervised, Seen Classes Only)")
    print(f"  Total Datasets : {len(DATASETS)}")
    print(f"  Seeds          : {SEEDS}")
    print(BANNER)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Instantiate predictor ONCE since it's a pre-trained wrapper
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
            # Retrieve the merged seen dataset
            X_seen_all, y_seen_all, n_classes = prepare_arrays(data)
        except Exception as e:
            print(f"  [Error] Skipping {dataset_name}: {e}")
            continue

        summary_results[dataset_name] = {"seen_accs": []}

        for seed in SEEDS:
            set_seed(seed)
            print(f"  → Running Seed {seed}...")

            # CREATE FRESH SPLIT FOR THIS SEED (80% train, 20% test)
            X_train, X_test_seen, y_train, y_test_local = train_test_split(
                X_seen_all, y_seen_all, test_size=0.2, random_state=seed, stratify=y_seen_all
            )

            # Subsample per seed so the stratified split is slightly different each time
            if CFG["subsample"] and len(X_train) > CFG["subsample"]:
                X_train_fit, y_train_fit = maybe_subsample(X_train, y_train, max_n=CFG["subsample"], seed=seed)
            else:
                X_train_fit, y_train_fit = X_train, y_train

            # Scale
            scaler = StandardScaler()
            X_train_fit = scaler.fit_transform(X_train_fit)
            X_test_scaled = scaler.transform(X_test_seen)

            # Fit & Predict
            t0 = time.time()
            predictor.fit(X_train_fit, y_train_fit)
            y_pred = predictor.predict(X_test_scaled)

            # Evaluate Seen Accuracy
            acc_seen = accuracy_score(y_test_local, y_pred) * 100.0
            elapsed = time.time() - t0

            summary_results[dataset_name]["seen_accs"].append(acc_seen)
            detailed_rows.append({
                "dataset": dataset_name,
                "seed": seed,
                "seen_acc": acc_seen,
                "elapsed_s": round(elapsed, 2)
            })

    # Save detailed per-seed results
    df = pd.DataFrame(detailed_rows)
    df.to_csv(CFG["output_path"], index=False)
    print(f"\n[DONE] Detailed results saved to: {CFG['output_path']}")

    # Print the requested summary table
    print_summary_table(summary_results, len(SEEDS))


if __name__ == "__main__":
    main()