#!/usr/bin/env python3
"""
run_tabPFN_ceiling_multiseed.py
═══════════════════════════════════════════════════════════════════════════════
TabPFN v2 / v2.5 — Fully Supervised Ceiling Setting (Seen Classes Only)
Multi-Seed, Multi-Dataset Evaluator.
Dynamically re-splits seen train/test data per seed for valid variance.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import time
import random
import warnings

import numpy as np
import torch
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DATASETS = {
    "APIGRAPH": "/home/iamontoyasa/02_VENOM/03_Baseline/04_run1_prepared_data_zero_shot.pt",
    "Goodread": "/home/iamontoyasa/02_VENOM/03_Baseline/06_run1_prepared_data_zero_shot.pt",
    "Petfinder": "/home/iamontoyasa/02_VENOM/03_Baseline/07_run1_prepared_data_zero_shot.pt",
    "Fakeddit": "/home/iamontoyasa/02_VENOM/03_Baseline/08_run1_prepared_data_zero_shot.pt"
}
# DATASETS = {
#     "AndMAL": "/home/iamontoyasa/02_VENOM/03_Baseline/01_run1_prepared_data_zero_shot.pt",
#     "BODMAS": "/home/iamontoyasa/02_VENOM/03_Baseline/02_run1_prepared_data_zero_shot.pt",
#     "AVASTCTU": "/home/iamontoyasa/02_VENOM/03_Baseline/03_run1_prepared_data_zero_shot.pt",
#     "APIGRAPH": "/home/iamontoyasa/02_VENOM/03_Baseline/04_run1_prepared_data_zero_shot.pt",
#     "Goodread": "/home/iamontoyasa/02_VENOM/03_Baseline/06_run1_prepared_data_zero_shot.pt",
#     "Petfinder": "/home/iamontoyasa/02_VENOM/03_Baseline/07_run1_prepared_data_zero_shot.pt",
#     "Fakeddit": "/home/iamontoyasa/02_VENOM/03_Baseline/08_run1_prepared_data_zero_shot.pt"
# }
SEEDS = [42, 123, 456, 789, 2025]

CFG = {
    "output_path": "/home/iamontoyasa/02_VENOM/03_Baseline/ceiling_tabpfn_results_multiseed.csv",
    "tabpfn_version": "v2",
    "subsample": 10_000,
}

TABPFN_CLASS_LIMIT = 10
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


def build_label_mapping(seen_label_ids: list, all_class_names: list):
    sorted_names = sorted(all_class_names)
    sorted_seen = sorted(seen_label_ids)
    global_to_local = {g: l for l, g in enumerate(sorted_seen)}
    local_to_name = {l: sorted_names[g] for g, l in global_to_local.items()}
    return global_to_local, local_to_name


def filter_seen_test(X_test: torch.Tensor, y_test: torch.Tensor, unseen_label_ids: list):
    unseen_set = set(unseen_label_ids)
    mask = torch.tensor([y.item() not in unseen_set for y in y_test], dtype=torch.bool)
    return X_test[mask], y_test[mask]


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

    # MERGE train and test for seen classes to dynamically re-split them per seed
    X_seen_all = np.concatenate([X_train, X_test_seen], axis=0)
    y_seen_all = np.concatenate([y_train_local, y_test_local], axis=0)

    n_classes = len(seen_label_ids)
    return X_seen_all, y_seen_all, local_to_name, n_classes


def maybe_subsample(X: np.ndarray, y: np.ndarray, max_n: int, seed: int = 42) -> tuple:
    if len(X) <= max_n:
        return X, y
    ratio = max_n / len(X)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - ratio, random_state=seed)
    idx, _ = next(sss.split(X, y))
    return X[idx], y[idx]


# ═══════════════════════════════════════════════════════════════════════════════
# TABPFN BASELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_tabpfn(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, n_classes: int, version: str,
               seed: int) -> np.ndarray:
    from tabpfn import TabPFNClassifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "true")

    # Instantiate base TabPFN
    if version == "v2":
        try:
            from tabpfn.constants import ModelVersion
            base_clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device=device,
                                                                   ignore_pretraining_limits=True)
        except Exception:
            base_clf = TabPFNClassifier(device=device, ignore_pretraining_limits=True)
    else:
        base_clf = TabPFNClassifier(device=device, ignore_pretraining_limits=True)

    # ManyClassClassifier when n_classes > 10
    if n_classes > TABPFN_CLASS_LIMIT:
        from tabpfn_extensions.many_class import ManyClassClassifier
        clf = ManyClassClassifier(
            estimator=base_clf,
            alphabet_size=getattr(base_clf, "max_num_classes_", 10),
            n_estimators_redundancy=4,
            random_state=seed,
            verbose=0,
        )
    else:
        clf = base_clf

    clf.fit(X_train, y_train)
    return clf.predict(X_test)


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS & MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary_table(results_dict, n_seeds):
    print(f"\n{BANNER}")
    print(f" TabPFN CEILING RESULTS   (mean ± std, %,  {n_seeds} seeds)")
    print(f"{BANNER}")
    print(f" {'Dataset':<13} │ {'Seen Acc':>16}       │ {'Unseen Acc':>16}       │ {'H-Mean':>11}")
    print(f" ──────────────┼────────────────────────┼────────────────────────┼─────────────────")

    for dataset, metrics in results_dict.items():
        mean_seen = np.mean(metrics["seen_accs"])
        std_seen = np.std(metrics["seen_accs"], ddof=1) if len(metrics["seen_accs"]) > 1 else 0.0

        # Unseen and H-Mean are N/A for Supervised Ceiling
        print(f" {dataset:<13} │     {mean_seen:>5.2f} ± {std_seen:<4.2f}       │          N/A           │       N/A")
    print(f"{BANNER}\n")


def main():
    print(f"\n{BANNER}")
    print(f"  VENOM — TabPFN {CFG['tabpfn_version']} Ceiling Baseline (Seen Classes Only)")
    print("  Dynamically re-splits seen data per seed.")
    print(f"{BANNER}")

    summary_results = {}
    detailed_rows = []

    for dataset_name, data_path in DATASETS.items():
        print(f"\n{'─' * 50}")
        print(f" Processing Dataset: {dataset_name}")
        print(f"{'─' * 50}")

        try:
            data = load_prepared_data(data_path)
            X_seen_all, y_seen_all, local_to_name, n_classes = prepare_arrays(data)
        except Exception as e:
            print(f"  [Error] Skipping {dataset_name}: {e}")
            continue

        summary_results[dataset_name] = {"seen_accs": []}

        for seed in SEEDS:
            set_seed(seed)
            print(f"  → Running Seed {seed} ... ", end="", flush=True)

            # CREATE FRESH SPLIT FOR THIS SEED (80% train, 20% test - seen classes only)
            X_train, X_test, y_train, y_test = train_test_split(
                X_seen_all, y_seen_all, test_size=0.2, random_state=seed, stratify=y_seen_all
            )

            # Subsample Context
            if CFG["subsample"] and len(X_train) > CFG["subsample"]:
                X_train_fit, y_train_fit = maybe_subsample(X_train, y_train, max_n=CFG["subsample"], seed=seed)
            else:
                X_train_fit, y_train_fit = X_train, y_train

            # Run TabPFN
            t0 = time.time()
            y_pred = run_tabpfn(X_train_fit, y_train_fit, X_test, n_classes, CFG["tabpfn_version"], seed)

            # Evaluate Seen Accuracy
            acc_seen = accuracy_score(y_test, y_pred) * 100.0

            summary_results[dataset_name]["seen_accs"].append(acc_seen)

            detailed_rows.append({
                "dataset": dataset_name, "seed": seed,
                "seen_acc": acc_seen, "elapsed_s": round(time.time() - t0, 1)
            })

            print(f"✓ (Seen Acc: {acc_seen:.1f}%)")  # Mark seed as complete

    # Save detailed per-seed results
    if detailed_rows:
        df = pd.DataFrame(detailed_rows)
        df.to_csv(CFG["output_path"], index=False)
        print(f"\n[DONE] Detailed results saved to: {CFG['output_path']}")

    # Print the requested summary table
    print_summary_table(summary_results, len(SEEDS))


if __name__ == "__main__":
    main()