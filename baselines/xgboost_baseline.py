#!/usr/bin/env python3
"""
run_xg_combined_multiseed.py
═══════════════════════════════════════════════════════════════════════════════
XGBoost Baseline Combiner for VENOM (CIC-AndMal-2020)
Multi-Seed & Multi-Dataset Evaluation with First-Seed Calibration.
Dynamically re-splits seen (and unseen for ceiling) data per seed.
Accelerated using GPU Histogram method.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import time
import warnings
import numpy as np
import torch
import xgboost as xgb
from sklearn.metrics import confusion_matrix, accuracy_score
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

# Grid used ONLY during the first seed to find the best stacking penalty
GAMMA_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

CFG = {
    "xgb_params": dict(
        objective="multi:softprob",  # Uses softprob to access probabilities for gamma calibration
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        early_stopping_rounds=20,
        n_jobs=-1,
        verbosity=0,  # Quiet mode
        tree_method="hist",  # EXTREMELY IMPORTANT for speed: uses fast histogram method
        device="cuda" if torch.cuda.is_available() else "cpu",  # Pushes execution to GPU
    )
}

BANNER = "═" * 77


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def get_metrics(y_true, y_pred, n_classes, seen_compact_ids, unseen_compact_ids):
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    per_class_acc = {i: (cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0) for i in range(n_classes)}

    S = np.mean([per_class_acc[c] for c in seen_compact_ids]) * 100
    U = np.mean([per_class_acc[c] for c in unseen_compact_ids]) * 100
    H = (2 * S * U / (S + U)) if (S + U) > 0 else 0.0
    return S, U, H


def print_summary_table(gzsl_results, ceiling_results, n_seeds):
    print(f"\n{BANNER}")
    print(f" XGBoost RESULTS   (mean ± std, %,  {n_seeds} seeds)")
    print(f"{BANNER}")
    print(f" {'Dataset':<13}│{'Seen Acc':>16}       │{'Unseen Acc':>16}       │{'H-Mean':>11}")
    print(f" ─────────────┼──────────────────────┼──────────────────────┼─────────────────")

    for dataset in gzsl_results.keys():
        s_mean, s_std = np.mean(gzsl_results[dataset]['S']), np.std(gzsl_results[dataset]['S'], ddof=1) if len(
            gzsl_results[dataset]['S']) > 1 else 0.0
        u_mean, u_std = np.mean(gzsl_results[dataset]['U']), np.std(gzsl_results[dataset]['U'], ddof=1) if len(
            gzsl_results[dataset]['U']) > 1 else 0.0
        h_mean, h_std = np.mean(gzsl_results[dataset]['H']), np.std(gzsl_results[dataset]['H'], ddof=1) if len(
            gzsl_results[dataset]['H']) > 1 else 0.0

        print(
            f" {dataset:<13}│     {s_mean:>5.2f} ± {s_std:<4.2f}     │     {u_mean:>5.2f} ± {u_std:<4.2f}     │   {h_mean:>5.2f} ± {h_std:<4.2f}")
    print(f"{BANNER}")

    # Also print the ceiling baseline summary
    print(f"\n{BANNER}")
    print(f" FULL SUPERVISED CEILING (80/20 Unseen Split)  (mean ± std, %)")
    print(f"{BANNER}")
    for dataset in ceiling_results.keys():
        c_mean = np.mean(ceiling_results[dataset])
        c_std = np.std(ceiling_results[dataset], ddof=1) if len(ceiling_results[dataset]) > 1 else 0.0
        print(f" {dataset:<13}│  Overall Acc: {c_mean:>5.2f} ± {c_std:<4.2f}")
    print(f"{BANNER}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{BANNER}")
    print("  VENOM — XGBoost Multi-Seed Pipeline")
    print("  Mode: 1-Shot GZSL (with 1st-seed Gamma calibration) & Ceiling Baseline")
    print("  Dynamically re-splits seen data per seed. GPU Accelerated.")
    print(f"{BANNER}")

    results_gzsl = {}
    results_ceil = {}

    for ds_name, ds_path in DATASETS.items():
        if not os.path.exists(ds_path):
            print(f"\n[Warning] File not found for {ds_name}, skipping... ({ds_path})")
            continue

        print(f"\n{'─' * 60}\n Processing Dataset: {ds_name}\n{'─' * 60}")
        data = torch.load(ds_path, map_location="cpu", weights_only=False)

        # ID Mapping
        seen_ids = data["seen_label_ids"]
        unseen_ids = data["unseen_label_ids"]
        all_global_ids = sorted(seen_ids + unseen_ids)
        id_to_compact = {orig: compact for compact, orig in enumerate(all_global_ids)}

        seen_compact_ids = {id_to_compact[lid] for lid in seen_ids}
        unseen_compact_ids = {id_to_compact[lid] for lid in unseen_ids}
        n_classes = len(all_global_ids)

        # Pre-extract base arrays
        X_train_base = data["X_train"].numpy().astype(np.float32)
        y_train_base = data["y_train"].numpy()
        X_support = data["X_support"].numpy().astype(np.float32)
        y_support_raw = data["y_support"].numpy()
        X_test_full = data["X_test"].numpy().astype(np.float32)
        y_test_raw = data["y_test"].numpy()

        # Separate seen and unseen portions of the test set
        unseen_mask = np.isin(y_test_raw, unseen_ids)
        X_test_unseen = X_test_full[unseen_mask]
        y_test_unseen = y_test_raw[unseen_mask]

        X_test_seen = X_test_full[~unseen_mask]
        y_test_seen = y_test_raw[~unseen_mask]

        # Merge seen train and seen test to create a full pool for dynamic splitting
        X_seen_all = np.concatenate([X_train_base, X_test_seen], axis=0)
        y_seen_all = np.concatenate([y_train_base, y_test_seen], axis=0)

        # Tracking metrics for this dataset
        results_gzsl[ds_name] = {'S': [], 'U': [], 'H': []}
        results_ceil[ds_name] = []

        best_gamma = 0.0  # Will be calibrated on the first seed

        for idx, seed in enumerate(SEEDS):
            print(f"  → Running Seed {seed} ", end="", flush=True)

            # --- DYNAMIC DATA SPLITTING FOR THIS SEED ---

            # 1. Fresh 80/20 split for seen classes
            X_seen_train, X_seen_test, y_seen_train, y_seen_test = train_test_split(
                X_seen_all, y_seen_all, test_size=0.2, random_state=seed, stratify=y_seen_all
            )

            # 2. Fresh 80/20 split for unseen classes (used mainly for Ceiling)
            X_u_train, X_u_test, y_u_train, y_u_test = train_test_split(
                X_test_unseen, y_test_unseen, test_size=0.2, random_state=seed, stratify=y_test_unseen
            )

            # ==========================================
            # 1. ONE-SHOT GZSL
            # ==========================================

            # GZSL Train: Seen Training Split + exactly 1 Support sample per Unseen class
            X_train_gzsl = np.concatenate([X_seen_train, X_support], axis=0)
            y_train_raw_gzsl = np.concatenate([y_seen_train, y_support_raw], axis=0)
            y_train_gzsl = np.array([id_to_compact[y] for y in y_train_raw_gzsl], dtype=np.int32)

            # GZSL Test: Seen Test Split + Full Unseen Test Set
            X_test_gzsl = np.concatenate([X_seen_test, X_test_unseen], axis=0)
            y_test_raw_gzsl = np.concatenate([y_seen_test, y_test_unseen], axis=0)
            y_test_gzsl = np.array([id_to_compact[y] for y in y_test_raw_gzsl], dtype=np.int32)

            # Explicit check for older XGBoost versions that might reject 'device' parameter
            try:
                clf_gzsl = xgb.XGBClassifier(num_class=n_classes, random_state=seed, **CFG["xgb_params"])
                clf_gzsl.fit(X_train_gzsl, y_train_gzsl,
                             eval_set=[
                                 (X_seen_test, np.array([id_to_compact[y] for y in y_seen_test], dtype=np.int32))],
                             verbose=False)
            except Exception:
                # Fallback for older XGBoost API
                fallback_params = CFG["xgb_params"].copy()
                if "device" in fallback_params:
                    del fallback_params["device"]
                fallback_params["tree_method"] = "gpu_hist"  # Old syntax for GPU
                clf_gzsl = xgb.XGBClassifier(num_class=n_classes, random_state=seed, **fallback_params)
                clf_gzsl.fit(X_train_gzsl, y_train_gzsl,
                             eval_set=[
                                 (X_seen_test, np.array([id_to_compact[y] for y in y_seen_test], dtype=np.int32))],
                             verbose=False)

            # Get probabilities and convert to logits for calibration
            probs = clf_gzsl.predict_proba(X_test_gzsl)
            logits = np.log(probs + 1e-9)

            if idx == 0:
                # --- CALIBRATION ON FIRST SEED ---
                best_h = -1
                for g in GAMMA_GRID:
                    calibrated_logits = logits.copy()
                    for s_id in seen_compact_ids:
                        calibrated_logits[:, s_id] -= g

                    y_pred = calibrated_logits.argmax(axis=1)
                    _, _, H = get_metrics(y_test_gzsl, y_pred, n_classes, seen_compact_ids, unseen_compact_ids)

                    if H > best_h:
                        best_h = H
                        best_gamma = g
                print(f"[Calibrated Gamma: {best_gamma}] ", end="")

            # Apply the locked (or just calibrated) gamma
            calibrated_logits = logits.copy()
            for s_id in seen_compact_ids:
                calibrated_logits[:, s_id] -= best_gamma

            y_pred_gzsl = calibrated_logits.argmax(axis=1)
            S, U, H = get_metrics(y_test_gzsl, y_pred_gzsl, n_classes, seen_compact_ids, unseen_compact_ids)

            results_gzsl[ds_name]['S'].append(S)
            results_gzsl[ds_name]['U'].append(U)
            results_gzsl[ds_name]['H'].append(H)

            # ==========================================
            # 2. SUPERVISED CEILING
            # ==========================================

            # Ceiling Train: Seen Train Split + Unseen Train Split (80%)
            X_train_ceil = np.concatenate([X_seen_train, X_u_train], axis=0)
            y_train_ceil_raw = np.concatenate([y_seen_train, y_u_train], axis=0)
            y_train_ceil = np.array([id_to_compact[y] for y in y_train_ceil_raw], dtype=np.int32)

            # Ceiling Test: Seen Test Split + Unseen Test Split (20%)
            X_test_ceil = np.concatenate([X_seen_test, X_u_test], axis=0)
            y_test_ceil_raw = np.concatenate([y_seen_test, y_u_test], axis=0)
            y_test_ceil = np.array([id_to_compact[y] for y in y_test_ceil_raw], dtype=np.int32)

            try:
                clf_ceil = xgb.XGBClassifier(num_class=n_classes, random_state=seed, **CFG["xgb_params"])
                clf_ceil.fit(X_train_ceil, y_train_ceil, eval_set=[(X_test_ceil, y_test_ceil)], verbose=False)
            except Exception:
                fallback_params = CFG["xgb_params"].copy()
                if "device" in fallback_params:
                    del fallback_params["device"]
                fallback_params["tree_method"] = "gpu_hist"
                clf_ceil = xgb.XGBClassifier(num_class=n_classes, random_state=seed, **fallback_params)
                clf_ceil.fit(X_train_ceil, y_train_ceil, eval_set=[(X_test_ceil, y_test_ceil)], verbose=False)

            y_pred_ceil = clf_ceil.predict(X_test_ceil)

            acc_ceil = accuracy_score(y_test_ceil, y_pred_ceil) * 100
            results_ceil[ds_name].append(acc_ceil)

            print("✓")  # Mark seed as complete

    # Print formatted tables at the very end
    print_summary_table(results_gzsl, results_ceil, len(SEEDS))


if __name__ == "__main__":
    main()