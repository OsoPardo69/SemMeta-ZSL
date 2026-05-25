"""
run_updated.py — Sequential baseline runner  (final version)
=============================================================
Rename to run_baselines.py before running.

Prerequisite: python data_prep.py  →  generates prepared_data.pt
Usage:        python run_baselines.py
"""

import time
import traceback
import numpy as np
import pandas as pd
import torch

# ── Config ────────────────────────────────────────────────────
PREPARED_DATA_PATH = "prepared_data.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

print(f"\nLoading prepared data from '{PREPARED_DATA_PATH}'...")
data = torch.load(PREPARED_DATA_PATH, weights_only=False)
print(f"  X_train : {data['X_train'].shape}")
print(f"  X_test  : {data['X_test'].shape}")
print(f"  d_in    : {data['d_in']}")
print("Data ready.\n")

results = {}


# ═══════════════════════════════════════════════════════
# BASELINE 1 — FL-ZSL-Feedback
# ═══════════════════════════════════════════════════════
print("=" * 55)
print("BASELINE 1 / 3 — FL-ZSL-Feedback")
print("=" * 55)
try:
    from fl_zsl_feedback import FLZSLFeedback

    fl_model = FLZSLFeedback(
        d_in                 = data["d_in"],
        n_seen               = len(data["seen_label_ids"]),
        n_total              = len(data["all_class_names"]),
        class_descriptions   = data["class_descriptions"],
        seen_class_names     = data["seen_class_names"],
        all_class_names      = data["all_class_names"],
        seen_label_ids       = data["seen_label_ids"],
        n_clients            = 3,
        hidden               = 128,
        dropout              = 0.4,
        proto_weight         = 0.5,
        confidence_threshold = 0.5,
        device               = DEVICE,
    )

    t0 = time.time()
    fl_model.fit(
        data["X_train"], data["y_train"],
        n_rounds=10, local_epochs=5, batch_size=64, lr=1e-3,
    )
    print(f"  Training time: {time.time() - t0:.1f}s")

    fl_res = fl_model.evaluate_gzsl(
        data["X_test"], data["y_test"],
        data["seen_label_ids"], data["unseen_label_ids"],
    )
    results["FL-ZSL-Feedback"] = fl_res
    print(f"  Acc Seen   : {fl_res['acc_seen']  * 100:.2f}%")
    print(f"  Acc Unseen : {fl_res['acc_unseen'] * 100:.2f}%")
    print(f"  H-Mean     : {fl_res['H_mean']     * 100:.2f}%")

except Exception:
    print("  [FAILED] FL-ZSL-Feedback:")
    traceback.print_exc()
    results["FL-ZSL-Feedback"] = {"acc_seen": None, "acc_unseen": None, "H_mean": None}


# ═══════════════════════════════════════════════════════
# BASELINE 2 — RoboGuardZ
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("BASELINE 2 / 3 — RoboGuardZ")
print("=" * 55)
try:
    from roboguardz import RoboGuardZ

    rg_model = RoboGuardZ(
        seq_len        = data["d_in"],
        n_seen         = len(data["seen_label_ids"]),
        n_filters      = 22,
        kernel_sizes   = (3, 5, 8),
        chi_gamma      = 1.0,
        n_estimators   = 200,
        val_ratio      = 0.2,
        seen_retention = 0.95,
        cnn_epochs     = 20,
        device         = DEVICE,
    )

    t0 = time.time()
    rg_model.fit(data["X_train"].numpy(), data["y_train"].numpy())
    print(f"  Training time: {time.time() - t0:.1f}s")

    rg_res = rg_model.evaluate_gzsl(
        data["X_test"].numpy(), data["y_test"].numpy(),
        data["seen_label_ids"], data["unseen_label_ids"],
    )
    results["RoboGuardZ"] = rg_res
    print(f"  Acc Seen : {rg_res['acc_seen'] * 100:.2f}%")
    print(f"  OOD Rate : {rg_res['ood_rate']  * 100:.2f}%  "
          f"(unseen flagged as unknown — not family classification)")
    print(f"  H-Mean   : {rg_res['H_mean']    * 100:.2f}%")

except Exception:
    print("  [FAILED] RoboGuardZ:")
    traceback.print_exc()
    results["RoboGuardZ"] = {"acc_seen": None, "ood_rate": None, "H_mean": None}


# ═══════════════════════════════════════════════════════
# BASELINE 3 — TZSL
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("BASELINE 3 / 3 — TZSL")
print("=" * 55)
try:
    from tzsl import TZSL

    tzsl_model = TZSL(
        d_in             = data["d_in"],
        class_attrs      = data["class_attrs"],
        seen_class_ids   = data["seen_label_ids"],
        unseen_class_ids = data["unseen_label_ids"],
        hidden_dim       = 256,
        embedding_dim    = 64,
        n_embeddings     = 512,
        gcn_hidden       = 128,
        kg_threshold     = 0.3,
        device           = DEVICE,
    )

    t0 = time.time()

    # Phase 1: inductive training on seen data
    tzsl_model.fit(
        data["X_train"], data["y_train"],
        epochs=150, lr=1e-3, batch_size=128,
    )
    print(f"  Phase 1 time: {time.time() - t0:.1f}s")

    # Phase 2: transductive fine-tuning on unlabeled test features
    t1 = time.time()
    tzsl_model.transductive_finetune(
        data["X_test"], epochs=30, lr=5e-4, batch_size=128,
    )
    print(f"  Phase 2 time: {time.time() - t1:.1f}s")
    print(f"  Total time  : {time.time() - t0:.1f}s")

    tz_res = tzsl_model.evaluate_gzsl(data["X_test"], data["y_test"])
    results["TZSL"] = tz_res
    print(f"  Acc Seen   : {tz_res['acc_seen']  * 100:.2f}%")
    print(f"  Acc Unseen : {tz_res['acc_unseen'] * 100:.2f}%")
    print(f"  H-Mean     : {tz_res['H_mean']     * 100:.2f}%")

except Exception:
    print("  [FAILED] TZSL:")
    traceback.print_exc()
    results["TZSL"] = {"acc_seen": None, "acc_unseen": None, "H_mean": None}


# ═══════════════════════════════════════════════════════
# FINAL COMPARISON TABLE
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("GZSL BENCHMARK — CIC-AndMal-2020")
print("=" * 55)

def fmt(v):
    return f"{v * 100:.2f}%" if v is not None else "FAILED"

rows = []
for model_name, res in results.items():
    if model_name == "RoboGuardZ":
        unseen_val  = res.get("ood_rate")
        unseen_type = "OOD detection†"
    else:
        unseen_val  = res.get("acc_unseen")
        unseen_type = "Family classification"

    rows.append({
        "Model":         model_name,
        "Acc Seen (S)":  fmt(res.get("acc_seen")),
        "Unseen Metric": fmt(unseen_val),
        "H-Mean":        fmt(res.get("H_mean")),
        "Unseen Type":   unseen_type,
    })

df = pd.DataFrame(rows).set_index("Model")
print(df.to_string())
print("\n† RoboGuardZ flags unseen samples as \'unknown\' — not family classification.")

# ── Save CSV ──────────────────────────────────────────
numeric_rows = []
for model_name, res in results.items():
    unseen_val = res.get("ood_rate") if model_name == "RoboGuardZ"                  else res.get("acc_unseen")
    numeric_rows.append({
        "Model":         model_name,
        "Acc_Seen":      round(res["acc_seen"]  * 100, 2) if res.get("acc_seen")  is not None else None,
        "Unseen_Metric": round(unseen_val        * 100, 2) if unseen_val           is not None else None,
        "H_Mean":        round(res["H_mean"]     * 100, 2) if res.get("H_mean")   is not None else None,
        "Unseen_Type":   "OOD_detection" if model_name == "RoboGuardZ"
                         else "Family_classification",
    })

pd.DataFrame(numeric_rows).set_index("Model").to_csv("baseline_results.csv")
print("\nSaved to baseline_results.csv")
