#!/usr/bin/env python3
"""
data_prep.py — Label-agnostic GZSL / One-Shot data preparation
═══════════════════════════════════════════════════════════════════
Inputs
  tabular_path    : CSV file, any headers, LAST column = label
  text_path       : JSON file, records with {id, label, description}
  unseen_families : 1-D list of family names held out for ZSL/OSL test
  drop_families   : optional list of families to remove entirely
  mode            : "zero_shot" (default) or "one_shot"
                    In one_shot mode, 1 tabular sample per unseen family
                    is carved out into a support set and added to training.

Outputs
  A single dict with every tensor needed by all baselines.
  One-shot exclusive keys (only populated when mode="one_shot"):
    X_support      : torch.float32 (n_unseen, d_in) — 1 sample per unseen class
    y_support      : torch.long    (n_unseen,)
    y_support_str  : np.ndarray    (n_unseen,)
"""

import json
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# I/O HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_tabular(tabular_path: str) -> tuple:
    """
    Reads a CSV where the LAST column is always the label.
    - Drops any row with NaN or blank in any feature column.
    - Returns X (float32), y_str (string labels), feature_names (list[str]).
    """
    df = pd.read_csv(tabular_path)
    label_col = df.columns[-1]
    feat_cols  = df.columns[:-1].tolist()

    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce")

    before = len(df)
    df     = df.dropna(subset=feat_cols)
    after  = len(df)
    print(f"  Dropped {before - after} rows with NaN/blank, "
          f"{after} rows remaining")

    X     = df[feat_cols].values.astype(np.float32)
    y_str = df[label_col].astype(str).values
    d_in  = X.shape[1]

    print(f"  Label column : {label_col}")
    print(f"  Features     : {d_in}")
    print(f"  Feature sample: {feat_cols[:5]} ... {feat_cols[-3:]}")
    print(f"  Classes found : {np.unique(y_str).tolist()}")
    return X, y_str, feat_cols


def load_text(text_path: str) -> dict:
    """
    Reads the JSON file and returns a dict mapping label → list of
    description strings. Handles both JSON array [...] and
    newline-delimited JSON objects.
    """
    with open(text_path, "r", encoding="utf-8") as f:
        try:
            records = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            records = [json.loads(line) for line in f if line.strip()]

    descriptions: dict[str, list] = {}
    for rec in records:
        lbl  = str(rec["label"])
        desc = str(rec["description"])
        descriptions.setdefault(lbl, []).append(desc)

    print(f"  Text families : {list(descriptions.keys())}")
    print(f"  Descriptions per family: "
          f"{ {k: len(v) for k, v in descriptions.items()} }")
    return descriptions


# ─────────────────────────────────────────────────────────────────────────────
# GZSL SPLIT (zero-shot)
# ─────────────────────────────────────────────────────────────────────────────

def gzsl_split(X: np.ndarray, y_str: np.ndarray,
               seen_families: list, unseen_families: list,
               test_ratio: float = 0.2, seed: int = 42) -> dict:
    """
    GZSL protocol (zero-shot):
      Train : seen classes only, (1 - test_ratio) of seen samples
      Test  : seen_test + ALL unseen samples
    Scaler fitted on train only — no leakage.
    """
    seen_mask   = np.isin(y_str, seen_families)
    unseen_mask = np.isin(y_str, unseen_families)

    X_seen,   y_seen   = X[seen_mask],   y_str[seen_mask]
    X_unseen, y_unseen = X[unseen_mask], y_str[unseen_mask]

    X_train, X_seen_test, y_train_str, y_seen_test_str = train_test_split(
        X_seen, y_seen,
        test_size=test_ratio, random_state=seed, stratify=y_seen,
    )

    X_test    = np.concatenate([X_seen_test, X_unseen], axis=0)
    y_test_str = np.concatenate([y_seen_test_str, y_unseen], axis=0)

    rng  = np.random.default_rng(seed)
    perm = rng.permutation(len(y_test_str))
    X_test, y_test_str = X_test[perm], y_test_str[perm]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    print(f"  Train : {X_train.shape} ({len(seen_families)} seen classes)")
    print(f"  Test  : {X_test.shape} "
          f"({len(seen_families)} seen + {len(unseen_families)} unseen)")

    return {
        "X_train"     : X_train,
        "y_train_str" : y_train_str,
        "X_test"      : X_test,
        "y_test_str"  : y_test_str,
        "scaler"      : scaler,
        # Support set is empty in zero-shot mode
        "X_support"    : np.empty((0, X_train.shape[1]), dtype=np.float32),
        "y_support_str": np.empty((0,), dtype=object),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ONE-SHOT SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def oneshot_split(X: np.ndarray, y_str: np.ndarray,
                  seen_families: list, unseen_families: list,
                  test_ratio: float = 0.2, seed: int = 42) -> dict:
    """
    One-shot protocol:
      Seen train  : (1 - test_ratio) of seen samples   ← same as GZSL
      Seen test   : test_ratio of seen samples          ← same as GZSL
      Support set : EXACTLY 1 sample per unseen family  ← NEW
                    (sampled before scaling, scaled with seen-fitted scaler)
      Unseen test : all remaining unseen samples        ← rest after support

    Critical design choices
    ───────────────────────
    1. Scaler is fitted ONLY on seen-train data — no leakage.
    2. Support samples are scaled with that same fitted scaler.
    3. The support set is returned separately so each baseline can decide
       how to consume it (append to X_train, fine-tune, prototype update, …).
    4. The full test set still contains seen_test + unseen_remaining
       so GZSL harmonic-mean metrics remain valid.
    """
    rng = np.random.default_rng(seed)

    # ── Seen split (unchanged from GZSL) ─────────────────────────────────────
    seen_mask = np.isin(y_str, seen_families)
    X_seen, y_seen = X[seen_mask], y_str[seen_mask]

    X_train_raw, X_seen_test_raw, y_train_str, y_seen_test_str = train_test_split(
        X_seen, y_seen,
        test_size=test_ratio, random_state=seed, stratify=y_seen,
    )

    # ── Fit scaler on seen-train ONLY ────────────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)

    # ── Unseen: carve out 1 support sample per family ────────────────────────
    X_support_list     = []
    y_support_list     = []
    X_unseen_test_list = []
    y_unseen_test_list = []

    for fam in unseen_families:
        fam_mask   = y_str == fam
        X_fam      = X[fam_mask]
        y_fam      = y_str[fam_mask]

        if len(X_fam) == 0:
            raise ValueError(f"Unseen family '{fam}' has no samples in the dataset.")
        if len(X_fam) == 1:
            # Edge case: only 1 sample — it becomes support AND test
            print(f"  WARN: '{fam}' has only 1 sample; used as support AND test.")
            support_idx   = np.array([0])
            remaining_idx = np.array([0])
        else:
            # Randomly pick exactly 1 support sample
            support_idx   = rng.choice(len(X_fam), size=1, replace=False)
            remaining_idx = np.setdiff1d(np.arange(len(X_fam)), support_idx)

        X_support_list.append(X_fam[support_idx])
        y_support_list.append(y_fam[support_idx])

        X_unseen_test_list.append(X_fam[remaining_idx])
        y_unseen_test_list.append(y_fam[remaining_idx])

    X_support_raw  = np.concatenate(X_support_list,     axis=0)  # (n_unseen, d)
    y_support_str  = np.concatenate(y_support_list,     axis=0)  # (n_unseen,)
    X_unseen_rest  = np.concatenate(X_unseen_test_list, axis=0)
    y_unseen_rest  = np.concatenate(y_unseen_test_list, axis=0)

    # ── Scale support + test with seen-trained scaler ────────────────────────
    X_support = scaler.transform(X_support_raw).astype(np.float32)

    X_seen_test = scaler.transform(X_seen_test_raw).astype(np.float32)
    X_unseen_test = scaler.transform(X_unseen_rest).astype(np.float32)

    X_test_raw  = np.concatenate([X_seen_test, X_unseen_test], axis=0)
    y_test_str  = np.concatenate([y_seen_test_str, y_unseen_rest], axis=0)

    # Shuffle test set
    perm = rng.permutation(len(y_test_str))
    X_test, y_test_str = X_test_raw[perm], y_test_str[perm]

    print(f"  Seen train   : {X_train.shape} ({len(seen_families)} classes)")
    print(f"  Support set  : {X_support.shape} (1 per unseen class — "
          f"{unseen_families})")
    print(f"  Test set     : {X_test.shape} "
          f"(seen_test + unseen_remaining)")

    return {
        "X_train"      : X_train,
        "y_train_str"  : y_train_str,
        "X_test"       : X_test,
        "y_test_str"   : y_test_str,
        "scaler"       : scaler,
        "X_support"    : X_support,       # (n_unseen, d_in) — 1 per unseen family
        "y_support_str": y_support_str,   # (n_unseen,)
    }


# ─────────────────────────────────────────────────────────────────────────────
# LABEL ENCODING
# ─────────────────────────────────────────────────────────────────────────────

def encode_labels(y_train_str: np.ndarray, y_test_str: np.ndarray,
                  seen_families: list, unseen_families: list,
                  y_support_str: np.ndarray = None) -> tuple:
    """
    Single LabelEncoder fitted on ALL families (seen + unseen) so integer
    IDs are consistent across every split.
    Also encodes y_support_str if provided (one-shot mode).
    """
    le = LabelEncoder().fit(seen_families + unseen_families)

    y_train   = le.transform(y_train_str).astype(np.int64)
    y_test    = le.transform(y_test_str).astype(np.int64)
    y_support = (le.transform(y_support_str).astype(np.int64)
                 if y_support_str is not None and len(y_support_str) > 0
                 else np.empty((0,), dtype=np.int64))

    return y_train, y_test, y_support, le


# ─────────────────────────────────────────────────────────────────────────────
# CLASS ATTRIBUTE MATRIX (mean feature vectors)
# ─────────────────────────────────────────────────────────────────────────────

def build_class_attrs_from_features(X_train: np.ndarray,
                                    y_train_str: np.ndarray,
                                    X_test: np.ndarray,
                                    y_test_str: np.ndarray,
                                    all_families: list,
                                    X_support: np.ndarray = None,
                                    y_support_str: np.ndarray = None,
                                    unseen_families: list = None,
                                    mode: str = "zero_shot") -> torch.Tensor:
    """
    Per-class mean feature vectors as class attribute matrix.

    zero_shot : unseen prototypes built from ALL unseen test samples
                (transductive — valid for ZSL, labels never used for clf)
    one_shot  : unseen prototypes built ONLY from the 1 support sample
                per unseen family (strictly inductive, no test leakage)
    """
    global_mean = X_train.mean(axis=0)

    if mode == "one_shot" and X_support is not None and len(X_support) > 0:
        # Pool: seen → train, unseen → support only
        X_all  = np.concatenate([X_train, X_support], axis=0)
        y_all  = np.concatenate([y_train_str, y_support_str], axis=0)
    else:
        # Zero-shot: pool seen train + all unseen test
        X_all = np.concatenate([X_train, X_test], axis=0)
        y_all = np.concatenate([y_train_str, y_test_str], axis=0)

    rows = []
    for fam in all_families:
        mask = y_all == fam
        vec  = X_all[mask].mean(axis=0) if mask.any() else global_mean.copy()
        rows.append(vec)

    attrs = torch.tensor(np.stack(rows), dtype=torch.float32)
    print(f"  class_attrs ({mode}): {attrs.shape}")
    return attrs


# ─────────────────────────────────────────────────────────────────────────────
# BINARY CLASS ATTRIBUTE MATRIX (Jaccard-adjacency source)
# ─────────────────────────────────────────────────────────────────────────────

def build_binary_class_attrs(X_train: np.ndarray,
                             y_train_str: np.ndarray,
                             X_test: np.ndarray,
                             y_test_str: np.ndarray,
                             all_families: list,
                             top_k: int = None,
                             X_support: np.ndarray = None,
                             y_support_str: np.ndarray = None,
                             mode: str = "zero_shot") -> torch.Tensor:
    """
    Binary top-k feature-presence vector per class.

    zero_shot : unseen vectors built from ALL unseen test samples (transductive)
    one_shot  : unseen vectors built ONLY from the 1 support sample (inductive)
    """
    global_mean = X_train.mean(axis=0)
    n_features  = X_train.shape[1]

    if mode == "one_shot" and X_support is not None and len(X_support) > 0:
        X_all = np.concatenate([X_train, X_support], axis=0)
        y_all = np.concatenate([y_train_str, y_support_str], axis=0)
    else:
        X_all = np.concatenate([X_train, X_test], axis=0)
        y_all = np.concatenate([y_train_str, y_test_str], axis=0)

    rows = []
    for fam in all_families:
        mask = y_all == fam
        vec  = np.zeros(n_features, dtype=np.float32)
        if mask.any():
            fam_mean  = X_all[mask].mean(axis=0)
            deviation = np.abs(fam_mean - global_mean)
            top_idx   = np.argsort(deviation)[::-1][:top_k]
            vec[top_idx] = 1.0
        rows.append(vec)

    binary_attrs = torch.tensor(np.stack(rows), dtype=torch.float32)
    print(f"  binary_class_attrs ({mode}): {binary_attrs.shape} "
          f"(sparsity: {1 - binary_attrs.mean().item():.2%})")
    return binary_attrs


# ─────────────────────────────────────────────────────────────────────────────
# CLASS DESCRIPTIONS (for FL-ZSL / text-based baselines)
# ─────────────────────────────────────────────────────────────────────────────

def build_class_descriptions(text_descriptions: dict,
                              all_families: list) -> dict:
    """
    Returns {family_name: description_string} for all families.
    Falls back to a generic placeholder if no text data found.
    """
    class_desc = {}
    for fam in all_families:
        if fam in text_descriptions and text_descriptions[fam]:
            combined      = " ".join(text_descriptions[fam][:3])
            class_desc[fam] = combined[:512]
        else:
            class_desc[fam] = (
                f"A malware sample belonging to the {fam} family, "
                f"exhibiting network and system behavioral activity."
            )
            print(f"  WARN: No text description found for {fam}, using fallback.")
    return class_desc


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def prepare(tabular_path: str, text_path: str,
            unseen_families: list, drop_families: list = None,
            test_ratio: float = 0.2, seed: int = 42,
            mode: str = "zero_shot") -> dict:
    """
    Single entry point. Everything is derived from the data.

    Parameters
    ----------
    tabular_path    : path to CSV (last col = label)
    text_path       : path to JSON {id, label, description} records
    unseen_families : family names to hold out for ZSL / OSL
    drop_families   : families to remove entirely before any split
    test_ratio      : fraction of seen data used as seen test set
    seed            : random seed
    mode            : "zero_shot" (default) | "one_shot"
                      In "one_shot" mode, 1 tabular sample per unseen
                      family is moved into a support set (X_support /
                      y_support) and NOT included in X_test.

    Returns (all modes)
    -------------------
    X_train, y_train, X_test, y_test
    class_attrs, binary_class_attrs, class_descriptions
    seen_label_ids, unseen_label_ids, all_class_names
    feature_names, seen_class_names, d_in, le
    y_train_str, y_test_str

    Additional returns (one_shot only — empty tensors in zero_shot)
    ---------------------------------------------------------------
    X_support      : torch.float32 (n_unseen, d_in)
    y_support      : torch.long    (n_unseen,)
    y_support_str  : np.ndarray    (n_unseen,)

    Downstream usage
    ----------------
    For TabPFN / XGBoost / HistGradBoost one-shot:
        X_tr = torch.cat([data["X_train"], data["X_support"]])
        y_tr = torch.cat([data["y_train"], data["y_support"]])

    For ZSL baselines that should remain zero-shot:
        just use data["X_train"] / data["y_train"] as before.
    """
    assert mode in ("zero_shot", "one_shot"), (
        f"mode must be 'zero_shot' or 'one_shot', got '{mode}'"
    )

    print("─" * 15, "Loading tabular CSV...")
    X, y_str, feat_names = load_tabular(tabular_path)

    if drop_families:
        before    = len(y_str)
        keep      = ~np.isin(y_str, drop_families)
        X, y_str  = X[keep], y_str[keep]
        after     = len(y_str)
        print(f"  Dropped families {drop_families}: "
              f"{before - after} rows removed, {after} remaining")
        print(f"  Remaining classes: {np.unique(y_str).tolist()}")

    all_families_in_data = np.unique(y_str).tolist()
    unseen_not_found = [f for f in unseen_families
                        if f not in all_families_in_data]
    if unseen_not_found:
        raise ValueError(f"Unseen families not found in CSV: {unseen_not_found}. "
                         f"Available: {all_families_in_data}")

    seen_families = [f for f in all_families_in_data
                     if f not in unseen_families]
    all_families  = seen_families + unseen_families   # seen first

    print(f"  Seen   : {seen_families}")
    print(f"  Unseen : {unseen_families}")
    print(f"  Mode   : {mode}")

    print("─" * 15, "Loading text JSON...")
    text_descriptions = load_text(text_path)

    print("─" * 15, f"Split + scaling [{mode}]...")
    if mode == "one_shot":
        tab = oneshot_split(X, y_str, seen_families, unseen_families,
                            test_ratio, seed)
    else:
        tab = gzsl_split(X, y_str, seen_families, unseen_families,
                         test_ratio, seed)

    print("─" * 15, "Encoding labels...")
    y_train, y_test, y_support, le = encode_labels(
        tab["y_train_str"], tab["y_test_str"],
        seen_families, unseen_families,
        y_support_str=tab["y_support_str"],
    )

    seen_ids   = le.transform(seen_families).tolist()
    unseen_ids = le.transform(unseen_families).tolist()

    print("─" * 15, "Building class attribute matrix and descriptions...")
    class_attrs = build_class_attrs_from_features(
        X_train       = tab["X_train"],
        y_train_str   = tab["y_train_str"],
        X_test        = tab["X_test"],
        y_test_str    = tab["y_test_str"],
        all_families  = all_families,
        X_support     = tab["X_support"],
        y_support_str = tab["y_support_str"],
        unseen_families = unseen_families,
        mode          = mode,
    )

    class_desc = build_class_descriptions(text_descriptions, all_families)

    binary_class_attrs = build_binary_class_attrs(
        X_train       = tab["X_train"],
        y_train_str   = tab["y_train_str"],
        X_test        = tab["X_test"],
        y_test_str    = tab["y_test_str"],
        all_families  = all_families,
        top_k         = 40,
        X_support     = tab["X_support"],
        y_support_str = tab["y_support_str"],
        mode          = mode,
    )

    print("\n  Data preparation complete.")
    print(f"  d_in        : {X.shape[1]}")
    print(f"  class_attrs : {class_attrs.shape}")
    if mode == "one_shot":
        print(f"  X_support   : {tab['X_support'].shape}")

    return {
        "X_train"          : torch.tensor(tab["X_train"],   dtype=torch.float32),
        "y_train"          : torch.tensor(y_train,           dtype=torch.long),
        "X_test"           : torch.tensor(tab["X_test"],    dtype=torch.float32),
        "y_test"           : torch.tensor(y_test,            dtype=torch.long),
        # ── One-shot support set ──────────────────────────────────────────────
        # Zero-shot mode: empty tensors with correct shape — no code breakage
        "X_support"        : torch.tensor(tab["X_support"],  dtype=torch.float32),
        "y_support"        : torch.tensor(y_support,          dtype=torch.long),
        "y_support_str"    : tab["y_support_str"],
        # ── Shared ───────────────────────────────────────────────────────────
        "class_attrs"      : class_attrs,
        "class_descriptions": class_desc,
        "seen_label_ids"   : seen_ids,
        "unseen_label_ids" : unseen_ids,
        "all_class_names"  : all_families,
        "feature_names"    : feat_names,
        "seen_class_names" : seen_families,
        "binary_class_attrs": binary_class_attrs,
        "d_in"             : X.shape[1],
        "le"               : le,
        "y_train_str"      : tab["y_train_str"],
        "y_test_str"       : tab["y_test_str"],
        "mode"             : mode,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SCRIPT ENTRY
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    UNSEEN = ["True Content", "Satire or Parody"]
    DROP   = None #['skymobi', 'kuguo']

    # ── Run both modes ────────────────────────────────────────────────────────
    for run_mode in ("zero_shot", "one_shot"):
        print("\n" + "═" * 60)
        print(f"  MODE: {run_mode.upper()}")
        print("═" * 60)

        data = prepare(
            tabular_path    = "/home/iamontoyasa/02_VENOM/00_Datasets/08_fakeddit/processed/fakeddit_tabular.csv",
            text_path       = "/home/iamontoyasa/02_VENOM/00_Datasets/08_fakeddit/processed/fakeddit_text.json",
            unseen_families = UNSEEN,
            drop_families   = DROP,
            mode            = run_mode,
        )

        le = data["le"]

        print("\n" + "─" * 55)
        print("  DATA PREPARATION SUMMARY")
        print("─" * 55)
        print(f"  X_train    : {data['X_train'].shape}")
        print(f"  y_train    : {data['y_train'].shape}")
        print(f"  X_test     : {data['X_test'].shape}")
        print(f"  y_test     : {data['y_test'].shape}")
        print(f"  X_support  : {data['X_support'].shape}  ← 1 per unseen (empty in ZSL)")
        print(f"  y_support  : {data['y_support'].shape}")
        print(f"  class_attrs: {data['class_attrs'].shape}")
        print(f"  d_in       : {data['d_in']}")

        print("\n  ENCODING")
        for fam in data["all_class_names"]:
            fam_id = le.transform([fam])[0]
            split  = "SEEN" if fam in data["seen_class_names"] else "UNSEEN"
            print(f"  {fam_id:>3} {split:<8} {fam}")

        if run_mode == "one_shot":
            print("\n  SUPPORT SET (1-shot samples)")
            for i, fam in enumerate(data["y_support_str"]):
                fam_id = le.transform([fam])[0]
                print(f"    sample {i}: label={fam} (id={fam_id})")

            print("\n  ONE-SHOT USAGE EXAMPLE (for TabPFN / XGBoost / HistGB)")
            print("    X_tr = torch.cat([data[\'X_train\'], data[\'X_support\']])")
            print("    y_tr = torch.cat([data[\'y_train\'], data[\'y_support\']])")

        # ── Save ─────────────────────────────────────────────────────────────
        save_path = (
            f"/home/iamontoyasa/02_VENOM/03_Baseline/08_run1_prepared_data_{run_mode}.pt"
        )
        torch.save({
            "X_train"          : data["X_train"],
            "y_train"          : data["y_train"],
            "y_train_str"      : data["y_train_str"],
            "X_test"           : data["X_test"],
            "y_test"           : data["y_test"],
            "y_test_str"       :data["y_test_str"],
            "X_support"        : data["X_support"],
            "y_support"        : data["y_support"],
            "y_support_str"    : data["y_support_str"],
            "class_attrs"      : data["class_attrs"],
            "class_descriptions": data["class_descriptions"],
            "seen_label_ids"   : data["seen_label_ids"],
            "unseen_label_ids" : data["unseen_label_ids"],
            "all_class_names"  : data["all_class_names"],
            "feature_names"    : data["feature_names"],
            "seen_class_names" : data["seen_class_names"],
            "binary_class_attrs": data["binary_class_attrs"],
            "d_in"             : data["d_in"],
            "le"               : data["le"],
            "mode"             : data["mode"],
        }, save_path)
        print(f"\n  Saved → {save_path}")
