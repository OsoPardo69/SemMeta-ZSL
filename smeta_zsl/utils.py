import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_embeddings(emb_dir: Path):
    """Loads the three .npy embedding files and three .pkl metadata files from Stage 1."""
    train_emb = np.load(emb_dir / "train_embeddings.npy")
    test_seen_emb = np.load(emb_dir / "test_seen_embeddings.npy")
    unseen_emb = np.load(emb_dir / "unseen_embeddings.npy")
    train_meta = pd.read_pickle(emb_dir / "train_data.pkl")
    test_seen_meta = pd.read_pickle(emb_dir / "test_seen_data.pkl")
    unseen_meta = pd.read_pickle(emb_dir / "unseen_data.pkl")
    return train_emb, test_seen_emb, unseen_emb, train_meta, test_seen_meta, unseen_meta


def build_prototypes(all_emb: np.ndarray, all_meta: pd.DataFrame) -> dict:
    """Computes an L2-normalized mean embedding (prototype) per class."""
    prototypes = {}
    for family in all_meta["label"].unique():
        idx = all_meta.index[all_meta["label"] == family].tolist()
        proto = all_emb[idx].mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-8)
        prototypes[family] = proto
    return prototypes


def prepare_tabular(
    tab_csv: Path,
    seen_families: list,
    unseen_families: list,
    text_prototypes: dict,
    test_size: float = 0.2,
    seed: int = 42,
):
    """
    Loads the tabular CSV, splits seen data into train/val, standardizes features.

    The tabular CSV must have features in all columns except the last, which is the label.

    Returns:
        X_train, y_train, X_val, y_val, X_unseen, y_unseen, scaler, input_dim
    """
    df = pd.read_csv(tab_csv)
    label_col = df.columns[-1]
    feature_cols = df.columns[:-1].tolist()

    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=feature_cols)

    df_seen = df[df[label_col].isin(seen_families)].copy()
    df_unseen = df[df[label_col].isin(unseen_families)].copy()

    X_all = df_seen[feature_cols].values
    y_all = df_seen[label_col].values

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=test_size, stratify=y_all, random_state=seed
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train).astype(np.float32)
    X_val_sc = scaler.transform(X_val).astype(np.float32)
    X_unseen_sc = scaler.transform(df_unseen[feature_cols].values).astype(np.float32)

    return (
        X_train_sc, list(y_train),
        X_val_sc, list(y_val),
        X_unseen_sc, list(df_unseen[label_col].values),
        scaler,
        len(feature_cols),
    )
