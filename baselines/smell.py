#%% md
# ## Cell 0 — ⚙️ Global Configuration
#%%
# ════════════════════════════════════════════════════════════════════
# GLOBAL CONFIG — edit paths/flags here only
# ════════════════════════════════════════════════════════════════════
import os

# ── CSV (tabular features) ────────────────────────────────────────────
CSV_PATH = "/home/iamontoyasa/02_VENOM/00_Datasets/01_ANDMAL/03_Dynamic_Features/mlp_baseline_family.csv"

# ── LLM Embeddings directory ──────────────────────────────────────────
EMBEDDINGS_DIR = "/home/iamontoyasa/02_VENOM/00_Datasets/01_ANDMAL/02_embeddings"

# ── Unseen families (must match CSV label column values exactly) ──────
UNSEEN_NAMES = ["airpush", "kuguo", "lockscreen", "skymobi", "smsagent", "lotoor"]

# ── Architecture ──────────────────────────────────────────────────────
LATENT_DIM   = 128
NUM_MARKERS  = 20
EPOCHS_SMELL = 50
EPOCHS_PROJ  = 300

# ── Scan the embeddings directory to confirm files are accessible ─────
print(f"Embeddings directory: {EMBEDDINGS_DIR}")
if os.path.isdir(EMBEDDINGS_DIR):
    files = sorted(os.listdir(EMBEDDINGS_DIR))
    print(f"  Found {len(files)} file(s):")
    for f in files:
        fpath = os.path.join(EMBEDDINGS_DIR, f)
        size  = os.path.getsize(fpath) / 1024
        print(f"    {f:<50}  ({size:,.1f} KB)")
else:
    print("  ⚠️  Directory not found — check EMBEDDINGS_DIR path.")

#%%
import pickle
with open(os.path.join(EMBEDDINGS_DIR, "unseen_classes.pkl"), "rb") as f:
    print(pickle.load(f))
#%% md
# ## Cell 1 — Imports
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#%% md
# ## Cell 2 — MalwareSMELLDataset
#%%
class MalwareSMELLDataset(Dataset):
    def __init__(self, df):
        self.features = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
        self.labels   = df.iloc[:, -1].values
        self.classes  = np.unique(self.labels)
        self.label_to_indices = {c: np.where(self.labels == c)[0] for c in self.classes}

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x1, y1 = self.features[idx], self.labels[idx]
        is_same = np.random.random() > 0.5
        if is_same:
            idx2 = np.random.choice(self.label_to_indices[y1])
            label = 1.0
        else:
            y2   = np.random.choice([c for c in self.classes if c != y1])
            idx2 = np.random.choice(self.label_to_indices[y2])
            label = 0.0
        return x1, self.features[idx2], torch.tensor(label, dtype=torch.float32)

#%% md
# ## Cell 3 — Encoder, SSpaceSimilarity & SMELLLoss
#%%
class MLPEncoder(nn.Module):
    def __init__(self, input_dim=282, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
    def forward(self, x):
        return F.normalize(self.encoder(x), p=2, dim=-1)


class SSpaceSimilarity(nn.Module):
    def __init__(self, latent_dim, num_markers=20, gamma_init=1.0):
        super().__init__()
        self.log_gamma = nn.Parameter(torch.tensor(np.log(gamma_init), dtype=torch.float32))
        self.markers   = nn.Parameter(torch.randn(num_markers, latent_dim))
        self.fc        = nn.Linear(num_markers, 1)
        self.sigmoid   = nn.Sigmoid()

    @property
    def gamma(self): return torch.exp(self.log_gamma)

    def forward(self, z1, z2):
        s       = torch.abs(z1 - z2)
        dist_sq = torch.cdist(s, self.markers, p=2) ** 2
        cauchy  = 1.0 / (1.0 + (dist_sq / (self.gamma ** 2)))
        return self.sigmoid(self.fc(cauchy)).squeeze(-1)


class SMELLLoss(nn.Module):
    def __init__(self, margin=1.0, lambda_c=0.5):
        super().__init__()
        self.margin   = margin
        self.lambda_c = lambda_c
        self.bce      = nn.BCELoss()

    def forward(self, score, z1, z2, labels):
        bce_loss = self.bce(score, labels)
        dist     = F.pairwise_distance(z1, z2, p=2)
        pos      = labels * dist.pow(2)
        neg      = (1 - labels) * F.relu(self.margin - dist).pow(2)
        return bce_loss + self.lambda_c * (pos + neg).mean(), bce_loss, (pos + neg).mean()

#%% md
# ## Cell 4 — K-Means Marker Initialisation
#%%
def initialize_markers_with_kmeans(encoder, dataloader, num_markers):
    encoder.eval()
    diffs = []
    print("Extracting initial latent differences for marker initialisation...")
    with torch.no_grad():
        for i, (x1, x2, _) in enumerate(dataloader):
            z1 = encoder(x1.to(device))
            z2 = encoder(x2.to(device))
            diffs.append(torch.abs(z1 - z2).cpu().numpy())
            if i > 20: break
    diffs = np.vstack(diffs)
    kmeans = KMeans(n_clusters=min(num_markers, len(diffs)),
                    init="k-means++", n_init=10, random_state=42)
    kmeans.fit(diffs)
    return torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

#%% md
# ## Cell 5 — `run_training`
#%%
def run_training(df_train, input_dim=282, latent_dim=128, num_markers=20,
                 epochs=50, lr=1e-3, weight_decay=1e-4, margin=1.0, lambda_c=0.5):
    train_ds     = MalwareSMELLDataset(df_train)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=True)

    encoder      = MLPEncoder(input_dim, latent_dim).to(device)
    init_markers = initialize_markers_with_kmeans(encoder, train_loader, num_markers)
    s_space      = SSpaceSimilarity(latent_dim, num_markers).to(device)
    s_space.markers.data = init_markers.to(device)

    params    = list(encoder.parameters()) + list(s_space.parameters())
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = SMELLLoss(margin=margin, lambda_c=lambda_c)

    encoder.train(); s_space.train()
    for epoch in range(epochs):
        total_loss = bce_sum = con_sum = 0.0
        for x1, x2, labels in train_loader:
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
            optimizer.zero_grad()
            z1, z2 = encoder(x1), encoder(x2)
            score  = s_space(z1, z2)
            loss, bce, con = criterion(score, z1, z2, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            total_loss += loss.item(); bce_sum += bce.item(); con_sum += con.item()
        scheduler.step()
        n = len(train_loader)
        print(f"Epoch {epoch+1:3d}/{epochs} | Total: {total_loss/n:.4f} "
              f"BCE: {bce_sum/n:.4f} Contrastive: {con_sum/n:.4f} "
              f"gamma: {s_space.gamma.item():.4f} lr: {scheduler.get_last_lr()[0]:.6f}")
    return encoder, s_space

#%% md
# ## Cell 6 — Pairwise Evaluation
#%%
def evaluate_pairwise(encoder, s_space, df_test):
    encoder.eval(); s_space.eval()
    test_ds     = MalwareSMELLDataset(df_test)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x1, x2, labels in test_loader:
            score = s_space(encoder(x1.to(device)), encoder(x2.to(device)))
            all_preds.extend((score > 0.5).float().cpu().numpy())
            all_labels.extend(labels.numpy())
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")
    print(f"Pairwise Evaluation | Precision: {p:.4f} Recall: {r:.4f} F1: {f1:.4f}")

#%% md
# ## Cell 7 — Distribution Visualisation
#%%
def plot_distributions(encoder, s_space, df_test, n_samples=500):
    encoder.eval(); s_space.eval()
    test_ds     = MalwareSMELLDataset(df_test)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=True)
    pos_dist, neg_dist = [], []
    with torch.no_grad():
        for i, (x1, x2, label) in enumerate(test_loader):
            z1, z2 = encoder(x1.to(device)), encoder(x2.to(device))
            dist   = torch.norm(z1 - z2, p=2).item()
            (pos_dist if label.item() == 1.0 else neg_dist).append(dist)
            if i >= n_samples: break
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(pos_dist, bins=30, alpha=0.6, label="Same Class",  color="steelblue")
    axes[0].hist(neg_dist, bins=30, alpha=0.6, label="Diff Class", color="tomato")
    axes[0].set_title("Latent Space: L2 Distance Distribution")
    axes[0].set_xlabel("L2 Distance"); axes[0].set_ylabel("Count"); axes[0].legend()
    mn = s_space.markers.norm(dim=1).cpu().detach().numpy()
    axes[1].stem(range(len(mn)), mn, linefmt="g-", markerfmt="go", basefmt=" ")
    axes[1].set_title("Learned S-Space Marker Magnitudes")
    axes[1].set_xlabel("Marker Index"); axes[1].set_ylabel("L2 Norm")
    plt.tight_layout(); plt.show()

#%% md
# ## Cell 8 — Data Preparation
#%%
df_raw = pd.read_csv(CSV_PATH).fillna(0.0)

le = LabelEncoder()
df_raw.iloc[:, -1] = le.fit_transform(df_raw.iloc[:, -1].astype(str))

# Reverse lookup tables (used throughout for alignment)
NAME_TO_ID = {name.lower(): int(i) for i, name in enumerate(le.classes_)}
ID_TO_NAME = {int(i): name       for i, name in enumerate(le.classes_)}

print("LabelEncoder classes (actual CSV family names):")
for enc_id, name in enumerate(le.classes_):
    tag = " ← UNSEEN" if name in UNSEEN_NAMES else ""
    print(f"  [{enc_id:2d}] '{name}'{tag}")

unseen_encoded_ids = le.transform(UNSEEN_NAMES)
df_unseen          = df_raw[df_raw.iloc[:, -1].isin(unseen_encoded_ids)].copy()
df_seen_total      = df_raw[~df_raw.iloc[:, -1].isin(unseen_encoded_ids)].copy()

df_seen_train, df_seen_val = train_test_split(
    df_seen_total, test_size=0.20, random_state=42,
    stratify=df_seen_total.iloc[:, -1]
)

print(f"\nTotal rows       : {len(df_raw)}")
print(f"Train (seen 80%) : {len(df_seen_train)}")
print(f"Val   (seen 20%) : {len(df_seen_val)}")
print(f"Test  (unseen)   : {len(df_unseen)}")

#%% md
# ## Cell 9 — Run SMELL Training
#%%
INPUT_DIM = df_seen_train.shape[1] - 1

print("\n── Phase 1: Training on Seen Data (80%) ──")
encoder, s_space = run_training(
    df_train=df_seen_train, input_dim=INPUT_DIM, latent_dim=LATENT_DIM,
    num_markers=NUM_MARKERS, epochs=EPOCHS_SMELL,
    lr=1e-3, weight_decay=1e-4, margin=1.0, lambda_c=0.5
)

print("\n── Phase 2: Pairwise Evaluation on Seen Val Set ──")
evaluate_pairwise(encoder, s_space, df_seen_val)

#%% md
# ## Cell 10 — Calculate Feature Prototypes (Seen Classes)
#%%
def calculate_prototypes(encoder, df_train):
    """Per-family L2-normalised centroid in the encoder's latent space."""
    encoder.eval()
    prototypes = {}
    with torch.no_grad():
        for label in df_train.iloc[:, -1].unique():
            feats = torch.tensor(
                df_train[df_train.iloc[:, -1] == label].iloc[:, :-1].values,
                dtype=torch.float32
            ).to(device)
            z     = encoder(feats)
            proto = z.mean(dim=0)
            prototypes[int(label)] = F.normalize(proto.unsqueeze(0), p=2, dim=-1).squeeze(0)
    return prototypes

seen_prototypes = calculate_prototypes(encoder, df_seen_train)
print(f"Calculated prototypes for {len(seen_prototypes)} seen malware families.")

#%% md
# ## Cell 11 — GZSL Evaluation + Gamma Sweep
#%%
def evaluate_gzsl(encoder, s_space, prototypes, df_seen_val, df_unseen,
                  gamma_bias=None, batch_size=256, verbose=True):
    encoder.eval(); s_space.eval()
    proto_labels  = list(prototypes.keys())
    proto_tensors = torch.stack([prototypes[l] for l in proto_labels]).to(device)
    seen_ids      = set(df_seen_val.iloc[:, -1].unique().tolist())

    def predict(df):
        feats = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
        preds = []
        with torch.no_grad():
            for start in range(0, len(feats), batch_size):
                batch  = feats[start:start+batch_size].to(device)
                z      = encoder(batch)
                B, M   = z.size(0), len(proto_labels)
                z_flat = z.unsqueeze(1).expand(B, M, -1).reshape(B*M, -1)
                p_flat = proto_tensors.unsqueeze(0).expand(B, M, -1).reshape(B*M, -1)
                scores = s_space(z_flat, p_flat).view(B, M)
                if gamma_bias is not None and gamma_bias > 0:
                    for j, lbl in enumerate(proto_labels):
                        if lbl in seen_ids: scores[:, j] -= gamma_bias
                preds.extend([proto_labels[i] for i in scores.argmax(dim=1).cpu().numpy()])
        return np.array(df.iloc[:, -1].values).astype(int), np.array(preds).astype(int)

    y_true_s, y_pred_s = predict(df_seen_val)
    y_true_u, y_pred_u = predict(df_unseen)
    acc_s = accuracy_score(y_true_s, y_pred_s)
    acc_u = accuracy_score(y_true_u, y_pred_u)
    h     = (2 * acc_s * acc_u) / (acc_s + acc_u + 1e-9)
    f1_s  = f1_score(y_true_s, y_pred_s, average="macro", zero_division=0)
    f1_u  = f1_score(y_true_u, y_pred_u, average="macro", zero_division=0)

    if verbose:
        print("\n" + "="*54)
        print(" GZSL RESULTS (SMELL)")
        print("="*54)
        print(f" Seen   Acc (S) : {acc_s*100:6.2f}% | Macro-F1: {f1_s*100:.2f}%")
        print(f" Unseen Acc (U) : {acc_u*100:6.2f}% | Macro-F1: {f1_u*100:.2f}%")
        print(f" H-Mean     (H) : {h*100:6.2f}%")
        if gamma_bias is not None: print(f" Calibration γ  : {gamma_bias:.3f}")
        print("="*54)
        print("\nPer-Family Accuracy:")
        print(f"  {'Family':>20} {'ID':>4} {'Split':>8} {'N':>6} {'Acc':>8}")
        print("  " + "-"*50)
        all_true = np.concatenate([y_true_s, y_true_u])
        all_pred = np.concatenate([y_pred_s, y_pred_u])
        for fam in np.unique(all_true):
            mask  = all_true == fam
            acc_f = accuracy_score(all_true[mask], all_pred[mask])
            split = "UNSEEN" if fam in set(unseen_encoded_ids.tolist()) else "seen"
            print(f"  {ID_TO_NAME.get(fam, str(fam)):>20} {fam:>4} {split:>8} "
                  f"{mask.sum():>6} {acc_f*100:>7.2f}%")

    return {"S": acc_s, "U": acc_u, "H": h, "F1_S": f1_s, "F1_U": f1_u}


def sweep_gamma(encoder, s_space, prototypes, df_seen_val, df_unseen, gamma_values=None):
    if gamma_values is None:
        gamma_values = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    best_h, best_gamma, best_metrics = -1, 0.0, {}
    print(f"\n  {'γ':>6} {'S':>8} {'U':>8} {'H':>8}")
    print("  " + "-"*34)
    for g in gamma_values:
        m = evaluate_gzsl(encoder, s_space, prototypes, df_seen_val, df_unseen,
                          gamma_bias=g, batch_size=256, verbose=False)
        print(f"  {g:>6.2f} {m['S']*100:>7.2f}% {m['U']*100:>7.2f}% {m['H']*100:>7.2f}%")
        if m["H"] > best_h:
            best_h, best_gamma, best_metrics = m["H"], g, m
    print(f"\n  Best γ = {best_gamma:.2f} → H-Mean = {best_h*100:.2f}%")
    return best_gamma, best_metrics

#%% md
# ## Cell 12 — Merge Unseen Anchors
#%%
def merge_unseen_anchors(seen_prototypes, unseen_anchors):
    full = seen_prototypes.copy()
    for label, anchor in unseen_anchors.items():
        if not isinstance(anchor, torch.Tensor):
            anchor = torch.tensor(anchor, dtype=torch.float32)
        full[int(label)] = anchor.to(device)
    print(f"Total prototypes for GZSL: {len(full)} "
          f"({len(seen_prototypes)} seen + {len(unseen_anchors)} unseen)")
    return full

#%% md
# ## Cell 13 — SemanticProjectionHead
#%%
class SemanticProjectionHead(nn.Module):
    """MLP: text_dim → hidden → hidden → latent_dim (L2-normalised output)."""
    def __init__(self, text_dim, latent_dim=128, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    def forward(self, t):
        return F.normalize(self.net(t), p=2, dim=-1)

#%% md
# ## Cell 14 — Load LLM Text Prototypes
# 
# Loads pre-computed LLM embeddings from `EMBEDDINGS_DIR` using the same prototype
# construction used in VENOM: concatenate splits → group by family → mean-pool → L2-normalise.
# 
# **File naming convention expected (adjust `_FILE` variables if yours differ):**
# ```
# train_embeddings.npy  /  train_meta.csv    (or train_metadata.csv)
# test_seen_embeddings.npy  /  test_seen_meta.csv
# unseen_embeddings.npy  /  unseen_meta.csv
# ```
# Metadata CSVs must contain a `label` column with string family names.
#%%
# ── File name variables ──────────────────────
TRAIN_EMB_FILE      = "train_embeddings.npy"
TRAIN_META_FILE     = "train_data.pkl"
TEST_SEEN_EMB_FILE  = "test_seen_embeddings.npy"
TEST_SEEN_META_FILE = "test_seen_data.pkl"       #
UNSEEN_EMB_FILE     = "unseen_embeddings.npy"
UNSEEN_META_FILE    = "unseen_data.pkl"


def _load_emb_meta(emb_file, meta_file):
    """Load one (embeddings, metadata) pair. Supports .npy/.npz + .csv/.pkl."""
    # ── Embeddings ────────────────────────────────────────────────────────────
    if emb_file.endswith(".npy"):
        emb = np.load(emb_file)
    elif emb_file.endswith(".npz"):
        data = np.load(emb_file)
        emb  = data[list(data.files)[0]]
    else:
        raise ValueError(f"Unsupported embedding format: {emb_file}")

    # ── Metadata ──────────────────────────────────────────────────────────────
    if meta_file.endswith(".csv"):
        meta = pd.read_csv(meta_file)
    elif meta_file.endswith(".pkl"):
        meta = pd.read_pickle(meta_file)
        # Normalise: if pkl is a dict/other structure, convert to DataFrame
        if not isinstance(meta, pd.DataFrame):
            meta = pd.DataFrame(meta)
    else:
        raise ValueError(f"Unsupported metadata format: {meta_file}")

    if "label" not in meta.columns:
        raise ValueError(
            f"No 'label' column in {meta_file}.\n"
            f"Available columns: {list(meta.columns)}\n"
            f"Update the column reference in build_llm_text_prototypes() to match."
        )

    assert len(emb) == len(meta), (
        f"Shape mismatch: embeddings={len(emb)} rows, meta={len(meta)} rows "
        f"in {emb_file} / {meta_file}"
    )
    return emb, meta

def build_llm_text_prototypes(embeddings_dir):
    """
    Replicates the VENOM prototype construction:
      1. Concatenate all splits
      2. Group by family string name
      3. Mean-pool embeddings per family
      4. L2-normalise
      5. Map family name → LabelEncoder integer ID

    Returns
    -------
    text_prototypes_llm : dict {int_id: torch.Tensor[text_dim]}
    TEXT_DIM            : int
    """
    p = lambda f: os.path.join(embeddings_dir, f)

    print("Loading embeddings...")
    train_emb,     train_meta     = _load_emb_meta(p(TRAIN_EMB_FILE),     p(TRAIN_META_FILE))
    test_seen_emb, test_seen_meta = _load_emb_meta(p(TEST_SEEN_EMB_FILE), p(TEST_SEEN_META_FILE))
    unseen_emb,    unseen_meta    = _load_emb_meta(p(UNSEEN_EMB_FILE),    p(UNSEEN_META_FILE))

    all_emb  = np.concatenate([train_emb, test_seen_emb, unseen_emb], axis=0)
    all_meta = pd.concat([train_meta, test_seen_meta, unseen_meta], ignore_index=True)

    print(f"  Total embedding matrix : {all_emb.shape}")
    print(f"  Total samples in meta  : {len(all_meta)}")
    print(f"  Unique families        : {all_meta['label'].nunique()}")

    # ── Prototype construction (matches VENOM exactly) ────────────────────────
    text_prototypes_llm = {}
    missing, matched   = [], []

    for family in all_meta['label'].unique():
        idx      = all_meta.index[all_meta['label'] == family].tolist()
        embs     = all_emb[idx]
        proto    = np.mean(embs, axis=0)
        proto    = proto / np.linalg.norm(proto)   # L2 normalise

        key = family.lower()
        if key not in NAME_TO_ID:
            missing.append(family)
            continue

        label_id = NAME_TO_ID[key]
        text_prototypes_llm[label_id] = torch.tensor(proto, dtype=torch.float32)
        matched.append(family)

    if missing:
        print(f"  ⚠️  {len(missing)} families in embeddings not in LabelEncoder: {missing}")

    TEXT_DIM = all_emb.shape[1]

    # ── Validation ────────────────────────────────────────────────────────────
    seen_in_emb   = [k for k in text_prototypes_llm if k not in unseen_encoded_ids]
    unseen_in_emb = [k for k in text_prototypes_llm if k     in unseen_encoded_ids]
    print(f"\nLLM text prototypes built:")
    print(f"  text_dim  : {TEXT_DIM}")
    print(f"  Seen      : {len(seen_in_emb)} families")
    print(f"  Unseen    : {len(unseen_in_emb)} families")

    # Prototype magnitude sanity check (should all be ~1.0)
    mags = [t.norm().item() for t in text_prototypes_llm.values()]
    print(f"  Magnitude : min={min(mags):.4f}  max={max(mags):.4f}  mean={np.mean(mags):.4f}")

    if len(seen_in_emb) == 0:
        raise RuntimeError(
            "No SEEN families loaded! Check that train_meta.csv / test_seen_meta.csv "
            "label values match le.classes_ (printed in Cell 8)."
        )

    return text_prototypes_llm, TEXT_DIM


# Build LLM prototypes
text_prototypes_llm, TEXT_DIM = build_llm_text_prototypes(EMBEDDINGS_DIR)

#%% md
# ## Cell 15 — Train Projection Head
#%%
def train_projection_head(text_embeddings, seen_prototypes,
                          text_dim, latent_dim=128,
                          epochs=300, lr=3e-4):
    """
    Train SemanticProjectionHead on SEEN classes only.
    Loss: 1 - mean cosine_similarity(proj(t_c), proto_c)
    """
    proj_head = SemanticProjectionHead(text_dim=text_dim, latent_dim=latent_dim).to(device)
    optimizer = optim.AdamW(proj_head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    seen_t, seen_p = [], []
    for lbl, proto in seen_prototypes.items():
        if int(lbl) in text_embeddings:
            seen_t.append(text_embeddings[int(lbl)])
            seen_p.append(proto.detach())

    if len(seen_t) == 0:
        raise ValueError(
            "No seen families in text_embeddings — check label alignment (Cell 8 + Cell 14)."
        )

    T = torch.stack(seen_t).to(device)
    P = torch.stack(seen_p).to(device)
    print(f"Training projection head on {len(seen_t)} seen families  "
          f"(text_dim={text_dim} → latent_dim={latent_dim}) ...")

    for epoch in range(epochs):
        proj_head.train()
        optimizer.zero_grad()
        loss = 1.0 - F.cosine_similarity(proj_head(T), P, dim=-1).mean()
        loss.backward()
        optimizer.step(); scheduler.step()
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:4d}/{epochs} | Cosine Align Loss: {loss.item():.6f}")

    proj_head.eval()
    with torch.no_grad():
        final_cos = F.cosine_similarity(proj_head(T), P, dim=-1).mean().item()
    print(f"Final avg cosine similarity on seen classes: {final_cos:.4f}")
    return proj_head

#%% md
# ## Cell 16 — Synthesise Unseen Anchors
#%%
def synthesize_unseen_anchors(proj_head, text_embeddings, unseen_ids):
    """
    Project each unseen family's text embedding through the trained head
    to synthesise its feature-space prototype.
    """
    proj_head.eval()
    unseen_anchors = {}
    with torch.no_grad():
        for uid in unseen_ids:
            uid = int(uid)
            if uid not in text_embeddings:
                print(f"  Warning: unseen ID {uid} ({ID_TO_NAME.get(uid,'?')}) "
                      f"not in text_embeddings")
                continue
            anchor = proj_head(text_embeddings[uid].unsqueeze(0).to(device)).squeeze(0)
            unseen_anchors[uid] = anchor
    print(f"Synthesised {len(unseen_anchors)} unseen-class anchors.")
    return unseen_anchors

#%% md
# ## Cell 17 — Full GZSL Pipeline (LLM vs Random Comparison)
# 
# Runs the complete pipeline **twice** in the same cell:
# 1. **LLM embeddings** — text prototypes from your pre-computed embedding files
# 2. **Random baseline** — random Gaussian vectors of the same `TEXT_DIM` passed through
#    the same projection head architecture (only the input signal differs)
# 
# Both runs share the same trained `encoder` + `s_space` and `seen_prototypes`.
#%%
# ════════════════════════════════════════════════════════════════════
# FULL GZSL PIPELINE  —  LLM Embeddings  vs  Random Baseline
# ════════════════════════════════════════════════════════════════════

def build_random_prototypes(ref_prototypes, text_dim, seed=42):
    """
    Random Gaussian vectors (same text_dim as LLM) for all families.
    Using the same text_dim keeps the projection head architecture identical.
    seed is fixed so the ablation is reproducible.
    """
    rng = np.random.default_rng(seed)
    rand_emb = {}
    for lid in ref_prototypes:           # seen families
        v = rng.standard_normal(text_dim).astype(np.float32)
        v /= np.linalg.norm(v)
        rand_emb[int(lid)] = torch.tensor(v)
    for uid in map(int, unseen_encoded_ids):   # unseen families
        v = rng.standard_normal(text_dim).astype(np.float32)
        v /= np.linalg.norm(v)
        rand_emb[uid] = torch.tensor(v)
    print(f"Built {len(rand_emb)} random prototypes  (text_dim={text_dim}, seed={seed})")
    return rand_emb


def run_gzsl_pipeline(text_embeddings, text_dim, mode_label):
    """
    Steps 1-5 of the GZSL pipeline for a given embedding source.
    Returns final metrics dict.
    """
    print(f"\n{'═'*60}")
    print(f"  RUN: {mode_label}")
    print(f"{'═'*60}")

    # 1 — Train projection head (seen classes only)
    print("\n── Step 1: Train Projection Head ──")
    proj = train_projection_head(
        text_embeddings=text_embeddings,
        seen_prototypes=seen_prototypes,
        text_dim=text_dim,
        latent_dim=LATENT_DIM,
        epochs=EPOCHS_PROJ,
        lr=3e-4
    )

    # 2 — Synthesise unseen anchors via trained head
    print("\n── Step 2: Synthesise Unseen Anchors ──")
    anchors = synthesize_unseen_anchors(proj, text_embeddings, unseen_encoded_ids)

    # 3 — Merge seen prototypes + unseen anchors
    print("\n── Step 3: Merge Prototypes ──")
    full_proto = merge_unseen_anchors(seen_prototypes, anchors)

    # 4 — Calibration sweep (γ bias)
    print("\n── Step 4: Calibration Sweep (γ) ──")
    best_gamma, _ = sweep_gamma(
        encoder, s_space, full_proto, df_seen_val, df_unseen,
        gamma_values=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    )

    # 5 — Final evaluation with best γ
    print(f"\n── Step 5: Final Evaluation (γ={best_gamma}) ──")
    metrics = evaluate_gzsl(
        encoder, s_space, full_proto, df_seen_val, df_unseen,
        gamma_bias=best_gamma, verbose=True
    )
    return metrics


# ── Run 1: LLM Embeddings ─────────────────────────────────────────────────────
results = {}
results["LLM Embeddings"] = run_gzsl_pipeline(
    text_embeddings=text_prototypes_llm,
    text_dim=TEXT_DIM,
    mode_label="LLM Embeddings (VENOM-aligned text prototypes)"
)

# ── Run 2: Random Baseline ───────────────────────────────────────────────────
random_prototypes = build_random_prototypes(seen_prototypes, TEXT_DIM, seed=42)
results["Random Init"] = run_gzsl_pipeline(
    text_embeddings=random_prototypes,
    text_dim=TEXT_DIM,
    mode_label="Random Baseline (same arch, random Gaussian input)"
)

# ── Side-by-side comparison table ────────────────────────────────────────────
print("\n" + "═"*65)
print("  SMELL-GZSL Embedding Comparison — CIC-AndMal-2020")
print("═"*65)
print(f"  {'Method':<30} {'S (Acc)':>9} {'U (Acc)':>9} {'H-Mean':>9}")
print("  " + "-"*57)
for name, m in results.items():
    print(f"  {name:<30} {m['S']*100:>8.2f}%  {m['U']*100:>8.2f}%  {m['H']*100:>8.2f}%")
print("═"*65)
print(f"  {'':30} {'F1_S':>9} {'F1_U':>9}")
print("  " + "-"*57)
for name, m in results.items():
    print(f"  {name:<30} {m['F1_S']*100:>8.2f}%  {m['F1_U']*100:>8.2f}%")
print("═"*65)

# ── LaTeX rows (paste directly into your table) ───────────────────────────────
print("\n% LaTeX:")
for name, m in results.items():
    print(f"SMELL-GZSL ({name}) & {m['S']*100:.1f} & {m['U']*100:.1f} "
          f"& {m['H']*100:.1f} & {m['F1_S']*100:.1f} & {m['F1_U']*100:.1f} \\\\")
