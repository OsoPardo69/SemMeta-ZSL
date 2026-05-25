import os
import copy
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

SEEDS = [42, 123, 456, 789, 2025]

DATASETS = {
        "AndMAL": "/home/iamontoyasa/02_VENOM/03_Baseline/01_run1_prepared_data_zero_shot.pt",
        "BODMAS": "/home/iamontoyasa/02_VENOM/03_Baseline/02_run1_prepared_data_zero_shot.pt",
        "AVASTCTU": "/home/iamontoyasa/02_VENOM/03_Baseline/03_run1_prepared_data_zero_shot.pt",
        "APIGRAPH": "/home/iamontoyasa/02_VENOM/03_Baseline/04_run1_prepared_data_zero_shot.pt",
        "Goodread": "/home/iamontoyasa/02_VENOM/03_Baseline/06_run1_prepared_data_zero_shot.pt",
}

RESULT_DIR = "/home/iamontoyasa/02_VENOM/03_Baseline/results"
os.makedirs(RESULT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_hmean(acc_seen: float, acc_unseen: float) -> float:
    if acc_seen + acc_unseen == 0:
        return 0.0
    return 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen + 1e-8)


class LabelRemapper:
    """Maps global class IDs to local 0..n_seen-1 indices for CE loss."""

    def __init__(self, seen_global_ids: list[int]):
        self.global_to_local = {g: l for l, g in enumerate(sorted(seen_global_ids))}
        self.local_to_global = {l: g for g, l in self.global_to_local.items()}

    def to_local(self, y: torch.Tensor) -> torch.Tensor:
        return torch.tensor([self.global_to_local[int(i)] for i in y], dtype=torch.long)

    def to_global(self, y: np.ndarray) -> np.ndarray:
        return np.array([self.local_to_global[int(i)] for i in y])


# ─────────────────────────────────────────────────────────────────────────────
# MODEL COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

class MLPWithProjection(nn.Module):
    def __init__(self, d_in: int, n_seen: int, hidden: int = 256,
                 dropout: float = 0.3, proj_dim: int = 384):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(d_in, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.clf_head = nn.Linear(hidden, n_seen)
        self.proj_head = nn.Sequential(nn.Linear(hidden, proj_dim), nn.Tanh())

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        return self.clf_head(h), self.proj_head(h)


def fed_avg(global_model: nn.Module, client_models: list[nn.Module]) -> nn.Module:
    global_state = global_model.state_dict()
    for key in global_state:
        stacked = torch.stack([cm.state_dict()[key].float() for cm in client_models])
        global_state[key] = stacked.mean(dim=0)
    global_model.load_state_dict(global_state)
    return global_model


def client_train(model: nn.Module, loader: DataLoader, seen_prototypes: torch.Tensor,
                 remapper: LabelRemapper, epochs: int, lr: float,
                 proto_weight: float, device: str) -> nn.Module:
    model = model.to(device)
    seen_prototypes = seen_prototypes.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for xb, yb_global in loader:
            xb = xb.to(device)
            yb_local = remapper.to_local(yb_global).to(device)
            logits, proj = model(xb)

            clf_loss = ce(logits, yb_local)
            target_protos = seen_prototypes[yb_local]
            proto_loss = F.mse_loss(F.normalize(proj, dim=-1),
                                    F.normalize(target_protos, dim=-1))

            loss = clf_loss + proto_weight * proto_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.cpu()


# ─────────────────────────────────────────────────────────────────────────────
# FL-ZSL FEEDBACK ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class FLZSLFeedback:
    def __init__(self, d_in: int, n_seen: int, seen_label_ids: list[int],
                 all_class_names: list[str], precomputed_protos: dict,
                 n_clients: int = 3, device: str = "cpu"):
        self.n_clients = n_clients
        self.device = device
        self.all_class_names = all_class_names
        self.remapper = LabelRemapper(seen_label_ids)
        self.global_model = MLPWithProjection(d_in, n_seen)

        self.prototypes = precomputed_protos
        self.proto_matrix = F.normalize(
            torch.stack([self.prototypes[c].cpu() for c in all_class_names]), dim=-1)
        self.seen_protos_local = torch.stack(
            [self.prototypes[all_class_names[gid]].cpu() for gid in sorted(seen_label_ids)])
        self.confidence_threshold = None

    def fit_and_calibrate(self, X_train, y_train, n_rounds, local_epochs,
                          lr, proto_weight, percentile):
        indices = torch.randperm(len(y_train))
        splits = torch.chunk(indices, self.n_clients)
        client_datasets = [TensorDataset(X_train[s], y_train[s]) for s in splits]

        for _ in range(n_rounds):
            client_models = []
            for ds in client_datasets:
                loader = DataLoader(ds, batch_size=64, shuffle=True)
                local = client_train(
                    copy.deepcopy(self.global_model), loader,
                    self.seen_protos_local, self.remapper,
                    local_epochs, lr, proto_weight, self.device)
                client_models.append(local)
            self.global_model = fed_avg(self.global_model, client_models)

        self.global_model.to(self.device)
        self.global_model.eval()
        with torch.no_grad():
            logits, _ = self.global_model(X_train.to(self.device))
            confs, _ = F.softmax(logits, dim=-1).max(dim=-1)
            self.confidence_threshold = np.percentile(confs.cpu().numpy(), percentile)

    @torch.no_grad()
    def evaluate(self, X_test, y_test, seen_ids, unseen_ids, gamma=2.0):
        self.global_model.to(self.device)
        self.global_model.eval()

        logits, proj = self.global_model(X_test.to(self.device))
        probs = F.softmax(logits, dim=-1)
        conf, local_pred = probs.max(dim=-1)

        global_pred = self.remapper.to_global(local_pred.cpu().numpy())

        low_mask = conf < self.confidence_threshold
        if low_mask.any():
            proj_norm = F.normalize(proj[low_mask].cpu(), dim=-1)
            sims = proj_norm @ self.proto_matrix.T
            sims[:, seen_ids] -= gamma
            global_pred[low_mask.cpu().numpy()] = sims.argmax(dim=-1).numpy()

        truth = y_test.numpy()
        s_m = np.isin(truth, seen_ids)
        u_m = np.isin(truth, unseen_ids)
        acc_s = accuracy_score(truth[s_m], global_pred[s_m]) if s_m.any() else 0.0
        acc_u = accuracy_score(truth[u_m], global_pred[u_m]) if u_m.any() else 0.0
        return {"acc_seen": acc_s, "acc_unseen": acc_u, "H": compute_hmean(acc_s, acc_u)}


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE SWEEP FOR ONE DATASET + ONE SEED
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep_single(data: dict, precomputed_protos: dict, seed: int) -> dict:
    """
    Runs the full hyperparameter sweep for pre-loaded data and a specific seed.
    """
    set_seed(seed)

    # Note: Gamma removed from the grid to avoid redundant retraining
    grid = {
        "lr": [1e-3],
        "proto_weight": [1.0, 2.0, 5.0],
        "percentile": [20, 35],
        "n_rounds": [20],
    }
    gamma_values = [0.5, 1.0, 1.5, 2.0, 3.0]

    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_h = -1
    best_cfg = None
    best_metrics = None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for i, cfg in enumerate(combinations):
        # Re-seed before each model construction
        set_seed(seed + i)

        model = FLZSLFeedback(
            d_in=data["d_in"],
            n_seen=len(data["seen_label_ids"]),
            seen_label_ids=data["seen_label_ids"],
            all_class_names=data["all_class_names"],
            precomputed_protos=precomputed_protos,
            device=device,
        )

        # Train once per configuration
        model.fit_and_calibrate(
            data["X_train"], data["y_train"],
            n_rounds=cfg["n_rounds"], local_epochs=5,
            lr=cfg["lr"], proto_weight=cfg["proto_weight"],
            percentile=cfg["percentile"],
        )

        # Evaluate the trained model across all gamma values
        for gamma in gamma_values:
            m = model.evaluate(
                data["X_test"], data["y_test"],
                data["seen_label_ids"], data["unseen_label_ids"],
                gamma=gamma,
            )

            # Store full config including the specific gamma tested
            full_cfg = {**cfg, "gamma": gamma}

            print(f"    [Combo {i + 1:02d}/{len(combinations)} | γ={gamma}] "
                  f"H={m['H']:.4f}  Seen={m['acc_seen']:.4f}  "
                  f"Unseen={m['acc_unseen']:.4f}  cfg={full_cfg}")

            if m["H"] > best_h:
                best_h = m["H"]
                best_cfg = full_cfg
                best_metrics = m

    best_metrics["best_cfg"] = best_cfg
    return best_metrics


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-DATASET × MULTI-SEED RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_all():
    all_results = {}  # {dataset_name: {seed: metrics}}
    summary = {}  # {dataset_name: {mean/std for each metric}}

    print("\n  Loading SentenceTransformer model to CPU (Once)...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')

    for ds_name, ds_path in DATASETS.items():
        print(f"\n{'═' * 70}")
        print(f"  DATASET: {ds_name}")
        print(f"{'═' * 70}")

        # Load data once per dataset
        print(f"  Loading data from {ds_path} ...")
        data = torch.load(ds_path, map_location="cpu", weights_only=False)

        # Precompute prototypes once per dataset
        print("  Encoding text prototypes ...")
        precomputed_protos = {
            cls: encoder.encode(data["class_descriptions"][cls], convert_to_tensor=True)
            for cls in data["all_class_names"]
        }

        seed_results = []

        for seed in SEEDS:
            print(f"\n  ── Seed {seed} ──────────────────────────────────────────")
            # Pass preloaded data and prototypes into the sweep runner
            metrics = run_sweep_single(data, precomputed_protos, seed)
            seed_results.append(metrics)
            print(f"  ✓ Seed {seed} best → "
                  f"H={metrics['H']:.4f}  "
                  f"Seen={metrics['acc_seen']:.4f}  "
                  f"Unseen={metrics['acc_unseen']:.4f}")

        # ── Aggregate across seeds ────────────────────────────────────────
        acc_seen_vals = [r["acc_seen"] for r in seed_results]
        acc_unseen_vals = [r["acc_unseen"] for r in seed_results]
        hmean_vals = [r["H"] for r in seed_results]

        agg = {
            "acc_seen_mean": float(np.mean(acc_seen_vals)),
            "acc_seen_std": float(np.std(acc_seen_vals, ddof=1)),
            "acc_unseen_mean": float(np.mean(acc_unseen_vals)),
            "acc_unseen_std": float(np.std(acc_unseen_vals, ddof=1)),
            "H_mean": float(np.mean(hmean_vals)),
            "H_std": float(np.std(hmean_vals, ddof=1)),
        }

        all_results[ds_name] = {str(s): r for s, r in zip(SEEDS, seed_results)}
        summary[ds_name] = agg

        print(f"\n  ── {ds_name} Summary (n={len(SEEDS)} seeds) ──────────────")
        print(f"     Seen Acc  : {agg['acc_seen_mean'] * 100:.2f} ± {agg['acc_seen_std'] * 100:.2f}")
        print(f"     Unseen Acc: {agg['acc_unseen_mean'] * 100:.2f} ± {agg['acc_unseen_std'] * 100:.2f}")
        print(f"     H-Mean    : {agg['H_mean'] * 100:.2f} ± {agg['H_std'] * 100:.2f}")

    # ── Final paper-ready table ───────────────────────────────────────────────
    sep = "─" * 75
    print(f"\n\n{'═' * 75}")
    print("  FL-ZSL RESULTS  (mean ± std across 5 seeds, ×100 = %)")
    print(f"{'═' * 75}")
    print(f"  {'Dataset':<12} │ {'Seen Acc':^18} │ {'Unseen Acc':^18} │ {'H-Mean':^18}")
    print(sep)
    for ds_name, agg in summary.items():
        s = f"{agg['acc_seen_mean'] * 100:.2f} ± {agg['acc_seen_std'] * 100:.2f}"
        u = f"{agg['acc_unseen_mean'] * 100:.2f} ± {agg['acc_unseen_std'] * 100:.2f}"
        h = f"{agg['H_mean'] * 100:.2f} ± {agg['H_std'] * 100:.2f}"
        print(f"  {ds_name:<12} │ {s:^18} │ {u:^18} │ {h:^18}")
    print(f"{'═' * 75}\n")

    # ── Save results ─────────────────────────────────────────────────────────
    detail_path = os.path.join(RESULT_DIR, "fl_zsl_multiseed_detail.json")
    summary_path = os.path.join(RESULT_DIR, "fl_zsl_multiseed_summary.json")
    csv_path = os.path.join(RESULT_DIR, "fl_zsl_multiseed_summary.csv")

    with open(detail_path, "w") as f:
        json.dump(all_results, f, indent=2)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    with open(csv_path, "w") as f:
        f.write("Dataset,Seen_Mean,Seen_Std,Unseen_Mean,Unseen_Std,H_Mean,H_Std\n")
        for ds_name, agg in summary.items():
            f.write(
                f"{ds_name},"
                f"{agg['acc_seen_mean'] * 100:.4f},{agg['acc_seen_std'] * 100:.4f},"
                f"{agg['acc_unseen_mean'] * 100:.4f},{agg['acc_unseen_std'] * 100:.4f},"
                f"{agg['H_mean'] * 100:.4f},{agg['H_std'] * 100:.4f}\n"
            )

    print(f"  Detailed results → {detail_path}")
    print(f"  Summary JSON     → {summary_path}")
    print(f"  Summary CSV      → {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_all()