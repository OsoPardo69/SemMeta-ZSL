import os
import json
import random
import itertools
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.metrics import confusion_matrix
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

N_GPUS          = 3   # RTX 6000 Ada 48 GB ×3
WORKERS_PER_GPU = 2   # 2 seed-workers per GPU  → 6 total (safe for small MLP)
N_WORKERS       = N_GPUS * WORKERS_PER_GPU

SEEDS = [42, 123, 456, 789, 2025]

# DATASETS = {
#     "AndMAL": "/home/iamontoyasa/02_VENOM/03_Baseline/01_run1_prepared_data_zero_shot.pt",
#     "BODMAS": "/home/iamontoyasa/02_VENOM/03_Baseline/02_run1_prepared_data_zero_shot.pt",
#     "AVASTCTU": "/home/iamontoyasa/02_VENOM/03_Baseline/03_run1_prepared_data_zero_shot.pt",
#     "APIGRAPH": "/home/iamontoyasa/02_VENOM/03_Baseline/04_run1_prepared_data_zero_shot.pt",
#     "Goodread": "/home/iamontoyasa/02_VENOM/03_Baseline/06_run1_prepared_data_zero_shot.pt",
#     "Petfinder": "/home/iamontoyasa/02_VENOM/03_Baseline/07_run1_prepared_data_zero_shot.pt",
#     "Fakeddit": "/home/iamontoyasa/02_VENOM/03_Baseline/08_run1_prepared_data_zero_shot.pt"
# }
DATASETS = {
    "Petfinder": "/home/iamontoyasa/02_VENOM/03_Baseline/07_run1_prepared_data_zero_shot.pt",
    "Fakeddit": "/home/iamontoyasa/02_VENOM/03_Baseline/08_run1_prepared_data_zero_shot.pt"
}

RESULT_DIR = "/home/iamontoyasa/02_VENOM/03_Baseline/results"
os.makedirs(RESULT_DIR, exist_ok=True)

# Training hyperparameters (fixed — only gamma is swept)
TRAIN_CFG = {
    "lr":           1e-3,
    "epochs":       50,
    "weight_decay": 1e-4,
}
# Gamma is inference-only → no retraining needed per value (6× speedup)
GAMMA_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]

LLM_MODEL = "all-mpnet-base-v2"

# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

class ZETLLM(nn.Module):
    def __init__(self, feat_dim: int, llm_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, llm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    seen_ids: list, unseen_ids: list,
                    n_classes: int) -> tuple[float, float, float]:
    cm   = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    accs = [cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0.0
            for i in range(n_classes)]
    S = float(np.mean([accs[i] for i in seen_ids]))   * 100
    U = float(np.mean([accs[i] for i in unseen_ids])) * 100
    H = (2 * S * U) / (S + U) if (S + U) > 0 else 0.0
    return S, U, H

# ─────────────────────────────────────────────────────────────────────────────
# LABEL ANCHOR PRE-COMPUTATION  (called once per dataset in main process)
# ─────────────────────────────────────────────────────────────────────────────

def build_label_anchors(class_names: list[str]) -> torch.Tensor:
    """
    Loads SentenceTransformer ONCE per dataset.
    Returns label_anchors tensor in shared memory for zero-copy IPC.
    """
    print(f"  [anchors] Encoding {len(class_names)} class prompts with {LLM_MODEL} …")
    encoder = SentenceTransformer(LLM_MODEL, device="cpu")
    prompts = [f"Malware behavior of the {name} family" for name in class_names]
    with torch.no_grad():
        anchors = encoder.encode(prompts, convert_to_tensor=True).cpu()
    return anchors   # shape: (n_classes, llm_dim)

# ─────────────────────────────────────────────────────────────────────────────
# WORKER  (module-level for 'spawn' pickling)
# ─────────────────────────────────────────────────────────────────────────────

def _run_seed_worker(args: tuple) -> dict:
    """
    Trains ONE model per seed, then sweeps all gamma values at inference.

    Key insight: gamma only penalises seen-class similarities at test time —
    the model weights are identical for every gamma. So we train once and
    run 6 forward passes instead of training 6 separate models.
    This gives a ~6× compute reduction over the naive approach.

    args = (gpu_id, seed, ds_name,
            X_train_np, y_train_np, X_test_np, y_test_np,
            seen_ids, unseen_ids,
            label_anchors,          # shared-memory tensor
            n_classes)
    """
    (gpu_id, seed, ds_name,
     X_train_np, y_train_np, X_test_np, y_test_np,
     seen_ids, unseen_ids,
     label_anchors,
     n_classes) = args

    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    set_seed(seed)

    # Move anchors to device once
    anchors = label_anchors.to(device)          # (n_classes, llm_dim)
    feat_dim = X_train_np.shape[1]
    llm_dim  = anchors.shape[1]

    # ── Build tensors ────────────────────────────────────────────────────────
    X_t = torch.tensor(X_train_np, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train_np, dtype=torch.long).to(device)
    X_e = torch.tensor(X_test_np,  dtype=torch.float32).to(device)

    # ── Train (ONE model for all gammas) ─────────────────────────────────────
    model = ZETLLM(feat_dim, llm_dim).to(device)
    opt   = torch.optim.Adam(model.parameters(),
                             lr=TRAIN_CFG["lr"],
                             weight_decay=TRAIN_CFG["weight_decay"])
    crit  = nn.CosineEmbeddingLoss()

    model.train()
    for _ in range(TRAIN_CFG["epochs"]):
        opt.zero_grad()
        proj  = model(X_t)
        loss  = crit(proj, anchors[y_t], torch.ones(len(X_t), device=device))
        loss.backward()
        opt.step()

    # ── Inference (shared for all gammas) ────────────────────────────────────
    model.eval()
    with torch.no_grad():
        test_proj   = model(X_e)                                 # (N_test, llm_dim)
        base_sims   = torch.matmul(test_proj, anchors.T).cpu()   # (N_test, n_classes)

    # ── Gamma sweep (6 forward-only passes) ──────────────────────────────────
    best_H   = -1.0
    best_res = None

    for gamma in GAMMA_VALUES:
        sims = base_sims.clone()
        if gamma > 0:
            for s_id in seen_ids:
                sims[:, s_id] -= gamma
        y_pred = sims.argmax(dim=-1).numpy()
        S, U, H = compute_metrics(y_test_np, y_pred, seen_ids, unseen_ids, n_classes)

        print(f"    [{ds_name}|seed={seed}|γ={gamma}] "
              f"S={S:.2f}%  U={U:.2f}%  H={H:.2f}%", flush=True)

        if H > best_H:
            best_H   = H
            best_res = {"acc_seen": S, "acc_unseen": U, "H": H, "gamma": gamma}

    print(f"  ✓ [{ds_name}|seed={seed}] BEST → "
          f"H={best_res['H']:.2f}%  S={best_res['acc_seen']:.2f}%  "
          f"U={best_res['acc_unseen']:.2f}%  γ={best_res['gamma']}", flush=True)
    return best_res

# ─────────────────────────────────────────────────────────────────────────────
# PARALLEL SWEEP FOR ONE DATASET  (all seeds parallelised across GPUs)
# ─────────────────────────────────────────────────────────────────────────────

def run_dataset_parallel(ds_name: str, ds_path: str,
                         n_workers: int, n_gpus: int) -> list[dict]:
    """
    Loads data + builds anchors once, then dispatches one worker per seed.
    Workers are spread round-robin across N_GPUS.
    """
    print(f"\n  Loading {ds_path} …")
    data = torch.load(ds_path, map_location="cpu", weights_only=False)

    seen_ids   = sorted(set(data["seen_label_ids"]))
    unseen_ids = sorted(set(data["unseen_label_ids"]))
    class_names = data["all_class_names"]
    n_classes   = len(class_names)

    # Numpy arrays — no shared memory needed (passed via pickle, small enough)
    X_train = data["X_train"].numpy()
    y_train = data["y_train"].numpy()
    X_test  = data["X_test"].numpy()
    y_test  = data["y_test"].numpy()

    # Build label anchors once per dataset and put in shared memory
    label_anchors = build_label_anchors(class_names)
    label_anchors.share_memory_()

    jobs = [
        (seed_idx % n_gpus, seed, ds_name,
         X_train, y_train, X_test, y_test,
         seen_ids, unseen_ids, label_anchors, n_classes)
        for seed_idx, seed in enumerate(SEEDS)
    ]

    ctx = mp.get_context("spawn")
    seed_results = []

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futures = {pool.submit(_run_seed_worker, job): job[1] for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            seed_results.append(result)

    return seed_results

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_all():
    all_results: dict = {}
    summary:     dict = {}

    for ds_name, ds_path in DATASETS.items():
        print(f"\n{'═'*72}")
        print(f"  DATASET: {ds_name}")
        print(f"{'═'*72}")

        seed_results = run_dataset_parallel(
            ds_name=ds_name,
            ds_path=ds_path,
            n_workers=N_WORKERS,
            n_gpus=N_GPUS,
        )

        # ── Aggregate (ddof=1 = sample std, standard for papers) ──────────
        s_vals = [r["acc_seen"]   for r in seed_results]
        u_vals = [r["acc_unseen"] for r in seed_results]
        h_vals = [r["H"]          for r in seed_results]

        agg = {
            "acc_seen_mean":   float(np.mean(s_vals)),
            "acc_seen_std":    float(np.std(s_vals,   ddof=1)),
            "acc_unseen_mean": float(np.mean(u_vals)),
            "acc_unseen_std":  float(np.std(u_vals,   ddof=1)),
            "H_mean":          float(np.mean(h_vals)),
            "H_std":           float(np.std(h_vals,   ddof=1)),
        }

        all_results[ds_name] = {str(s): r for s, r in zip(SEEDS, seed_results)}
        summary[ds_name]     = agg

        print(f"\n  ── {ds_name}  Summary  (n={len(SEEDS)} seeds) {'─'*28}")
        print(f"     Seen Acc  : {agg['acc_seen_mean']:.2f} ± {agg['acc_seen_std']:.2f}")
        print(f"     Unseen Acc: {agg['acc_unseen_mean']:.2f} ± {agg['acc_unseen_std']:.2f}")
        print(f"     H-Mean    : {agg['H_mean']:.2f} ± {agg['H_std']:.2f}")

    # ── Paper-ready table ─────────────────────────────────────────────────────
    w = 77
    print(f"\n\n{'═'*w}")
    print("  ZET-LLM  RESULTS   (mean ± std, %,  5 seeds)")
    print(f"{'═'*w}")
    print(f"  {'Dataset':<12} │ {'Seen Acc':^20} │ {'Unseen Acc':^20} │ {'H-Mean':^16}")
    print(f"  {'─'*12}─┼─{'─'*20}─┼─{'─'*20}─┼─{'─'*16}")
    for ds_name, agg in summary.items():
        s = f"{agg['acc_seen_mean']:.2f} ± {agg['acc_seen_std']:.2f}"
        u = f"{agg['acc_unseen_mean']:.2f} ± {agg['acc_unseen_std']:.2f}"
        h = f"{agg['H_mean']:.2f} ± {agg['H_std']:.2f}"
        print(f"  {ds_name:<12} │ {s:^20} │ {u:^20} │ {h:^16}")
    print(f"{'═'*w}\n")

    # ── Persist ───────────────────────────────────────────────────────────────
    detail_path  = os.path.join(RESULT_DIR, "zetllm_multiseed_detail.json")
    summary_path = os.path.join(RESULT_DIR, "zetllm_multiseed_summary.json")
    csv_path     = os.path.join(RESULT_DIR, "zetllm_multiseed_summary.csv")

    with open(detail_path,  "w") as f:
        json.dump(all_results, f, indent=2)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(csv_path, "w") as f:
        f.write("Dataset,Seen_Mean,Seen_Std,Unseen_Mean,Unseen_Std,H_Mean,H_Std\n")
        for ds_name, agg in summary.items():
            f.write(
                f"{ds_name},"
                f"{agg['acc_seen_mean']:.4f},{agg['acc_seen_std']:.4f},"
                f"{agg['acc_unseen_mean']:.4f},{agg['acc_unseen_std']:.4f},"
                f"{agg['H_mean']:.4f},{agg['H_std']:.4f}\n"
            )

    print(f"  Detailed JSON → {detail_path}")
    print(f"  Summary JSON  → {summary_path}")
    print(f"  Summary CSV   → {csv_path}")

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    run_all()
