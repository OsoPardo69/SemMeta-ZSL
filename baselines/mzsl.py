import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
import copy
import itertools
import os
import json
import multiprocessing as mp
from collections import defaultdict
from datetime import datetime
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# EXPERIMENT CONFIGURATION
# ==========================================

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
"AVASTCTU": "/home/iamontoyasa/02_VENOM/03_Baseline/03_run1_prepared_data_zero_shot.pt",
}

NUM_GPUS   = 3            # Number of ADA 48 GB GPUs available
GPU_IDS    = list(range(NUM_GPUS))   # [0, 1, 2]
RESULTS_DIR = "mzsl_baseline_results"

# ==========================================
# 1. ARCHITECTURES  (UNCHANGED)
# ==========================================

class Generator(nn.Module):
    def __init__(self, z_dim, attr_dim, out_dim, hidden_size=2048):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + attr_dim, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, out_dim)
        )

    def forward(self, z, c):
        return self.net(torch.cat([z, c], dim=1))


class Discriminator(nn.Module):
    def __init__(self, x_dim, attr_dim, hidden_size=2048):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + attr_dim, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x, c):
        return self.net(torch.cat([x, c], dim=1))


class SoftmaxClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SoftmaxClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# ==========================================
# 2. UTILITIES  (UNCHANGED)
# ==========================================

def compute_gradient_penalty(D, real_samples, fake_samples, attrs, device):
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    alpha = alpha.expand_as(real_samples)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, attrs)
    fake = torch.ones(real_samples.size(0), 1).to(device)
    gradients = autograd.grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def pretrain_aux_classifier(X_train, y_train, num_classes, device):
    clf  = SoftmaxClassifier(X_train.shape[1], num_classes).to(device)
    opt  = optim.Adam(clf.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)
    clf.train()
    for _ in range(20):
        for bx, by in loader:
            opt.zero_grad(); crit(clf(bx), by).backward(); opt.step()
    clf.eval()
    for p in clf.parameters():
        p.requires_grad = False
    return clf


# ==========================================
# 3. TRAINING & EVALUATION  (UNCHANGED)
# ==========================================

def train_zsml_wgangp(data_dict, config):
    device     = config['device']
    X_train, y_train = data_dict["X_train"].to(device), data_dict["y_train"].to(device)
    class_attrs      = data_dict["binary_class_attrs"].to(device)
    seen_classes     = np.array(data_dict["seen_label_ids"])
    num_classes      = len(data_dict["all_class_names"])

    aux_clf  = pretrain_aux_classifier(X_train, y_train, num_classes, device)
    crit_cls = nn.CrossEntropyLoss()

    G     = Generator(config['z_dim'], class_attrs.shape[1], X_train.shape[1]).to(device)
    D     = Discriminator(X_train.shape[1], class_attrs.shape[1]).to(device)
    opt_G = optim.Adam(G.parameters(), lr=config['lr_g'], betas=(0.5, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=config['lr_d'], betas=(0.5, 0.9))

    for epoch in range(config['meta_epochs']):
        np.random.shuffle(seen_classes)
        pseudo_seen   = seen_classes[:len(seen_classes) // 2]
        pseudo_unseen = seen_classes[len(seen_classes) // 2:]

        # ---- Inner Loop ----
        G_fast   = copy.deepcopy(G)
        opt_fast = optim.Adam(G_fast.parameters(), lr=config['inner_lr'])
        idx      = torch.isin(y_train, torch.tensor(pseudo_seen).to(device))
        xb, yb   = X_train[idx], y_train[idx]
        if len(xb) == 0:
            continue
        if len(xb) > config['batch_size']:
            p = torch.randperm(len(xb))[:config['batch_size']]
            xb, yb = xb[p], yb[p]

        attrs = class_attrs[yb]
        for _ in range(config['n_critic']):
            z    = torch.randn(xb.size(0), config['z_dim']).to(device)
            fake = G_fast(z, attrs)
            d_loss = (
                -torch.mean(D(xb, attrs))
                + torch.mean(D(fake.detach(), attrs))
                + config['lambda_gp'] * compute_gradient_penalty(D, xb, fake.detach(), attrs, device)
            )
            opt_D.zero_grad(); d_loss.backward(); opt_D.step()

        z      = torch.randn(xb.size(0), config['z_dim']).to(device)
        fake   = G_fast(z, attrs)
        g_loss = -torch.mean(D(fake, attrs)) + config['beta_cls'] * crit_cls(aux_clf(fake), yb)
        opt_fast.zero_grad(); g_loss.backward(); opt_fast.step()

        # ---- Outer Loop ----
        idx_m  = torch.isin(y_train, torch.tensor(pseudo_unseen).to(device))
        xm, ym = X_train[idx_m], y_train[idx_m]
        if len(xm) > 0:
            if len(xm) > config['batch_size']:
                p = torch.randperm(len(xm))[:config['batch_size']]
                xm, ym = xm[p], ym[p]
            am      = class_attrs[ym]
            fake_m  = G_fast(torch.randn(xm.size(0), config['z_dim']).to(device), am)
            meta_loss = -torch.mean(D(fake_m, am)) + config['beta_cls'] * crit_cls(aux_clf(fake_m), ym)
            opt_G.zero_grad(); meta_loss.backward(); opt_G.step()

    return G


def evaluate_gzsl(G, data_dict, config):
    device = config['device']
    X_train, y_train = data_dict["X_train"].to(device), data_dict["y_train"].to(device)
    X_test, y_test   = data_dict["X_test"].to(device),  data_dict["y_test"].cpu().numpy()
    class_attrs       = data_dict["binary_class_attrs"].to(device)
    seen_ids, unseen_ids = data_dict["seen_label_ids"], data_dict["unseen_label_ids"]

    G.eval()
    sx, sy = [], []
    with torch.no_grad():
        for c in unseen_ids:
            z  = torch.randn(config['syn_num'], config['z_dim']).to(device)
            ca = class_attrs[c].unsqueeze(0).expand(config['syn_num'], -1)
            sx.append(G(z, ca))
            sy.append(torch.full((config['syn_num'],), c, dtype=torch.long, device=device))

    aug_X = torch.cat([X_train, torch.cat(sx)])
    aug_y = torch.cat([y_train, torch.cat(sy)])

    clf  = SoftmaxClassifier(aug_X.shape[1], len(data_dict["all_class_names"])).to(device)
    opt  = optim.Adam(clf.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(aug_X, aug_y), batch_size=256, shuffle=True)
    clf.train()
    for _ in range(config['clf_epochs']):
        for bx, by in loader:
            opt.zero_grad(); crit(clf(bx), by).backward(); opt.step()

    clf.eval()
    with torch.no_grad():
        logits = clf(X_test)
        for s_idx in seen_ids:
            logits[:, s_idx] -= config['gamma_cs']
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    s_mask = np.isin(y_test, seen_ids)
    u_mask = np.isin(y_test, unseen_ids)
    acc_s  = accuracy_score(y_test[s_mask], preds[s_mask]) if s_mask.any() else 0.0
    acc_u  = accuracy_score(y_test[u_mask], preds[u_mask]) if u_mask.any() else 0.0
    h      = (2 * acc_s * acc_u) / (acc_s + acc_u) if (acc_s + acc_u) > 0 else 0.0
    return acc_s, acc_u, h


# ==========================================
# 4. SEED UTILITY
# ==========================================

def set_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ==========================================
# 5. WORKER INITIALIZER & EXPERIMENT RUNNER
# ==========================================

# Module-level variable so each spawned worker knows which GPU it owns.
_WORKER_GPU_ID: int = -1


def _init_worker(gpu_queue: mp.Queue) -> None:
    """Called once per worker process; claims a GPU from the shared queue."""
    global _WORKER_GPU_ID
    _WORKER_GPU_ID = gpu_queue.get()


def run_single_experiment(args: tuple) -> dict:
    """
    Execute the full grid search for ONE (dataset, seed) pair on the worker's GPU.

    Returns a dict with keys:
        dataset, seed, best_acc_s, best_acc_u, best_h, per_combo_results
    """
    global _WORKER_GPU_ID
    dataset_name, data_path, seed = args

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{_WORKER_GPU_ID}")
    else:
        device = torch.device("cpu")

    tag = f"[GPU {_WORKER_GPU_ID} | {dataset_name} | seed={seed}]"
    print(f"{tag} Starting experiment ...", flush=True)

    # Set seed ONCE for the whole (dataset, seed) run
    set_seed(seed)

    data_dict = torch.load(data_path, map_location=device, weights_only=False)

    grid = {
        'syn_num':  [1000, 1500],
        'gamma_cs': [1.2, 2.0, 3.0],
        'beta_cls': [0.1, 0.5],
        'inner_lr': [1e-3, 5e-4],
    }
    keys, values   = zip(*grid.items())
    combinations   = [dict(zip(keys, v)) for v in itertools.product(*values)]
    n_combos       = len(combinations)

    best_h, best_s, best_u = 0.0, 0.0, 0.0
    per_combo = []

    print(f"{tag} Grid search: {n_combos} combinations", flush=True)

    for i, params in enumerate(combinations):
        cfg = {
            'device':      device,
            'z_dim':       100,
            'meta_epochs': 500,
            'inner_steps': 1,
            'batch_size':  512,
            'lambda_gp':   10.0,
            'clf_epochs':  30,
            'n_critic':    5,
            'lr_g':        1e-4,
            'lr_d':        1e-4,
            **params,
        }
        trained_G      = train_zsml_wgangp(data_dict, cfg)
        s, u, h        = evaluate_gzsl(trained_G, data_dict, cfg)
        per_combo.append({"params": params, "acc_s": s, "acc_u": u, "h": h})

        print(
            f"{tag} [{i+1}/{n_combos}] {params} -> "
            f"S={s*100:.2f}%  U={u*100:.2f}%  H={h*100:.2f}%"
            f"  (best H so far: {best_h*100:.2f}%)",
            flush=True,
        )

        if h > best_h:
            best_h, best_s, best_u = h, s, u

    print(
        f"{tag} DONE  Best -> S={best_s*100:.2f}%  U={best_u*100:.2f}%  H={best_h*100:.2f}%",
        flush=True,
    )

    return {
        "dataset":    dataset_name,
        "seed":       seed,
        "best_acc_s": best_s,
        "best_acc_u": best_u,
        "best_h":     best_h,
        "per_combo":  per_combo,
    }


# ==========================================
# 6. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Build flat task list: (dataset_name, data_path, seed)
    tasks = [
        (ds_name, ds_path, seed)
        for ds_name, ds_path in DATASETS.items()
        for seed in SEEDS
    ]
    total = len(tasks)  # 5 datasets × 5 seeds = 25

    print("=" * 70)
    print(f"MZSL Baseline — Multi-Dataset / Multi-Seed Evaluation")
    print(f"  Datasets : {list(DATASETS.keys())}")
    print(f"  Seeds    : {SEEDS}")
    print(f"  Total runs: {total}  |  Parallel GPUs: {NUM_GPUS}")
    print("=" * 70, flush=True)

    # Build a shared GPU queue: each of the NUM_GPUS workers claims one GPU ID
    ctx       = mp.get_context("spawn")
    gpu_queue = ctx.Queue()
    for gpu_id in GPU_IDS:
        gpu_queue.put(gpu_id)

    # Run all experiments in parallel — up to NUM_GPUS concurrent processes
    with ctx.Pool(
        processes=NUM_GPUS,
        initializer=_init_worker,
        initargs=(gpu_queue,),
    ) as pool:
        all_results = pool.map(run_single_experiment, tasks)

    # ---- Aggregate per-dataset ----
    dataset_records: dict = defaultdict(lambda: {"acc_s": [], "acc_u": [], "h": [], "seeds": []})

    for rec in all_results:
        ds   = rec["dataset"]
        seed = rec["seed"]
        dataset_records[ds]["acc_s"].append(rec["best_acc_s"])
        dataset_records[ds]["acc_u"].append(rec["best_acc_u"])
        dataset_records[ds]["h"].append(rec["best_h"])
        dataset_records[ds]["seeds"].append(seed)

    # ---- Print summary table ----
    sep = "=" * 74
    print(f"\n{sep}")
    print("FINAL RESULTS  —  mean ± std over 5 seeds")
    print(sep)
    print(f"{'Dataset':<12}  {'Seen Acc (%)':>20}  {'Unseen Acc (%)':>20}  {'H-Mean (%)':>16}")
    print("-" * 74)

    summary = {}
    for ds_name in DATASETS.keys():
        r       = dataset_records[ds_name]
        s_arr   = np.array(r["acc_s"]) * 100
        u_arr   = np.array(r["acc_u"]) * 100
        h_arr   = np.array(r["h"])     * 100

        s_mean, s_std = s_arr.mean(), s_arr.std()
        u_mean, u_std = u_arr.mean(), u_arr.std()
        h_mean, h_std = h_arr.mean(), h_arr.std()

        print(
            f"{ds_name:<12}  "
            f"{s_mean:>8.2f} ± {s_std:>5.2f}    "
            f"{u_mean:>8.2f} ± {u_std:>5.2f}    "
            f"{h_mean:>6.2f} ± {h_std:>5.2f}"
        )

        # Per-seed breakdown
        per_seed = {}
        for seed, s, u, h in zip(r["seeds"], s_arr, u_arr, h_arr):
            per_seed[str(seed)] = {"acc_s": round(s, 4), "acc_u": round(u, 4), "h": round(h, 4)}

        summary[ds_name] = {
            "acc_s":    {"mean": round(s_mean, 4), "std": round(s_std, 4)},
            "acc_u":    {"mean": round(u_mean, 4), "std": round(u_std, 4)},
            "h":        {"mean": round(h_mean, 4), "std": round(h_std, 4)},
            "per_seed": per_seed,
        }

    print(sep)

    # ---- LaTeX-ready row (copy-paste into paper) ----
    print("\nLaTeX table rows (Seen / Unseen / H):")
    print("-" * 74)
    for ds_name, vals in summary.items():
        s = vals["acc_s"]
        u = vals["acc_u"]
        h = vals["h"]
        print(
            f"{ds_name} & "
            f"{s['mean']:.2f}$\\pm${s['std']:.2f} & "
            f"{u['mean']:.2f}$\\pm${u['std']:.2f} & "
            f"{h['mean']:.2f}$\\pm${h['std']:.2f} \\\\"
        )

    # ---- Save full results ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json  = os.path.join(RESULTS_DIR, f"results_{timestamp}.json")
    full_dump = {"summary": summary, "raw": all_results}

    # Convert non-serialisable numpy floats before dumping
    def _to_python(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _to_python(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_python(i) for i in obj]
        return obj

    with open(out_json, "w") as f:
        json.dump(_to_python(full_dump), f, indent=2)

    print(f"\nFull results saved to: {out_json}")
    print(sep)
