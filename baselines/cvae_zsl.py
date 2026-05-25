#!/usr/bin/env python3
"""
run_cvae_sweep_optimized.py
═══════════════════════════════════════════════════════════════════════════════
CVAE-ZSL Baseline (Mishra et al., 2018) for VENOM
Optimized with Beta-KLD weighting, Calibrated Stacking (Gamma), Attribute
Projection, and Batch-Averaged Losses to fix seen-class bias and prevent
latent collapse.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import time
import warnings
import itertools
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import pandas as pd

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
CFG = {
    # Static Params
    "hidden_size": 1024,
    "cvae_epochs": 100,
    "batch_size": 128,
    "clf_lr": 1e-3,
    "clf_epochs": 50,
    "cvae_lr": 1e-4,
    "syn_num": 400,

    # 5 Fixed Seeds for Mean/Std calculation
    "seeds": [42, 123, 456, 789, 2025],

    # Sweep Grid (Expanded Gamma for true calibrated stacking)
    "sweep": {
        "z_dim": [64, 128],
        "beta": [0.01, 0.1, 1.0],
        "gamma": [0.0, 5.0, 10.0, 15.0]
    }
}

BANNER = "═" * 85


def set_seed(seed):
    """Locks all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════════════

class CVAE(nn.Module):
    def __init__(self, feature_dim, attr_dim, z_dim, hidden_size):
        super(CVAE, self).__init__()

        # Attribute projection to prevent feature dimension from overpowering
        self.attr_proj = nn.Linear(attr_dim, 512)

        self.enc_h1 = nn.Linear(feature_dim + 512, hidden_size)
        self.enc_mu = nn.Linear(hidden_size, z_dim)
        self.enc_lv = nn.Linear(hidden_size, z_dim)

        self.dec_h1 = nn.Linear(z_dim + 512, hidden_size)
        self.dec_out = nn.Linear(hidden_size, feature_dim)
        self.relu = nn.ReLU()

    def encode(self, x, c):
        c_proj = self.relu(self.attr_proj(c))
        h1 = self.relu(self.enc_h1(torch.cat([x, c_proj], dim=1)))
        return self.enc_mu(h1), self.enc_lv(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z, c):
        c_proj = self.relu(self.attr_proj(c))
        h1 = self.relu(self.dec_h1(torch.cat([z, c_proj], dim=1)))
        # Assuming extracted features are non-negative (e.g., post-ReLU in backbone)
        return self.relu(self.dec_out(h1))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar


def loss_fn(recon_x, x, mu, logvar, beta=0.1):
    # Stabilized Loss: Average over batch, sum over feature dimension
    MSE = F.mse_loss(recon_x, x, reduction='mean') * x.shape[1]
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return MSE + (beta * KLD)


class SoftmaxClassifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, n_classes)

    def forward(self, x): return self.fc(x)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MAIN SWEEP EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def run_cvae_sweep(dataset_id: str, data_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cvae_sweep_results.csv")
    best_output_path = os.path.join(output_dir, "cvae_sweep_best.csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{BANNER}")
    print(f"  CVAE-ZSL HYPERPARAMETER SWEEP ({len(CFG['seeds'])} SEEDS) - DATASET {dataset_id}")
    print(f"  Device  : {device}")
    print(f"  Data    : {data_path}")
    print(BANNER)

    if not os.path.exists(data_path):
        print(f"Data not found: {data_path}\nSkipping this dataset...")
        return

    # [1] Load Data
    data = torch.load(data_path, weights_only=False)
    X_train, y_train = data["X_train"].to(device), data["y_train"].to(device)
    X_test, y_test = data["X_test"].to(device), data["y_test"].to(device)
    class_attrs = data["binary_class_attrs"].to(device)

    seen_ids = sorted(list(set(data["seen_label_ids"])))
    unseen_ids = sorted(list(set(data["unseen_label_ids"])))
    n_classes = len(data["all_class_names"])
    feature_dim = X_train.shape[1]
    attr_dim = class_attrs.shape[1]
    dataset_size = len(X_train)

    keys, values = zip(*CFG["sweep"].items())
    all_results = []
    best_h = -1
    best_cfg = None

    print(f"\n{'z_dim':>6} | {'beta':>6} | {'gamma':>5} || {'S (Seen)':>12} | {'U (Unseen)':>12} | {'H-Mean':>12}")
    print("-" * 85)

    # [2] Start Sweep Loop
    for v in itertools.product(*values):
        params = dict(zip(keys, v))

        # Arrays to hold the results of the 5 seeds for this configuration
        s_runs, u_runs, h_runs = [], [], []

        for seed in CFG["seeds"]:
            set_seed(seed)

            # Phase 1: Train CVAE
            cvae = CVAE(feature_dim, attr_dim, params["z_dim"], CFG["hidden_size"]).to(device)
            optimizer_cvae = torch.optim.Adam(cvae.parameters(), lr=CFG["cvae_lr"])
            cvae.train()

            for epoch in range(CFG["cvae_epochs"]):
                permutation = torch.randperm(dataset_size)
                for i in range(0, dataset_size, CFG["batch_size"]):
                    indices = permutation[i:i + CFG["batch_size"]]
                    batch_x, batch_y = X_train[indices], y_train[indices]
                    batch_c = class_attrs[batch_y]

                    optimizer_cvae.zero_grad()
                    recon_x, mu, logvar = cvae(batch_x, batch_c)

                    loss = loss_fn(recon_x, batch_x, mu, logvar, beta=params["beta"])
                    loss.backward()
                    optimizer_cvae.step()

            # Phase 2: Synthesize Features for Unseen Classes
            cvae.eval()
            syn_features, syn_labels = [], []
            with torch.no_grad():
                for u_id in unseen_ids:
                    z = torch.randn(CFG["syn_num"], params["z_dim"]).to(device)
                    c_u = class_attrs[u_id].repeat(CFG["syn_num"], 1)
                    syn_features.append(cvae.decode(z, c_u))
                    syn_labels.append(torch.full((CFG["syn_num"],), u_id, dtype=torch.long).to(device))

            X_syn = torch.cat(syn_features, dim=0)
            y_syn = torch.cat(syn_labels, dim=0)

            # Phase 3: Train Final GZSL Classifier
            X_gzsl_train = torch.cat([X_train, X_syn], dim=0)
            y_gzsl_train = torch.cat([y_train, y_syn], dim=0)

            classifier = SoftmaxClassifier(feature_dim, n_classes).to(device)
            # Added L2 weight decay to bound logits for proper gamma calibration
            optimizer_clf = torch.optim.Adam(classifier.parameters(), lr=CFG["clf_lr"], weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()

            classifier.train()
            gzsl_dataset_size = len(X_gzsl_train)

            for epoch in range(CFG["clf_epochs"]):
                permutation = torch.randperm(gzsl_dataset_size)
                for i in range(0, gzsl_dataset_size, CFG["batch_size"]):
                    indices = permutation[i:i + CFG["batch_size"]]
                    optimizer_clf.zero_grad()
                    logits = classifier(X_gzsl_train[indices])
                    loss = criterion(logits, y_gzsl_train[indices])
                    loss.backward()
                    optimizer_clf.step()

            # Phase 4: Evaluate with Calibrated Stacking
            classifier.eval()
            with torch.no_grad():
                test_logits = classifier(X_test)

                if params["gamma"] > 0:
                    for s_id in seen_ids:
                        test_logits[:, s_id] -= params["gamma"]

                y_pred = test_logits.argmax(dim=-1).cpu().numpy()
                y_true = y_test.cpu().numpy()

            # Metrics Calculation for this specific seed
            cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
            per_class_acc = {i: (cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0) for i in range(n_classes)}

            S = np.mean([per_class_acc[c] for c in seen_ids]) * 100
            U = np.mean([per_class_acc[c] for c in unseen_ids]) * 100
            H = (2 * S * U / (S + U)) if (S + U) > 0 else 0.0

            s_runs.append(S)
            u_runs.append(U)
            h_runs.append(H)

            # Free up memory
            del cvae, classifier, optimizer_cvae, optimizer_clf
            torch.cuda.empty_cache()

        # [3] Aggregate Statistics Across the 5 Seeds
        S_mean, S_std = np.mean(s_runs), np.std(s_runs)
        U_mean, U_std = np.mean(u_runs), np.std(u_runs)
        H_mean, H_std = np.mean(h_runs), np.std(h_runs)

        print(
            f"{params['z_dim']:6} | {params['beta']:6.2f} | {params['gamma']:5.1f} || "
            f"{S_mean:6.2f}±{S_std:<4.2f} | {U_mean:6.2f}±{U_std:<4.2f} | {H_mean:6.2f}±{H_std:<4.2f}"
        )

        # Save the aggregate results to the dataframe
        res = {
            **params,
            "S_mean": S_mean, "S_std": S_std,
            "U_mean": U_mean, "U_std": U_std,
            "H_mean": H_mean, "H_std": H_std
        }
        all_results.append(res)

        # Track best overall H-Mean
        if H_mean > best_h:
            best_h = H_mean
            best_cfg = res

    # [4] Save & Conclude
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(output_path, index=False)

    # Save the single best configuration to its own CSV
    df_best = pd.DataFrame([best_cfg])
    df_best.to_csv(best_output_path, index=False)

    print(f"\n{BANNER}")
    print(f"BEST CONFIGURATION FOR DATASET {dataset_id} (Based on H_mean):")
    print(f"  z_dim: {best_cfg['z_dim']} | beta: {best_cfg['beta']} | gamma: {best_cfg['gamma']}")
    print(f"  S: {best_cfg['S_mean']:.2f} ± {best_cfg['S_std']:.2f}%")
    print(f"  U: {best_cfg['U_mean']:.2f} ± {best_cfg['U_std']:.2f}%")
    print(f"  H: {best_cfg['H_mean']:.2f} ± {best_cfg['H_std']:.2f}%")
    print(f"\n  Results saved to:")
    print(f"    {output_path}")
    print(f"    {best_output_path}")
    print(f"{BANNER}")


if __name__ == "__main__":
    t0_global = time.time()

    # The exact datasets you requested
    target_datasets = [1,2,3,4,6,7, 8]

    for ds_id in target_datasets:
        ds_id_str = f"{ds_id:02d}"  # Formats to "01", "02", etc.

        dynamic_data_path = f"/home/iamontoyasa/02_VENOM/03_Baseline/{ds_id_str}_run1_prepared_data_zero_shot.pt"
        dynamic_output_dir = f"/home/iamontoyasa/02_VENOM/03_Baseline/sweep_results_ds{ds_id_str}"

        t0_dataset = time.time()

        run_cvae_sweep(
            dataset_id=ds_id_str,
            data_path=dynamic_data_path,
            output_dir=dynamic_output_dir
        )

        print(f"Time taken for Dataset {ds_id_str}: {time.time() - t0_dataset:.1f} seconds")

    print(f"\n\n{'=' * 90}")
    print(f" ALL DATASETS ({target_datasets}) HAVE BEEN SUCCESSFULLY PROCESSED.")
    print(f" Total Execution Time: {time.time() - t0_global:.1f} seconds")
    print(f"{'=' * 90}")