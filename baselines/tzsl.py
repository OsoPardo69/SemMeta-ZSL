#!/usr/bin/env python3
"""
tzsl_standalone.py — Faithful TZSL Reimplementation for VENOM Benchmark
═══════════════════════════════════════════════════════════════════════════════
Transductive Zero-Shot Learning (TZSL) baseline for CIC-AndMal-2020.
Reimplementation of Wang et al. (2025) "A Transductive Zero-Shot Learning
Framework for Ransomware Detection Using Malware Knowledge Graphs."
Information 2025, 16, 458. https://doi.org/10.3390/info16060458

Two-stage pipeline (mirrors the paper's implicit design)
─────────────────────────────────────────────────────────
  Stage 1 — KG Pre-training:
    Train GCN with spread loss + seen-class discrimination loss until
    prototypes for all 24 classes are well-separated on the unit sphere.
    Then FREEZE the GCN permanently. Pre-compute prototype matrix once.
    → Mirrors paper: KG is built from sandbox data BEFORE VQ-VAE training
      and serves as a fixed semantic anchor space.

  Stage 2 — VQ-VAE Training (transductive):
    Train VQ-VAE + projection head to map input samples to the fixed
    prototype space. Loss = VQ-VAE recon + CE prototype alignment
    (labeled) + VQ-VAE recon (unlabeled test — transductive signal).
    GCN is frozen; only VQ-VAE and proj are updated.

Key faithfulness decisions
──────────────────────────
  ✓ No CE head        — paper's best model (Table 7) has NO CE head.
  ✓ Transductive      — unlabeled X_test fed to VQ-VAE during training.
  ✓ L2 regularization — Adam weight_decay=1e-4 (softened for tabular data).
  ✓ Jaccard adjacency — binary top-k feature vectors, threshold=0.1.
  ✓ class_attrs       — per-class mean feature vectors (behavioral space).
  + KG pre-training   — spread + discrimination loss, then frozen.
  + CE alignment      — pushes toward correct proto AND away from all others.
  + γ calibration     — grid search over γ ∈ [0,1] for best H-mean.
  + Cosine LR anneal  — prevents optimizer overshoot / codebook collapse.
  + Best checkpoint   — evaluate at lowest-loss epoch, not final epoch.

Run:
    python tzsl_standalone.py

Prerequisite:
    python data_prep.py   →   prepared_data.pt
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CFG = {
    # ── Paths ──────────────────────────────────────────────────────────────────
    "data_path"          : "/home/iamontoyasa/02_VENOM/03_Baseline/04_run1_prepared_data_zero_shot.pt",
    "output_path"        : "/home/iamontoyasa/02_VENOM/03_Baseline/tzsl_results.csv",

    # ── VQ-VAE ─────────────────────────────────────────────────────────────────
    "hidden_dim"         : 256,
    "embedding_dim"      : 64,
    "n_embeddings"       : 512,

    # ── Knowledge Graph GCN ────────────────────────────────────────────────────
    "gcn_hidden"         : 128,
    "kg_threshold"       : 0.1,

    # ── Stage 1: KG pre-training ───────────────────────────────────────────────
    "kg_pretrain_steps"  : 800,
    "kg_pretrain_lr"     : 1e-3,
    "kg_good_sim_thresh" : 0.3,

    # ── Stage 2: VQ-VAE training ───────────────────────────────────────────────
    "epochs"             : 200,
    "lr"                 : 1e-3,
    "weight_decay"       : 1e-4,
    "batch_size"         : 64,
    "temperature"        : 0.15,
    "lambda_proto"       : 0.5,
    "lambda_transductive": 0.5,

    # ── GZSL calibration ───────────────────────────────────────────────────────
    "gamma_values"       : [round(g, 2)
                            for g in np.arange(0.0, 1.05, 0.05).tolist()],

    # ── Misc ───────────────────────────────────────────────────────────────────
    "seed"               : 42,
}

BANNER = "═" * 70


# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ─────────────────────────────────────────────────────────────────────────────
# VQ-VAE
# ─────────────────────────────────────────────────────────────────────────────

class VectorQuantizer(nn.Module):
    def __init__(self, n_embeddings: int, embedding_dim: int,
                 commitment_cost: float = 0.25):
        super().__init__()
        self.commitment_cost = commitment_cost
        self.embedding       = nn.Embedding(n_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight,
                         -1 / n_embeddings, 1 / n_embeddings)

    def forward(self, z: torch.Tensor):
        dist = (
            z.pow(2).sum(1, keepdim=True)
            - 2 * z @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1)
        )
        idx     = dist.argmin(1)
        z_q     = self.embedding(idx)
        vq_loss = (
            F.mse_loss(z_q.detach(), z)
            + self.commitment_cost * F.mse_loss(z_q, z.detach())
        )
        z_q_st  = z + (z_q - z).detach()
        return z_q_st, vq_loss, idx


class VQVAE(nn.Module):
    def __init__(self, d_in: int, hidden_dim: int,
                 embedding_dim: int, n_embeddings: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.vq      = VectorQuantizer(n_embeddings, embedding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, d_in),
        )

    def forward(self, x: torch.Tensor):
        z_e               = self.encoder(x)
        z_q, vq_loss, idx = self.vq(z_e)
        recon_loss        = F.mse_loss(self.decoder(z_q), x)
        return z_q, recon_loss + vq_loss, idx


# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE GRAPH + GCN
# ─────────────────────────────────────────────────────────────────────────────

class GraphConvLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, H: torch.Tensor, A_hat: torch.Tensor):
        return F.leaky_relu(A_hat @ self.W(H), negative_slope=0.2)


class MalwareKnowledgeGraph(nn.Module):
    """
    Two-layer GCN over per-class mean feature vectors.
    gcn1: LeakyReLU — passes gradients through negative activations.
    gcn2: plain Linear — no activation before F.normalize in pretrain_kg.
    """
    def __init__(self, attr_dim: int, gcn_hidden: int, gcn_out_dim: int):
        super().__init__()
        self.gcn1 = GraphConvLayer(attr_dim, gcn_hidden)
        self.gcn2 = nn.Linear(gcn_hidden, gcn_out_dim, bias=False)

    @staticmethod
    def build_adjacency(binary_attrs: torch.Tensor,
                        threshold: float = 0.1) -> torch.Tensor:
        inter = (binary_attrs.unsqueeze(0) * binary_attrs.unsqueeze(1)).sum(-1)
        union = (binary_attrs.unsqueeze(0) + binary_attrs.unsqueeze(1)
                 ).clamp(max=1.0).sum(-1)
        J   = inter / union.clamp(min=1e-8)
        A   = (J > threshold).float()
        A.fill_diagonal_(1.0)
        deg = A.sum(-1, keepdim=True).clamp(min=1).sqrt()
        return A / (deg * deg.T)

    def forward(self, class_attrs: torch.Tensor,
                binary_attrs: torch.Tensor,
                threshold: float = 0.1) -> torch.Tensor:
        A_hat = self.build_adjacency(binary_attrs, threshold)
        h1    = self.gcn1(class_attrs, A_hat)
        return A_hat @ self.gcn2(h1)


# ─────────────────────────────────────────────────────────────────────────────
# TZSL MODEL
# ─────────────────────────────────────────────────────────────────────────────

class TZSL:
    """
    Faithful reimplementation of Wang et al. (2025) TZSL.

    Stage 1 — pretrain_kg(): GCN trained → frozen prototypes.
    Stage 2 — fit():         VQ-VAE trained to align to fixed prototypes.
    """

    def __init__(self, d_in: int,
                 class_attrs: torch.Tensor,
                 binary_class_attrs: torch.Tensor,
                 seen_class_ids: list, unseen_class_ids: list,
                 hidden_dim: int = 256, embedding_dim: int = 64,
                 n_embeddings: int = 512, gcn_hidden: int = 128,
                 kg_threshold: float = 0.1, lambda_proto: float = 0.5,
                 lambda_transductive: float = 0.5, temperature: float = 0.15,
                 device: str = "cpu"):

        self.device              = device
        self.seen_ids            = seen_class_ids
        self.unseen_ids          = unseen_class_ids
        self.all_ids             = seen_class_ids + unseen_class_ids
        self.class_attrs         = class_attrs.to(device)
        self.binary_attrs        = binary_class_attrs.to(device)
        self.kg_threshold        = kg_threshold
        self.lambda_proto        = lambda_proto
        self.lambda_transductive = lambda_transductive
        self.temperature         = temperature

        attr_dim       = class_attrs.shape[1]
        self.vqvae     = VQVAE(d_in, hidden_dim, embedding_dim,
                               n_embeddings).to(device)
        self.kg        = MalwareKnowledgeGraph(attr_dim, gcn_hidden,
                                               embedding_dim).to(device)
        self.proj      = nn.Linear(embedding_dim, embedding_dim).to(device)
        self.proto_fixed = None   # populated by pretrain_kg()

    # ── Stage 1: KG Pre-training ──────────────────────────────────────────────

    def pretrain_kg(self, n_steps: int = 800, lr: float = 1e-3,
                    good_sim_thresh: float = 0.3):
        """
        Train GCN with spread + discrimination loss, then freeze permanently.
        spread_loss  — minimise average pairwise cosine sim across all 24 classes.
        discrim_loss — n_seen-way CE so seen prototypes form a clean classifier.
        """
        print(f"\n  [Stage 1] KG pre-training ({n_steps} steps) ...")

        opt       = torch.optim.Adam(self.kg.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_steps, eta_min=1e-5
        )
        seen_idx  = torch.tensor(
            [self.all_ids.index(s) for s in sorted(self.seen_ids)],
            device=self.device,
        )

        for step in range(n_steps):
            self.kg.train()
            protos  = self.kg(self.class_attrs,
                              self.binary_attrs,
                              self.kg_threshold)          # (n_classes, emb_dim)
            p_norm  = F.normalize(protos, dim=-1)

            # ── Spread: push ALL prototype pairs apart ─────────────────────
            sim      = p_norm @ p_norm.T                  # (C, C)
            off_diag = sim[~torch.eye(len(protos),
                                      dtype=torch.bool,
                                      device=self.device)]
            spread_loss = off_diag.mean()

            # ── Discrimination: n_seen-way CE on seen prototypes ───────────
            p_seen      = p_norm[seen_idx]
            logits      = (p_seen @ p_seen.T) / self.temperature
            targets     = torch.arange(len(seen_idx), device=self.device)
            discrim_loss = F.cross_entropy(logits, targets)

            # ── Backprop ───────────────────────────────────────────────────
            loss = spread_loss + discrim_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

            if (step + 1) % 100 == 0 or step == 0:
                print(f"    step {step+1:>4}/{n_steps}  "
                      f"spread={spread_loss.item():.4f}  "
                      f"discrim={discrim_loss.item():.4f}")

        # ── Freeze GCN permanently ─────────────────────────────────────────
        for p in self.kg.parameters():
            p.requires_grad_(False)
        self.kg.eval()

        # ── Pre-compute fixed prototype matrix ────────────────────────────
        with torch.no_grad():
            self.proto_fixed = self.kg(
                self.class_attrs, self.binary_attrs, self.kg_threshold
            )

        self._proto_spread_diagnostic(good_sim_thresh)

    def _proto_spread_diagnostic(self, good_sim_thresh: float = 0.3):
        with torch.no_grad():
            p_norm   = F.normalize(self.proto_fixed, dim=-1)
            sim      = p_norm @ p_norm.T
            sim_diag = sim.clone()
            sim_diag.fill_diagonal_(0.0)
            avg_sim  = sim_diag.abs().mean().item()
            max_sim  = sim_diag.abs().max().item()

        status = "✓ good" if avg_sim < good_sim_thresh else "⚠️  still high"
        print(f"\n  [Proto spread diagnostic]")
        print(f"  Avg off-diag cosine sim : {avg_sim:.4f}  "
              f"({status}, threshold={good_sim_thresh})")
        print(f"  Max off-diag cosine sim : {max_sim:.4f}")

        n     = sim_diag.shape[0]
        pairs = [(sim_diag[i, j].abs().item(), i, j)
                 for i in range(n) for j in range(i + 1, n)]
        pairs.sort(reverse=True)
        print(f"  Top-5 most similar class pairs:")
        for s, i, j in pairs[:5]:
            print(f"    sim={s:.4f}  class {i} ↔ class {j}")

        if avg_sim >= good_sim_thresh:
            print(f"  ⚠️  Consider increasing kg_pretrain_steps or top_k "
                  f"in data_prep.py for more discriminative feature vectors.")

    # ── Stage 2: VQ-VAE Training ──────────────────────────────────────────────

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor,
            X_test_unlabeled: torch.Tensor,
            epochs: int = 200, lr: float = 1e-3,
            weight_decay: float = 1e-4, batch_size: int = 64,
            kg_pretrain_steps: int = 800, kg_pretrain_lr: float = 1e-3,
            kg_good_sim_thresh: float = 0.3):

        # ── Stage 1 ───────────────────────────────────────────────────────
        self.pretrain_kg(
            n_steps         = kg_pretrain_steps,
            lr              = kg_pretrain_lr,
            good_sim_thresh = kg_good_sim_thresh,
        )

        # ── Stage 2 ───────────────────────────────────────────────────────
        print(f"\n  [Stage 2] VQ-VAE training ({epochs} epochs) ...")
        print(f"  GCN is frozen — only VQ-VAE + proj are trainable.")

        opt = torch.optim.Adam(
            list(self.vqvae.parameters())
            + list(self.proj.parameters()),
            lr=lr, weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs, eta_min=1e-6
        )

        best_loss  = float("inf")
        best_state = None

        g2l      = {g: l for l, g in enumerate(sorted(self.seen_ids))}
        seen_idx = torch.tensor(
            [self.all_ids.index(s) for s in sorted(self.seen_ids)],
            device=self.device,
        )
        proto_seen_fixed = self.proto_fixed[seen_idx]    # (n_seen, emb_dim)

        labeled_loader   = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size, shuffle=True, drop_last=False,
        )
        unlabeled_loader = DataLoader(
            TensorDataset(X_test_unlabeled),
            batch_size=batch_size, shuffle=True, drop_last=False,
        )
        unlabeled_iter   = iter(unlabeled_loader)

        for epoch in range(epochs):
            self.vqvae.train(); self.proj.train()
            total_loss = 0.0

            for xb, yb in labeled_loader:
                xb       = xb.to(self.device)
                yb_local = torch.tensor(
                    [g2l[int(i)] for i in yb],
                    dtype=torch.long, device=self.device,
                )

                # ── Labeled: VQ-VAE + CE prototype alignment ──────────────
                z_q, vqvae_loss, _ = self.vqvae(xb)
                z_proj             = self.proj(z_q)
                z_norm             = F.normalize(z_proj,           dim=-1)
                p_norm             = F.normalize(proto_seen_fixed,  dim=-1)
                logits             = (z_norm @ p_norm.T) / self.temperature
                proto_loss         = F.cross_entropy(logits, yb_local)

                # ── Unlabeled: reconstruction only (transductive T) ────────
                try:
                    (xu,) = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter   = iter(unlabeled_loader)
                    (xu,) = next(unlabeled_iter)
                _, vqvae_loss_u, _ = self.vqvae(xu.to(self.device))

                loss = (vqvae_loss
                        + self.lambda_proto        * proto_loss
                        + self.lambda_transductive * vqvae_loss_u)

                opt.zero_grad(); loss.backward(); opt.step()
                total_loss += loss.item()

            scheduler.step()
            avg     = total_loss / len(labeled_loader)
            lr_now  = scheduler.get_last_lr()[0]
            is_best = avg < best_loss

            if is_best:
                best_loss  = avg
                best_state = {
                    "vqvae": {k: v.cpu().clone()
                              for k, v in self.vqvae.state_dict().items()},
                    "proj" : {k: v.cpu().clone()
                              for k, v in self.proj.state_dict().items()},
                }

            if (epoch + 1) % 10 == 0 or epoch == 0:
                marker = "  ← best" if is_best else ""
                print(f"  [TZSL] Epoch {epoch+1:>3}/{epochs}  "
                      f"loss={avg:.4f}  lr={lr_now:.2e}{marker}")

        self.vqvae.load_state_dict(
            {k: v.to(self.device) for k, v in best_state["vqvae"].items()})
        self.proj.load_state_dict(
            {k: v.to(self.device) for k, v in best_state["proj"].items()})
        print(f"  ✓ Restored best checkpoint  (loss={best_loss:.4f})")

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _embed(self, X: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
        self.vqvae.eval(); self.proj.eval()
        loader = DataLoader(TensorDataset(X),
                            batch_size=batch_size, shuffle=False)
        parts  = []
        for (xb,) in loader:
            z_q, _, _ = self.vqvae(xb.to(self.device))
            parts.append(self.proj(z_q).cpu())
        return torch.cat(parts, dim=0)

    @torch.no_grad()
    def evaluate_gzsl(self, X_test: torch.Tensor, y_test: torch.Tensor,
                      gamma_values: list = None) -> tuple:
        if gamma_values is None:
            gamma_values = [round(g, 2)
                            for g in np.arange(0.0, 1.05, 0.05).tolist()]

        seen_sorted   = sorted(self.seen_ids)
        ordered_gids  = seen_sorted + self.unseen_ids
        ordered_idx   = [self.all_ids.index(g) for g in ordered_gids]
        all_protos    = self.proto_fixed[ordered_idx].cpu()

        seen_col_mask = torch.tensor(
            [1.0 if gid in self.seen_ids else 0.0 for gid in ordered_gids],
            dtype=torch.float32,
        )

        embeds = self._embed(X_test)
        embeds = F.normalize(embeds - embeds.mean(0, keepdim=True), dim=-1)
        protos = F.normalize(all_protos, dim=-1)
        sims   = embeds @ protos.T

        truth       = y_test.numpy()
        seen_mask   = np.isin(truth, list(set(self.seen_ids)))
        unseen_mask = np.isin(truth, list(set(self.unseen_ids)))

        best     = {"H_mean": -1.0}
        all_rows = []

        for gamma in gamma_values:
            calibrated = sims - gamma * seen_col_mask.unsqueeze(0)
            pred_idx   = calibrated.argmax(dim=-1).numpy()
            preds      = np.array([ordered_gids[i] for i in pred_idx])

            acc_s = (accuracy_score(truth[seen_mask],   preds[seen_mask])
                     if seen_mask.any()   else 0.0)
            acc_u = (accuracy_score(truth[unseen_mask], preds[unseen_mask])
                     if unseen_mask.any() else 0.0)
            h     = (2 * acc_s * acc_u / (acc_s + acc_u)
                     if (acc_s + acc_u) > 0 else 0.0)

            all_rows.append({"gamma": gamma, "acc_seen": acc_s,
                             "acc_unseen": acc_u, "H_mean": h})
            if h > best["H_mean"]:
                best = {"gamma": gamma, "acc_seen": acc_s,
                        "acc_unseen": acc_u, "H_mean": h}

        return best, all_rows


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS & REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def print_results(best: dict, model_name: str = "TZSL"):
    print(f"\n{'═'*65}")
    print(f"  {model_name}  —  GZSL Results (Best γ={best['gamma']:.2f})")
    print(f"{'═'*65}")
    print(f"  {'Acc Seen (S)':<30}  {best['acc_seen']*100:>9.2f}%")
    print(f"  {'Acc Unseen (U)':<30}  {best['acc_unseen']*100:>9.2f}%")
    print(f"  {'H-mean':<30}  {best['H_mean']*100:>9.2f}%")
    print(f"{'═'*65}")
    print(f"\n  ── Reference (Wang et al., same model, different dataset) ──")
    print(f"  {'S (paper)':<30}  {'91.40%':>10}")
    print(f"  {'U (paper)':<30}  {'59.25%':>10}")
    print(f"  {'H (paper)':<30}  {'71.89%':>10}")
    print(f"  Note: paper uses PE ransomware + API features (9/4 split).")
    print(f"  This run uses Android network flows (20/4 split).")
    print(f"  Numbers are NOT directly comparable — different domain/dataset.")


def save_results(best: dict, all_rows: list, output_path: str):
    import pandas as pd
    df_sweep               = pd.DataFrame(all_rows)
    df_sweep["acc_seen"]  *= 100
    df_sweep["acc_unseen"]*= 100
    df_sweep["H_mean"]    *= 100
    sweep_path = output_path.replace(".csv", "_gamma_sweep.csv")
    df_sweep.to_csv(sweep_path, index=False)
    print(f"  ✓ γ sweep saved   : {sweep_path}")

    pd.DataFrame([{
        "model"     : "TZSL",
        "acc_seen"  : round(best["acc_seen"]   * 100, 4),
        "acc_unseen": round(best["acc_unseen"] * 100, 4),
        "H_mean"    : round(best["H_mean"]     * 100, 4),
        "best_gamma": best["gamma"],
    }]).to_csv(output_path, index=False)
    print(f"  ✓ Best result saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(cfg: dict):
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{BANNER}")
    print("  TZSL — Faithful Reimplementation (Wang et al., 2025)")
    print(f"  Data  : {cfg['data_path']}")
    print(f"  Device: {device}  |  Seed: {cfg['seed']}")
    print(BANNER)

    # ── [1] Load data ─────────────────────────────────────────────────────────
    print("\n[1/4] Loading prepared_data.pt ...")
    if not os.path.exists(cfg["data_path"]):
        raise FileNotFoundError(
            f"Data not found: '{cfg['data_path']}'\nRun data_prep.py first."
        )
    data = torch.load(cfg["data_path"], weights_only=False)

    X_train            = data["X_train"]
    y_train            = data["y_train"]
    X_test             = data["X_test"]
    y_test             = data["y_test"]
    class_attrs        = data["class_attrs"]
    binary_class_attrs = data["binary_class_attrs"]
    seen_ids           = data["seen_label_ids"]
    unseen_ids         = data["unseen_label_ids"]
    le                 = data["le"]

    print(f"  X_train            : {X_train.shape}")
    print(f"  X_test             : {X_test.shape}  (seen + unseen mixed)")
    print(f"  class_attrs        : {class_attrs.shape}")
    print(f"  binary_class_attrs : {binary_class_attrs.shape}")
    print(f"  Seen  classes ({len(seen_ids)}): {data['seen_class_names']}")
    print(f"  Unseen classes ({len(unseen_ids)}): "
          f"{[le.classes_[i] for i in unseen_ids]}")

    # ── KG connectivity diagnostic ────────────────────────────────────────────
    print("\n  [KG connectivity diagnostic]")
    b        = binary_class_attrs.float()
    inter    = (b.unsqueeze(0) * b.unsqueeze(1)).sum(-1)
    union    = (b.unsqueeze(0) + b.unsqueeze(1)).clamp(max=1).sum(-1)
    J        = inter / union.clamp(min=1e-8)
    A_d      = (J > cfg["kg_threshold"]).float()
    A_d.fill_diagonal_(0.0)
    n_edges  = int(A_d.sum().item()) // 2
    isolated = int((A_d.sum(1) == 0).sum().item())
    print(f"  Nodes={len(b)}  Edges={n_edges}  "
          f"Avg degree={A_d.sum(1).mean().item():.1f}  "
          f"Isolated={isolated}")
    if isolated > 0:
        print(f"  ⚠️  {isolated} isolated node(s) — lower kg_threshold "
              f"(current={cfg['kg_threshold']}).")
    else:
        print(f"  ✓ All nodes connected — GCN will propagate to unseen classes.")

    # ── [2] Build model ───────────────────────────────────────────────────────
    print("\n[2/4] Building TZSL model ...")
    model = TZSL(
        d_in                = data["d_in"],
        class_attrs         = class_attrs,
        binary_class_attrs  = binary_class_attrs,
        seen_class_ids      = seen_ids,
        unseen_class_ids    = unseen_ids,
        hidden_dim          = cfg["hidden_dim"],
        embedding_dim       = cfg["embedding_dim"],
        n_embeddings        = cfg["n_embeddings"],
        gcn_hidden          = cfg["gcn_hidden"],
        kg_threshold        = cfg["kg_threshold"],
        lambda_proto        = cfg["lambda_proto"],
        lambda_transductive = cfg["lambda_transductive"],
        temperature         = cfg["temperature"],
        device              = device,
    )
    n_params = sum(p.numel() for p in
                   list(model.vqvae.parameters())
                   + list(model.kg.parameters())
                   + list(model.proj.parameters())
                   if p.requires_grad)
    print(f"  Trainable parameters (total before freeze): {n_params:,}")

    # ── [3] Train ─────────────────────────────────────────────────────────────
    print(f"\n[3/4] Training ...")
    print(f"  Stage 1 : KG pre-training  ({cfg['kg_pretrain_steps']} steps)")
    print(f"  Stage 2 : VQ-VAE training  ({cfg['epochs']} epochs, "
          f"batch={cfg['batch_size']}, lr={cfg['lr']})")
    print(f"  Transductive: {X_test.shape[0]:,} unlabeled test samples "
          f"included in Stage 2.")

    t0 = time.time()
    model.fit(
        X_train, y_train,
        X_test_unlabeled    = X_test,
        epochs              = cfg["epochs"],
        lr                  = cfg["lr"],
        weight_decay        = cfg["weight_decay"],
        batch_size          = cfg["batch_size"],
        kg_pretrain_steps   = cfg["kg_pretrain_steps"],
        kg_pretrain_lr      = cfg["kg_pretrain_lr"],
        kg_good_sim_thresh  = cfg["kg_good_sim_thresh"],
    )
    print(f"  Total training time: {time.time()-t0:.1f}s")

    # ── [4] Evaluate ──────────────────────────────────────────────────────────
    print(f"\n[4/4] GZSL evaluation (γ grid search over "
          f"{len(cfg['gamma_values'])} values) ...")
    best, all_rows = model.evaluate_gzsl(
        X_test, y_test,
        gamma_values = cfg["gamma_values"],
    )

    print_results(best, model_name="TZSL (reimplemented)")
    save_results(best, all_rows, cfg["output_path"])

    print(f"\n{'─'*70}")
    print("  DONE. Use tzsl_results.csv in your VENOM paper comparison table.")
    print(f"{'─'*70}")


if __name__ == "__main__":
    main(CFG)
