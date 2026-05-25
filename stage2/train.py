"""
Stage 2 — Episodic Meta-Alignment Training
============================================
Trains a StudentMLP to project tabular behavioral features into the semantic
prototype space produced by Stage 1, using episodic meta-learning with
knowledge distillation (SMETA-ZSL Stage 2).

Usage:
    python stage2/train.py --config stage2/configs/andmal.yaml
    python stage2/train.py --config stage2/configs/bodmas.yaml --seed 123
"""

import argparse
import copy
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from smeta_zsl import StudentMLP, kd_gzsl_loss, MalwareTabularDataset
from smeta_zsl import load_embeddings, build_prototypes, prepare_tabular


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Episode Sampling ──────────────────────────────────────────────────────────

def sample_episode(
    X: np.ndarray,
    y: list,
    text_prototypes: dict,
    seen_families: list,
    support_size: int,
    query_size: int,
    episode_batch: int,
    noise_std: float,
    device: torch.device,
):
    assert support_size + query_size <= len(seen_families), (
        f"Need {support_size + query_size} classes, only {len(seen_families)} available."
    )

    episode_classes = random.sample(seen_families, support_size + query_size)
    sup_classes = episode_classes[:support_size]
    qry_classes = episode_classes[support_size:]
    all_ep_classes = sup_classes + qry_classes

    ep_protos = torch.stack([
        torch.tensor(text_prototypes[c], dtype=torch.float32) for c in all_ep_classes
    ]).to(device)
    ep_protos = F.normalize(ep_protos, p=2, dim=1)

    def _sample(classes):
        feats, labs = [], []
        for c in classes:
            idxs = [i for i, l in enumerate(y) if l == c]
            k = min(episode_batch, len(idxs))
            chosen = random.sample(idxs, k)
            feats.append(X[chosen])
            labs.extend([c] * k)
        return torch.tensor(np.vstack(feats), dtype=torch.float32).to(device), labs

    sup_feats, sup_labs = _sample(sup_classes)
    qry_feats, qry_labs = _sample(qry_classes)

    if noise_std > 0:
        sup_feats = sup_feats + torch.randn_like(sup_feats) * noise_std
        qry_feats = qry_feats + torch.randn_like(qry_feats) * noise_std

    return sup_feats, sup_labs, qry_feats, qry_labs, ep_protos, all_ep_classes


# ── Training Loop ─────────────────────────────────────────────────────────────

def train(cfg: dict, seed: int):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = cfg["dataset_name"]
    print(f"\n[{dataset_name}] Device: {device} | Seed: {seed}")

    emb_dir = Path(cfg["embeddings_dir"])
    tab_csv = Path(cfg["tabular_csv"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Stage 1 outputs
    train_emb, test_seen_emb, unseen_emb, train_meta, test_seen_meta, unseen_meta = load_embeddings(emb_dir)

    all_emb = np.concatenate([train_emb, test_seen_emb, unseen_emb], axis=0)
    all_meta = pd.concat([train_meta, test_seen_meta, unseen_meta], ignore_index=True)

    seen_families = pd.concat([train_meta, test_seen_meta])["label"].unique().tolist()
    unseen_families = unseen_meta["label"].unique().tolist()

    print(f"[{dataset_name}] Seen classes: {len(seen_families)} | Unseen classes: {len(unseen_families)}")

    text_prototypes = build_prototypes(all_emb, all_meta)

    # Prepare tabular data
    (
        X_train, y_train,
        X_val, y_val,
        X_unseen, y_unseen,
        scaler,
        input_dim,
    ) = prepare_tabular(tab_csv, seen_families, unseen_families, text_prototypes, seed=seed)

    if cfg.get("input_dim") is not None:
        assert input_dim == cfg["input_dim"], (
            f"Config input_dim={cfg['input_dim']} but CSV has {input_dim} features."
        )
    else:
        cfg["input_dim"] = input_dim

    print(f"[{dataset_name}] Tabular features: {input_dim} | Train: {len(X_train)} | Val: {len(X_val)} | Unseen: {len(X_unseen)}")

    # DataLoaders for evaluation
    train_dataset = MalwareTabularDataset(X_train, y_train, text_prototypes)
    val_dataset = MalwareTabularDataset(X_val, y_val, text_prototypes)
    unseen_dataset = MalwareTabularDataset(X_unseen, y_unseen, text_prototypes)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    unseen_loader = DataLoader(unseen_dataset, batch_size=256, shuffle=False)

    # Build all-class prototype tensor for evaluation
    all_proto_labels = list(text_prototypes.keys())
    all_protos = torch.stack([
        torch.tensor(text_prototypes[l], dtype=torch.float32) for l in all_proto_labels
    ]).to(device)
    all_protos = F.normalize(all_protos, p=2, dim=1)

    # Model, optimizer, scheduler
    model = StudentMLP(
        input_dim=input_dim,
        output_dim=cfg["output_dim"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["num_episodes"], eta_min=1e-6)

    num_episodes = cfg["num_episodes"]
    support_size = cfg["support_size"]
    query_size = cfg["query_size"]
    episode_batch = cfg["episode_batch"]
    lambda_qry = cfg["lambda_query"]
    kd_temp = cfg["kd_temperature"]
    alpha = cfg["alpha"]
    grad_clip = cfg.get("grad_clip", 1.0)
    noise_std = cfg.get("noise_std", 0.0)

    best_qry_loss = float("inf")
    best_state = None
    history = {"sup": [], "qry": [], "total": [], "ep": []}

    print(f"\n[{dataset_name}] Starting episodic meta-training ({num_episodes} episodes)...")
    print(f"  Support/episode: {support_size} | Query/episode: {query_size} | Batch/class: {episode_batch}")

    for ep in range(1, num_episodes + 1):
        model.train()
        optimizer.zero_grad()

        sup_feats, sup_labs, qry_feats, qry_labs, ep_protos, ep_labels = sample_episode(
            X_train, y_train, text_prototypes, seen_families,
            support_size, query_size, episode_batch, noise_std, device,
        )

        sup_embs = model(sup_feats)
        loss_sup = kd_gzsl_loss(sup_embs, sup_labs, ep_protos, ep_labels, text_prototypes, kd_temp, alpha)

        qry_embs = model(qry_feats)
        qry_sims = torch.matmul(qry_embs, ep_protos.T)
        qry_idx = torch.tensor([ep_labels.index(l) for l in qry_labs], device=device)
        loss_qry = F.cross_entropy(qry_sims, qry_idx)

        loss = loss_sup + lambda_qry * loss_qry
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        if loss_qry.item() < best_qry_loss:
            best_qry_loss = loss_qry.item()
            best_state = copy.deepcopy(model.state_dict())

        if ep % 200 == 0:
            lr = scheduler.get_last_lr()[0]
            history["sup"].append(loss_sup.item())
            history["qry"].append(loss_qry.item())
            history["total"].append(loss.item())
            history["ep"].append(ep)
            print(
                f"  Ep {ep:>5} | Sup: {loss_sup.item():.4f} | "
                f"Qry: {loss_qry.item():.4f} | Total: {loss.item():.4f} | LR: {lr:.2e}"
            )

    model.load_state_dict(best_state)
    ckpt_path = output_dir / f"student_mlp_seed{seed}.pt"
    torch.save({"model_state": best_state, "config": cfg, "seed": seed}, ckpt_path)
    print(f"\n[{dataset_name}] Best model saved to {ckpt_path}")

    return model, all_protos, all_proto_labels, seen_families, unseen_families, val_loader, unseen_loader


# ── Quick evaluation after training ──────────────────────────────────────────

def zscore_gated_eval(model, loader, all_protos, all_proto_labels, seen_families, unseen_families, device, z_thresh):
    seen_idx = torch.tensor([all_proto_labels.index(l) for l in seen_families], device=device)
    unseen_idx = torch.tensor([all_proto_labels.index(l) for l in unseen_families], device=device)
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for b_feats, b_labels, _ in loader:
            embs = model(b_feats.to(device))
            sims = torch.matmul(embs, all_protos.T)
            for i in range(embs.size(0)):
                seen_sims = sims[i][seen_idx]
                z = (seen_sims.max() - seen_sims.mean()) / (seen_sims.std(unbiased=False) + 1e-8)
                if z < z_thresh:
                    pred = all_proto_labels[unseen_idx[sims[i][unseen_idx].argmax()].item()]
                else:
                    pred = all_proto_labels[seen_idx[seen_sims.argmax()].item()]
                preds.append(pred)
            targets.extend(b_labels)
    correct = sum(p == t for p, t in zip(preds, targets))
    return correct / len(targets) if targets else 0.0


def main():
    parser = argparse.ArgumentParser(description="SMETA-ZSL Stage 2 Training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None, help="Override seed from config")
    parser.add_argument("--runs", type=int, default=1, help="Number of independent runs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seeds = [args.seed] if args.seed is not None else [cfg.get("seed", 42)]
    if args.runs > 1:
        seeds = [42, 123, 456, 789, 2025][: args.runs]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for seed in seeds:
        model, all_protos, all_proto_labels, seen_fam, unseen_fam, val_loader, unseen_loader = train(cfg.copy(), seed)

        z_thresh = cfg.get("z_threshold", 1.5)
        s_acc = zscore_gated_eval(model, val_loader, all_protos, all_proto_labels, seen_fam, unseen_fam, device, z_thresh)
        u_acc = zscore_gated_eval(model, unseen_loader, all_protos, all_proto_labels, seen_fam, unseen_fam, device, z_thresh)
        h = (2 * s_acc * u_acc) / (s_acc + u_acc) if (s_acc + u_acc) > 0 else 0.0
        results.append({"seed": seed, "seen": s_acc, "unseen": u_acc, "h_mean": h})
        print(f"\n[{cfg['dataset_name']}] Seed {seed} — S: {s_acc*100:.2f}  U: {u_acc*100:.2f}  H: {h*100:.2f}")

    if len(results) > 1:
        h_vals = [r["h_mean"] for r in results]
        print(f"\n[{cfg['dataset_name']}] Mean H: {np.mean(h_vals)*100:.2f} ± {np.std(h_vals)*100:.2f}")


if __name__ == "__main__":
    main()
