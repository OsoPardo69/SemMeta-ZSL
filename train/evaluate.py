"""
Stage 2 — GZSL Evaluation
===========================
Loads a trained StudentMLP checkpoint and evaluates it using Z-score
confidence gating across seen and unseen classes.

Reports:
  - S (seen accuracy), U (unseen accuracy), H (harmonic mean)
  - Per-class breakdown for unseen classes
  - Optional Z-threshold sweep

Usage:
    python stage2/evaluate.py --config stage2/configs/andmal.yaml --checkpoint outputs/andmal/student_mlp_seed42.pt
    python stage2/evaluate.py --config stage2/configs/andmal.yaml --checkpoint outputs/andmal/student_mlp_seed42.pt --sweep_z
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from smeta_zsl import StudentMLP, MalwareTabularDataset
from smeta_zsl import load_embeddings, build_prototypes, prepare_tabular


def zscore_gated_predict(model, loader, all_protos, all_proto_labels, seen_families, unseen_families, device, z_thresh):
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
    return preds, targets


def accuracy(preds, targets):
    return sum(p == t for p, t in zip(preds, targets)) / len(targets) if targets else 0.0


def harmonic_mean(s, u):
    return (2 * s * u) / (s + u) if (s + u) > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="SMETA-ZSL Stage 2 Evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint from train.py")
    parser.add_argument("--sweep_z", action="store_true", help="Sweep Z-score thresholds and print table")
    parser.add_argument("--z_threshold", type=float, default=None, help="Override Z-threshold from config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = cfg["dataset_name"]

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_cfg = ckpt.get("config", cfg)
    seed = ckpt.get("seed", cfg.get("seed", 42))

    # Reconstruct data
    emb_dir = Path(cfg["embeddings_dir"])
    tab_csv = Path(cfg["tabular_csv"])

    train_emb, test_seen_emb, unseen_emb, train_meta, test_seen_meta, unseen_meta = load_embeddings(emb_dir)
    all_emb = np.concatenate([train_emb, test_seen_emb, unseen_emb], axis=0)
    all_meta = pd.concat([train_meta, test_seen_meta, unseen_meta], ignore_index=True)

    seen_families = pd.concat([train_meta, test_seen_meta])["label"].unique().tolist()
    unseen_families = unseen_meta["label"].unique().tolist()

    text_prototypes = build_prototypes(all_emb, all_meta)

    unseen_cal_size = cfg.get("unseen_cal_size", 0.4)
    (X_train, y_train,
     X_val, y_val,
     X_unseen_cal, y_unseen_cal,
     X_unseen_test, y_unseen_test,
     scaler, input_dim) = prepare_tabular(
        tab_csv, seen_families, unseen_families, text_prototypes,
        seed=seed, unseen_cal_size=unseen_cal_size,
    )

    val_loader         = DataLoader(MalwareTabularDataset(X_val,         y_val,         text_prototypes), batch_size=256)
    unseen_cal_loader  = DataLoader(MalwareTabularDataset(X_unseen_cal,  y_unseen_cal,  text_prototypes), batch_size=256)
    unseen_test_loader = DataLoader(MalwareTabularDataset(X_unseen_test, y_unseen_test, text_prototypes), batch_size=256)

    all_proto_labels = list(text_prototypes.keys())
    all_protos = torch.stack([
        torch.tensor(text_prototypes[l], dtype=torch.float32) for l in all_proto_labels
    ]).to(device)
    all_protos = F.normalize(all_protos, p=2, dim=1)

    # Load model
    model = StudentMLP(
        input_dim=input_dim,
        output_dim=cfg["output_dim"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _Z_CANDIDATES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    if args.z_threshold is not None:
        # Manual override: skip calibration
        z_thresh = args.z_threshold
        print(f"Z-threshold: {z_thresh} (manual override)")
    else:
        # Calibrate on the held-out unseen calibration split
        best_h, z_thresh = -1.0, _Z_CANDIDATES[0]
        for z in _Z_CANDIDATES:
            s_p, s_t = zscore_gated_predict(model, val_loader, all_protos, all_proto_labels,
                                            seen_families, unseen_families, device, z)
            u_p, u_t = zscore_gated_predict(model, unseen_cal_loader, all_protos, all_proto_labels,
                                            seen_families, unseen_families, device, z)
            h_cal = harmonic_mean(accuracy(s_p, s_t), accuracy(u_p, u_t))
            if h_cal > best_h:
                best_h, z_thresh = h_cal, z
        print(f"Z-threshold: {z_thresh} (calibrated on unseen-cal, H={best_h*100:.2f}%)")

    if args.sweep_z:
        print(f"\n{'Z-Threshold':<14} | {'Seen Acc':<10} | {'Unseen-test Acc':<17} | {'H-Mean':<10} | Set")
        print("-" * 62)
        for z in _Z_CANDIDATES:
            s_p, s_t = zscore_gated_predict(model, val_loader, all_protos, all_proto_labels,
                                            seen_families, unseen_families, device, z)
            u_p, u_t = zscore_gated_predict(model, unseen_test_loader, all_protos, all_proto_labels,
                                            seen_families, unseen_families, device, z)
            s = accuracy(s_p, s_t)
            u = accuracy(u_p, u_t)
            h = harmonic_mean(s, u)
            marker = " <--" if abs(z - z_thresh) < 1e-4 else ""
            print(f"{z:<14.1f} | {s*100:<10.2f} | {u*100:<17.2f} | {h*100:<10.2f}{marker}")
        print()

    # Final evaluation on the held-out unseen test split
    s_preds, s_targets = zscore_gated_predict(model, val_loader, all_protos, all_proto_labels,
                                               seen_families, unseen_families, device, z_thresh)
    u_preds, u_targets = zscore_gated_predict(model, unseen_test_loader, all_protos, all_proto_labels,
                                               seen_families, unseen_families, device, z_thresh)

    s_acc = accuracy(s_preds, s_targets)
    u_acc = accuracy(u_preds, u_targets)
    h = harmonic_mean(s_acc, u_acc)

    print(f"=== {dataset_name} (Z-threshold = {z_thresh}, calibrated on unseen-cal) ===")
    print(f"  Seen Acc  (S): {s_acc*100:.2f}%")
    print(f"  Unseen Acc(U): {u_acc*100:.2f}%")
    print(f"  H-Mean    (H): {h*100:.2f}%")
    print()
    print("Unseen-test class breakdown:")
    print(classification_report(u_targets, u_preds, labels=unseen_families, zero_division=0))


if __name__ == "__main__":
    main()
