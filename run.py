#!/usr/bin/env python
"""
SMETA-ZSL — single-command train + evaluate runner.

Trains a StudentMLP on the chosen dataset/model combination, then
evaluates it with the adaptive Z-score confidence gate.

Usage:
    python run.py --andmal                        # default model: llama3b
    python run.py --bodmas --qwen
    python run.py --andmal --llama8b --runs 5
    python run.py --apigraph --seed 123 --runs 3
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from train.train import train, zscore_gated_eval, calibrate_z_threshold

_DATASETS = ["andmal", "bodmas", "apigraph", "avast", "goodreads", "petfinder", "fakeddit"]
_MODELS   = ["llama3b", "llama8b", "gemma", "qwen", "qwen_coder", "mistral"]
_SEEDS    = [42, 123, 456, 789, 2025]


def resolve_embeddings_dir(dataset: str, model: str) -> Path:
    """Return the embeddings directory.

    Tries data/<dataset>/embeddings/<model>/ first; falls back to the flat
    data/<dataset>/embeddings/ for datasets that have only one embedding set.
    """
    specific = _ROOT / "data" / dataset / "embeddings" / model
    if specific.is_dir():
        return specific
    flat = _ROOT / "data" / dataset / "embeddings"
    if flat.is_dir():
        return flat
    raise FileNotFoundError(
        f"No embeddings found for dataset='{dataset}' model='{model}'. "
        f"Expected {specific} or {flat}."
    )


def load_config(dataset: str, model: str) -> dict:
    cfg_path = _ROOT / "train" / "configs" / f"{dataset}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config found at {cfg_path}")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["embeddings_dir"] = str(resolve_embeddings_dir(dataset, model))
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="SMETA-ZSL train + evaluate (Stage 2 — adaptive Z-score gating)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --andmal                    # llama3b (default)
  python run.py --bodmas --qwen
  python run.py --andmal --llama8b --runs 5
  python run.py --apigraph --seed 123
""",
    )

    # Dataset flags (exactly one required)
    ds_group = parser.add_mutually_exclusive_group(required=True)
    for ds in _DATASETS:
        ds_group.add_argument(f"--{ds}", action="store_const", dest="dataset", const=ds,
                              help=f"Run on the {ds.upper()} dataset.")

    # Model flags (at most one; default llama3b)
    model_group = parser.add_mutually_exclusive_group()
    for m in _MODELS:
        model_group.add_argument(f"--{m}", action="store_const", dest="model", const=m,
                                 help=f"Use {m} embeddings.")
    parser.set_defaults(model="llama3b")

    # Run options
    parser.add_argument(
        "--runs", type=int, default=1, metavar="N",
        help="Number of independent runs with different seeds (max 5, default 1).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Fix a single random seed (overrides --runs).",
    )
    parser.add_argument(
        "--z_threshold", type=float, default=None,
        help="Override the Z-score gate threshold from the dataset config.",
    )

    args = parser.parse_args()

    cfg = load_config(args.dataset, args.model)
    if args.z_threshold is not None:
        cfg["z_threshold"] = args.z_threshold

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.seed is not None:
        seeds = [args.seed]
    elif args.runs > 1:
        seeds = _SEEDS[: min(args.runs, len(_SEEDS))]
    else:
        seeds = [cfg.get("seed", 42)]

    print(f"\nSMETA-ZSL | dataset={args.dataset} | model={args.model} | device={device}")
    print(f"Embeddings : {cfg['embeddings_dir']}")
    print(f"Z-threshold: {cfg.get('z_threshold', 1.5)}")
    print(f"Seeds      : {seeds}\n")

    results = []
    for seed in seeds:
        (model_net, all_protos, all_proto_labels,
         seen_fam, unseen_fam,
         val_loader, unseen_cal_loader, unseen_test_loader) = train(copy.deepcopy(cfg), seed)

        if args.z_threshold is not None:
            z = args.z_threshold
        else:
            z = calibrate_z_threshold(
                model_net, val_loader, unseen_cal_loader,
                all_protos, all_proto_labels, seen_fam, unseen_fam, device,
            )
            print(f"[{args.dataset}/{args.model}] Calibrated Z-threshold: {z}")

        s_acc = zscore_gated_eval(
            model_net, val_loader,         all_protos, all_proto_labels, seen_fam, unseen_fam, device, z
        )
        u_acc = zscore_gated_eval(
            model_net, unseen_test_loader, all_protos, all_proto_labels, seen_fam, unseen_fam, device, z
        )
        h = (2 * s_acc * u_acc) / (s_acc + u_acc) if (s_acc + u_acc) > 0 else 0.0
        results.append({"seed": seed, "seen": s_acc, "unseen": u_acc, "h_mean": h})
        print(
            f"[{args.dataset}/{args.model}] seed={seed:>4}  "
            f"S={s_acc*100:6.2f}  U={u_acc*100:6.2f}  H={h*100:6.2f}"
        )

    if len(results) > 1:
        h_vals = [r["h_mean"] for r in results]
        s_vals = [r["seen"]   for r in results]
        u_vals = [r["unseen"] for r in results]
        print(
            f"\n[{args.dataset}/{args.model}] "
            f"S={np.mean(s_vals)*100:.2f}±{np.std(s_vals)*100:.2f}  "
            f"U={np.mean(u_vals)*100:.2f}±{np.std(u_vals)*100:.2f}  "
            f"H={np.mean(h_vals)*100:.2f}±{np.std(h_vals)*100:.2f}"
        )


if __name__ == "__main__":
    main()
