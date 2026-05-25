#!/usr/bin/env python3
"""
p2t_baseline.py — P2T: Tabular Transfer Learning via Prompting LLMs
═══════════════════════════════════════════════════════════════════

GZSL baseline for SMETA comparison table. Supports both local HuggingFace
models via vLLM (e.g., Llama-3.1-8B) and asynchronous OpenAI API calls.

CLI Examples
────────────
Local Llama (vLLM Multi-GPU):
python p2t_baseline.py --use_local --model meta-llama/Meta-Llama-3.1-8B-Instruct --top_k_feats 20

Async OpenAI:
python p2t_baseline.py --model gpt-4o-mini --concurrency 20 --top_k_feats 20
"""
import os
# NOTE: VLLM_USE_V1 was removed from vLLM ≥0.7. V1 is now always active.
#       Do NOT set it — it causes an "Unknown env variable" warning and is ignored.
# Prevent NCCL timeout crashes on large multi-GPU loads
os.environ["NCCL_BLOCKING_WAIT"] = "1"

import argparse
import json
import random
import subprocess
import sys
import asyncio
from pathlib import Path

import numpy as np
import torch
from tqdm.asyncio import tqdm_asyncio
from sklearn.feature_selection import mutual_info_classif

# ── data_prep must sit in the same directory ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from data_prep import prepare

# ─────────────────────────────────────────────────────────────────────────────
# DATASETS & SEEDS CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATASETS = {
    "AndMAL":   "/home/iamontoyasa/02_VENOM/03_Baseline/01_run1_prepared_data_zero_shot.pt",
    "BODMAS":   "/home/iamontoyasa/02_VENOM/03_Baseline/02_run1_prepared_data_zero_shot.pt",
    "AVASTCTU": "/home/iamontoyasa/02_VENOM/03_Baseline/03_run1_prepared_data_zero_shot.pt",
    "APIGRAPH": "/home/iamontoyasa/02_VENOM/03_Baseline/04_run1_prepared_data_zero_shot.pt",
    "Goodread": "/home/iamontoyasa/02_VENOM/03_Baseline/06_run1_prepared_data_zero_shot.pt",
    "Petfinder": "/home/iamontoyasa/02_VENOM/03_Baseline/07_run1_prepared_data_zero_shot.pt",
    "Fakeddit": "/home/iamontoyasa/02_VENOM/03_Baseline/08_run1_prepared_data_zero_shot.pt",
}

SEEDS = [42, 123, 456, 789, 2025]

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert malware analyst performing family classification. "
    "You will be given the behavioral feature profile of a malware sample "
    "and a list of candidate malware families with their descriptions "
    "and (where available) example profiles from training data. "
    "Output ONLY the exact family name that best matches the sample — "
    "no explanation, no punctuation, no extra text."
)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SERIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def select_top_features(
    X_train: np.ndarray,
    y_train_str: np.ndarray,
    feature_names: list,
    top_k: int,
) -> list:
    """
    Selects top_k features using Mutual Information.
    This avoids scale bias that occurs when using raw variance.
    """
    mi_scores = mutual_info_classif(X_train, y_train_str, random_state=42)
    top_idx = np.argsort(mi_scores)[::-1][:top_k].tolist()

    selected = [feature_names[i] for i in top_idx]
    print(f"  Top-{top_k} features (MI): {selected[:5]} ... {selected[-3:]}")
    return top_idx


def serialise_row(x: np.ndarray, feature_names: list, top_idx: list, precision: int = 3) -> str:
    parts = []
    for i in top_idx:
        val = float(x[i])
        parts.append(
            f"{feature_names[i]}={int(val)}"
            if val == int(val)
            else f"{feature_names[i]}={val:.{precision}f}"
        )
    return ", ".join(parts)

# ─────────────────────────────────────────────────────────────────────────────
# IN-CONTEXT EXAMPLE POOL
# ─────────────────────────────────────────────────────────────────────────────

def build_icl_pool(
    X_train: np.ndarray, y_train_str: np.ndarray, feature_names: list,
    top_idx: list, seen_families: list, k_shot: int, seed: int = 42
) -> dict:
    rng = np.random.default_rng(seed)
    pool = {}
    for fam in seen_families:
        indices = np.where(y_train_str == fam)[0]
        if len(indices) == 0:
            pool[fam] = []
            continue
        chosen = rng.choice(indices, size=min(k_shot, len(indices)), replace=False)
        pool[fam] = [serialise_row(X_train[i], feature_names, top_idx) for i in chosen]

    total = sum(len(v) for v in pool.values())
    print(f"  {len(seen_families)} seen classes × ≤{k_shot} shots = {total} ICL examples")
    return pool


def build_icl_pool_transductive(
    X_train: np.ndarray, y_train_str: np.ndarray, X_test: np.ndarray,
    feature_names: list, top_idx: list, seen_families: list,
    unseen_families: list, k_shot: int, seed: int = 42
) -> dict:
    pool = build_icl_pool(X_train, y_train_str, feature_names, top_idx, seen_families, k_shot, seed)
    test_centroid = X_test.mean(axis=0)
    for fam in unseen_families:
        dists = np.linalg.norm(X_test[:, top_idx] - test_centroid[top_idx], axis=1)
        chosen = np.argsort(dists)[:k_shot]
        pool[fam] = [serialise_row(X_test[i], feature_names, top_idx) for i in chosen]
    return pool

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(
    test_row_str: str, all_families: list, class_descriptions: dict,
    icl_pool: dict, seen_families: list
) -> str:
    lines = ["CANDIDATE CLASSES", "─" * 50]
    for fam in all_families:
        is_seen = fam in seen_families
        tag = "(seen)" if is_seen else "(unseen — no training examples)"
        lines.append(f"\nFamily: {fam} {tag}")
        desc = class_descriptions.get(fam, f"A malware family named {fam} with unknown behavioural profile.")
        lines.append(f"Description: {desc}")
        if is_seen and icl_pool.get(fam):
            lines.append("Examples:")
            for ex in icl_pool[fam]:
                lines.append(f"  {ex} → {fam}")

    lines += [
        "\n" + "─" * 50,
        "SAMPLE TO CLASSIFY",
        test_row_str,
        "\nReply with ONLY the exact family name from the list above.",
    ]
    return "\n".join(lines)


def match_family(raw_output: str, all_families: list) -> str:
    """Normalises LLM output to a valid family name."""
    raw = raw_output.strip()
    if raw in all_families: return raw
    rl = raw.lower()
    for fam in all_families:
        if fam.lower() == rl: return fam
    for fam in all_families:
        if fam.lower() in rl or rl in fam.lower(): return fam
    return raw

# ─────────────────────────────────────────────────────────────────────────────
# ASYNC OPENAI BACKEND
# ─────────────────────────────────────────────────────────────────────────────

async def query_llm_async(
    client, model: str, prompt: str, all_families: list, sem: asyncio.Semaphore, max_retries: int = 5
) -> str:
    """Async wrapper for OpenAI to allow concurrent requests."""
    async with sem:
        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=32,
                )
                return match_family(resp.choices[0].message.content, all_families)

            except Exception as e:
                err = str(e)
                if "rate_limit" in err.lower() or "429" in err:
                    delay = 1.0 * (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(delay)
                elif attempt < max_retries - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))
                else:
                    print(f"\n  [ERROR] Async LLM failed: {e}")
                    return all_families[0]


async def run_openai_batch(todo_indices, prompts_dict, model, all_families, api_key, max_concurrency, cache_path,
                           predictions, batch_size):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(max_concurrency)

    async def process_one(idx):
        pred = await query_llm_async(client, model, prompts_dict[idx], all_families, sem)
        return idx, pred

    tasks = [process_one(idx) for idx in todo_indices]
    n_saved = 0

    for f in tqdm_asyncio.as_completed(tasks, desc="OpenAI Async Inference", total=len(tasks), dynamic_ncols=True):
        idx, pred = await f
        predictions[idx] = pred
        n_saved += 1

        if n_saved % batch_size == 0:
            with open(cache_path, "w") as out_f:
                json.dump(predictions, out_f)

# ─────────────────────────────────────────────────────────────────────────────
# STRATIFIED SUBSAMPLE & METRICS
# ─────────────────────────────────────────────────────────────────────────────

def stratified_subsample(X_test: np.ndarray, y_test_str: np.ndarray, max_samples: int, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    n = len(y_test_str)
    if n <= max_samples: return X_test, y_test_str, np.arange(n)

    classes, counts = np.unique(y_test_str, return_counts=True)
    selected = []
    for cls, cnt in zip(classes, counts):
        cls_idx = np.where(y_test_str == cls)[0]
        quota = max(1, int(round(max_samples * cnt / n)))
        chosen = rng.choice(cls_idx, size=min(quota, len(cls_idx)), replace=False)
        selected.extend(chosen.tolist())

    selected = np.array(selected)
    rng.shuffle(selected)
    selected = selected[:max_samples]
    print(f"  Subsampled: {n} → {len(selected)} samples (stratified)")
    return X_test[selected], y_test_str[selected], selected


def gzsl_metrics(y_true: np.ndarray, y_pred: np.ndarray, seen_ids: list, unseen_ids: list, le) -> dict:
    per_class = {}
    for cls_id in np.unique(y_true):
        mask = y_true == cls_id
        per_class[int(cls_id)] = float((y_pred[mask] == cls_id).mean()) * 100.0

    seen_accs   = [per_class[c] for c in seen_ids   if c in per_class]
    unseen_accs = [per_class[c] for c in unseen_ids if c in per_class]

    S = float(np.mean(seen_accs))   if seen_accs   else 0.0
    U = float(np.mean(unseen_accs)) if unseen_accs else 0.0
    H = (2 * S * U / (S + U))      if (S + U) > 0 else 0.0

    return {"S": round(S, 2), "U": round(U, 2), "H": round(H, 2), "per_class_acc": per_class}

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION CORE
# ─────────────────────────────────────────────────────────────────────────────

def run_p2t(args, data_path: str, seed: int, out_path: str, llm=None, tokenizer=None) -> dict:
    dset_name = Path(data_path).stem

    print("\n" + "═" * 60)
    print("  P2T BASELINE — GZSL")
    print(f"  Dataset : {dset_name}")
    print(f"  Seed    : {seed}")
    print(f"  LLM     : {args.model}  k_shot={args.k_shot}  top_feats={args.top_k_feats}")
    print("═" * 60)

    # ── Load Data ────────────────────────────────────────────────────────────
    print(f"  Loading pre-computed tensors from: {data_path}")
    data = torch.load(data_path, weights_only=False)

    le = data["le"]

    X_train = data["X_train"].numpy()
    y_train_str = data["y_train_str"] if "y_train_str" in data else le.inverse_transform(data["y_train"].numpy())

    X_test_full   = data["X_test"].numpy()
    y_test_full   = data["y_test"].numpy()
    y_test_str_full = data["y_test_str"] if "y_test_str" in data else le.inverse_transform(y_test_full)

    feature_names     = data["feature_names"]
    class_descriptions = data["class_descriptions"]
    seen_families, all_families = data["seen_class_names"], data["all_class_names"]
    seen_label_ids, unseen_label_ids = data["seen_label_ids"], data["unseen_label_ids"]

    if args.max_test_samples and args.max_test_samples < len(X_test_full):
        X_test, y_test_str, sub_idx = stratified_subsample(X_test_full, y_test_str_full, args.max_test_samples, seed)
        y_test = y_test_full[sub_idx]
    else:
        X_test, y_test_str, y_test = X_test_full, y_test_str_full, y_test_full

    print("\n── Feature selection")
    top_idx = select_top_features(X_train, y_train_str, feature_names, top_k=args.top_k_feats)

    print("\n── Building ICL pool")
    if args.transductive:
        icl_pool = build_icl_pool_transductive(
            X_train, y_train_str, X_test, feature_names, top_idx,
            seen_families, args.unseen, args.k_shot, seed
        )
    else:
        icl_pool = build_icl_pool(X_train, y_train_str, feature_names, top_idx, seen_families, args.k_shot, seed)

    prompts_dict = {}
    for i in range(len(X_test)):
        row_str = serialise_row(X_test[i], feature_names, top_idx)
        prompts_dict[i] = build_prompt(row_str, all_families, class_descriptions, icl_pool, seen_families)

    # ── Inference Engine Selection ────────────────────────────────────────────
    cache_path  = Path(out_path).with_suffix(".cache.json")
    predictions = {}

    if cache_path.exists() and not args.no_cache:
        with open(cache_path) as f:
            predictions = {int(k): v for k, v in json.load(f).items()}
        print(f"  Resumed from cache: {len(predictions)}/{len(X_test)} done")

    todo = [i for i in range(len(X_test)) if i not in predictions]
    print(f"\n── Classifying {len(todo)} test samples")

    if not todo:
        print("  All samples already cached!")

    elif args.use_local and llm and tokenizer:
        # ── vLLM Inference Engine ─────────────────────────────────────────────
        from vllm import SamplingParams

        formatted_prompts = []
        for idx in todo:
            chat = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompts_dict[idx]},
            ]
            formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            formatted_prompts.append(formatted)

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=32,
            stop=["\n", "User:", "Assistant:"],
        )

        print(f"  Processing {len(todo)} prompts through vLLM...")
        outputs = llm.generate(formatted_prompts, sampling_params)

        for i, out in enumerate(outputs):
            idx = todo[i]
            raw = out.outputs[0].text
            predictions[idx] = match_family(raw, all_families)

        with open(cache_path, "w") as f:
            json.dump(predictions, f)

    else:
        # OpenAI API Inference
        print(f"  Engine: Async OpenAI (Concurrency: {args.concurrency})")
        api_key = os.environ.get("OPENAI_API_KEY") or args.api_key
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY env variable or pass --api_key.")

        asyncio.run(run_openai_batch(
            todo, prompts_dict, args.model, all_families, api_key,
            args.concurrency, cache_path, predictions, args.batch_size
        ))

        with open(cache_path, "w") as f:
            json.dump(predictions, f)

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_pred_str = np.array([predictions[i] for i in range(len(X_test))])
    y_pred_ids = np.array([
        int(le.transform([p])[0]) if p in le.classes_ else -1
        for p in y_pred_str
    ])

    print("\n── GZSL Evaluation")
    metrics = gzsl_metrics(y_test, y_pred_ids, seen_label_ids, unseen_label_ids, le)

    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │ P2T Results — {dset_name:<26}│")
    print(f"  │  Seen     (S) : {metrics['S']:>7.2f} %               │")
    print(f"  │  Unseen   (U) : {metrics['U']:>7.2f} %               │")
    print(f"  │  H-Mean   (H) : {metrics['H']:>7.2f} %               │")
    print(f"  └─────────────────────────────────────────┘")

    invalid = int((y_pred_ids == -1).sum())
    if invalid:
        print(f"\n  ⚠  {invalid} predictions were unrecognised (counted as wrong).")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "dataset": dset_name, "model": args.model, "k_shot": args.k_shot,
        "top_k_feats": args.top_k_feats, "transductive": args.transductive,
        "seed": seed, "n_test_evaluated": int(len(X_test)),
        "n_invalid_preds": invalid, "metrics": metrics,
        "predictions": {str(i): pred for i, pred in predictions.items()},
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results → {out_path}")

    return results

# ─────────────────────────────────────────────────────────────────────────────
# GPU COUNT — safe, no CUDA pre-init
# ─────────────────────────────────────────────────────────────────────────────

def count_gpus_safe() -> int:
    """Count available GPUs via nvidia-smi to avoid pre-initialising CUDA."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
        return max(1, len(lines))
    except Exception:
        return 1

# ─────────────────────────────────────────────────────────────────────────────
# CLI & GLOBAL ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("Data")
    g.add_argument("--unseen",     nargs="*", default=[],   help="Family names held out for ZSL test")
    g.add_argument("--drop",       nargs="*", default=[],   help="Families to drop")
    g.add_argument("--test_ratio", type=float, default=0.2)

    g = p.add_argument_group("P2T")
    g.add_argument("--model",            default="gpt-4o-mini", help="HF model path or OpenAI model name")
    g.add_argument("--k_shot",           type=int, default=3)
    g.add_argument("--top_k_feats",      type=int, default=20)
    g.add_argument("--transductive",     action="store_true")
    g.add_argument("--max_test_samples", type=int, default=None)

    g = p.add_argument_group("Runtime")
    g.add_argument("--use_local",  action="store_true",  help="Use local HuggingFace model via vLLM")
    g.add_argument("--n_gpus",     type=int, default=None,
                   help="Force tensor-parallel size (overrides auto-detect via nvidia-smi)")
    g.add_argument("--quantize",   choices=["4bit", "8bit"], default=None,
                   help="Quantization level for local models")
    g.add_argument("--concurrency", type=int, default=20,  help="Max concurrent async requests (OpenAI only)")
    g.add_argument("--api_key",    default="")
    g.add_argument("--batch_size", type=int, default=50,   help="Checkpoint every N predictions (OpenAI only)")
    g.add_argument("--no_cache",   action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    llm, tokenizer = None, None

    # ── Boot up vLLM ONCE for all datasets and seeds ──────────────────────────
    if args.use_local:
        print("\n  Engine: vLLM (Native Multi-GPU, PagedAttention)")
        from vllm import LLM
        from transformers import AutoTokenizer

        # FIX: use subprocess to count GPUs — avoids pre-initialising CUDA
        # context in the main process, which breaks vLLM's multiproc workers.
        if args.n_gpus is not None:
            tp_size = args.n_gpus
        else:
            actual_gpus = count_gpus_safe()
            tp_size = 2 if actual_gpus >= 2 else 1

        print(f"  Tensor Parallelism: tp_size={tp_size}")

        vllm_quant = "bitsandbytes" if args.quantize in ["4bit", "8bit"] else None
        _model = args.model.rstrip("/") if os.path.isdir(args.model.rstrip("/")) else args.model
        _is_local = os.path.isdir(_model)

        # Load tokenizer before LLM (pure CPU, no CUDA init)
        tokenizer = AutoTokenizer.from_pretrained(
            _model,
            trust_remote_code=True,
            local_files_only=_is_local,  # ← bypasses validate_repo_id for local dirs
        )

        llm = LLM(
            model=args.model,
            tensor_parallel_size=tp_size,
            quantization=vllm_quant,
            dtype="bfloat16",
            max_model_len=4096,
            trust_remote_code=True,
            enforce_eager=True,
            gpu_memory_utilization=0.85,
            disable_log_stats=True,
        )

    all_results = {}

    for d_name, d_path in DATASETS.items():
        print(f"\n{'=' * 80}")
        print(f"  EVALUATING DATASET: {d_name}")
        print(f"{'=' * 80}")

        dataset_runs      = []
        metrics_per_seed  = {"S": [], "U": [], "H": []}

        for seed in SEEDS:
            out_path = f"results/{d_name}_seed{seed}.json"

            res = run_p2t(args, data_path=d_path, seed=seed, out_path=out_path, llm=llm, tokenizer=tokenizer)

            m = res["metrics"]
            metrics_per_seed["S"].append(m["S"])
            metrics_per_seed["U"].append(m["U"])
            metrics_per_seed["H"].append(m["H"])

            dataset_runs.append({"Seed": seed, "S": m["S"], "U": m["U"], "H": m["H"]})

        s_mean, s_std = np.mean(metrics_per_seed["S"]), np.std(metrics_per_seed["S"])
        u_mean, u_std = np.mean(metrics_per_seed["U"]), np.std(metrics_per_seed["U"])
        h_mean, h_std = np.mean(metrics_per_seed["H"]), np.std(metrics_per_seed["H"])

        all_results[d_name] = {
            "runs":   dataset_runs,
            "S_mean": s_mean, "S_std": s_std,
            "U_mean": u_mean, "U_std": u_std,
            "H_mean": h_mean, "H_std": h_std,
        }

    # ── FINAL OUTPUT PRINTS ────────────────────────────────────────────────────
    print("\n═════════════════════════════════════════════════════════════════════════════")
    print("  P2T RESULTS PER DATASET")
    print("═════════════════════════════════════════════════════════════════════════════")

    for ds_name, stats in all_results.items():
        print(f"\n▶ Dataset: {ds_name}")
        print(f"{'Seed':<10} | {'Seen Acc (acc_s)':<18} | {'Unseen Acc (acc_u)':<18} | {'H-mean (h)':<12}")
        print("-" * 65)

        for row in stats["runs"]:
            print(f"{int(row['Seed']):<10} | {row['S']:<18.4f} | {row['U']:<18.4f} | {row['H']:<12.4f}")

        print("-" * 65)
        print(f"{'Mean':<10}    | {stats['S_mean']:<18.4f} | {stats['U_mean']:<18.4f} | {stats['H_mean']:<12.4f}")
        print(f"{'Std Dev':<10} | {stats['S_std']:<18.4f}  | {stats['U_std']:<18.4f}  | {stats['H_std']:<12.4f}\n")

    print("═════════════════════════════════════════════════════════════════════════════\n")
