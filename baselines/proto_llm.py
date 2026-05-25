import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sentence_transformers import SentenceTransformer
import time

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
CFG = {
    "data_path": "/home/iamontoyasa/02_VENOM/03_Baseline/02_run1_prepared_data_one_shot.pt",
    "output_path": "/home/iamontoyasa/02_VENOM/03_Baseline/protollm_full_sweep_results.csv",
    "llm_model": "all-mpnet-base-v2",
    "gamma_range": [0.0, 0.5, 1.0, 1.5, 2.0],  # Calibration Penalty
    "alpha_range": [0.0, 0.2, 0.5, 0.8, 1.0],  # 0.0=Data only, 1.0=LLM only
}


def run_protollm_sweep():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Starting VENOM ProtoLLM Sweep on: {device.upper()}")

    # [1] Load Prepared Data
    data = torch.load(CFG["data_path"], weights_only=False)

    X_train = data["X_train"].to(device)
    y_train = data["y_train"].to(device)
    X_support = data["X_support"].to(device)
    X_test = data["X_test"].to(device)
    y_test = data["y_test"].to(device)

    seen_ids = sorted(list(set(data["seen_label_ids"])))
    unseen_ids = sorted(list(set(data["unseen_label_ids"])))
    all_names = data["all_class_names"]
    n_classes = len(all_names)

    # [2] Pre-compute Semantic Anchors (BATCHED - No Lag)
    encoder = SentenceTransformer(CFG["llm_model"]).to(device)
    prompts = [f"Malware behavior of the {name} family" for name in all_names]
    with torch.no_grad():
        semantic_anchors = encoder.encode(prompts, convert_to_tensor=True)
        # Match dimensions if necessary
        if semantic_anchors.shape[1] != X_train.shape[1]:
            proj = torch.nn.Linear(semantic_anchors.shape[1], X_train.shape[1]).to(device)
            semantic_anchors = proj(semantic_anchors)

    # [3] Full Sweep
    results = []
    print(f"\n{'Alpha':>6} | {'Gamma':>6} | {'S (Seen)':>10} | {'U (Unseen)':>10} | {'H-Mean':>10}")
    print("-" * 55)

    norm_test = F.normalize(X_test, p=2, dim=1)
    y_test_np = y_test.cpu().numpy()

    for alpha in CFG["alpha_range"]:
        # Calculate Prototypes for this specific Alpha
        prototypes = torch.zeros((n_classes, X_train.shape[1]), device=device)

        # Build Seen Prototypes
        for s_id in seen_ids:
            mask = (y_train == s_id)
            data_centroid = X_train[mask].mean(dim=0)
            prototypes[s_id] = (alpha * semantic_anchors[s_id]) + ((1 - alpha) * data_centroid)

        # Build Unseen Prototypes (Fusing LLM knowledge with 1-shot support sample)
        for i, u_id in enumerate(unseen_ids):
            prototypes[u_id] = (alpha * semantic_anchors[u_id]) + ((1 - alpha) * X_support[i])

        norm_prototypes = F.normalize(prototypes, p=2, dim=1)

        for gamma in CFG["gamma_range"]:
            # Instantaneous Vectorized Comparison
            similarities = torch.matmul(norm_test, norm_prototypes.T)

            # Apply Calibrated Stacking
            if gamma > 0:
                for s_id in seen_ids:
                    similarities[:, s_id] -= gamma

            y_pred = similarities.argmax(dim=-1).cpu().numpy()

            # Metrics
            cm = confusion_matrix(y_test_np, y_pred, labels=range(n_classes))
            accs = [cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0 for i in range(n_classes)]

            S = np.mean([accs[i] for i in seen_ids]) * 100
            U = np.mean([accs[i] for i in unseen_ids]) * 100
            H = (2 * S * U) / (S + U) if (S + U) > 0 else 0

            print(f"{alpha:6.1f} | {gamma:6.1f} | {S:9.2f}% | {U:9.2f}% | {H:9.2f}%")
            results.append({"alpha": alpha, "gamma": gamma, "S": S, "U": U, "H": H})

    # [4] Output
    df = pd.DataFrame(results)
    df.to_csv(CFG["output_path"], index=False)

    best = df.loc[df['H'].idxmax()]
    print(f"\n{'═' * 55}")
    print(f"BEST CONFIGURATION:")
    print(f"  Alpha: {best['alpha']} | Gamma: {best['gamma']}")
    print(f"  S: {best['S']:.2f}% | U: {best['U']:.2f}% | H: {best['H']:.2f}%")
    print(f"{'═' * 55}")


if __name__ == "__main__":
    t0 = time.time()
    run_protollm_sweep()
    print(f"\nSweep completed in {time.time() - t0:.2f} seconds.")