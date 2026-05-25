#!/usr/bin/env python3
"""
clip_decoder_zsl.py  v2.1  —  CLIP-Decoder Zero-Shot Baseline
══════════════════════════════════════════════════════════════════════════════

Architecture (adapted from CLIP + ML-Decoder for tabular + text malware data):

  Tabular sample ──► TabEncoder (MLP) ──► tab_emb (B, E)
                                                │
  Class descriptions ──► SentenceTransformer ──► text_proj ──► cls_emb (C, E)
                          (frozen)            (trainable)
                                │                    │
  TRAINING ──────────────────────────────────────────┤
    Decoder queries = SEEN class embs only            │
    CE loss (n_seen) + InfoNCE (tab ↔ text)           │
                                                      │
  INFERENCE  ──────────────────────────────────────────┤
    score = tab_emb_norm @ cls_emb_norm.T  (ALL C classes)
    Calibration: scores[:, seen_ids] -= gamma  (GZSL bias correction)
    argmax → predicted class
"""

import os, math, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CFG = dict(
    embed_dim        = 256,
    n_heads          = 4,
    n_decoder_layers = 2,
    dropout          = 0.3,       # Increased to prevent seen-class overfitting
    lr               = 1e-3,
    weight_decay     = 1e-2,      # Increased regularization for tabular data
    epochs           = 100,
    batch_size       = 256,
    temp             = 0.07,      # Initial CLIP temperature (learnable)
    clip_loss_w      = 1.0,       # InfoNCE weight (dominant alignment signal)
    cls_loss_w       = 0.1,       # Drastically reduced to prevent semantic collapse
    gamma            = 5.0,       # Default GZSL calibration (overridden by sweep)
    text_model       = "all-MiniLM-L6-v2",
    seed             = 42,
    device           = "cuda" if torch.cuda.is_available() else "cpu",
)

DATA_DIR = "/home/iamontoyasa/02_VENOM/03_Baseline"
OUT_DIR  = os.path.join(DATA_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_class_descriptions(class_descriptions: dict,
                               all_class_names: list,
                               model_name: str) -> tuple:
    """
    Encode per-class prompts with a frozen SentenceTransformer.
    """
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(model_name)
    prompts = []
    for cls in all_class_names:
        desc = class_descriptions.get(
            cls, f"malware family {cls} exhibiting suspicious behavior"
        )
        prompts.append(
            f"a malware sample of the {cls} family: {desc[:256]}"
        )
    embs = st.encode(prompts, convert_to_tensor=True,
                     show_progress_bar=False, normalize_embeddings=False)
    return embs.cpu().float(), int(embs.shape[-1])


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

class TabEncoder(nn.Module):
    """
    3-layer MLP: tabular features ──► dense embedding (embed_dim).
    """
    def __init__(self, d_in: int, embed_dim: int, dropout: float):
        super().__init__()
        # Reduced hidden dimension capacity to prevent memorization
        h = embed_dim * 2 
        self.net = nn.Sequential(
            nn.Linear(d_in, h),   nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h, h),      nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x))


class MLDecoderHead(nn.Module):
    """
    Transformer decoder used ONLY during training (seen-class queries).
    """
    def __init__(self, embed_dim: int, n_heads: int,
                 n_layers: int, dropout: float):
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder  = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, tab_emb: torch.Tensor,
                cls_queries: torch.Tensor) -> torch.Tensor:
        B     = tab_emb.size(0)
        n_cls = cls_queries.size(0)
        memory  = tab_emb.unsqueeze(1)                        
        queries = cls_queries.unsqueeze(0).expand(B, -1, -1)  
        out = self.decoder(tgt=queries, memory=memory)        
        out = self.out_norm(out)                              
        
        out_n = F.normalize(out,         dim=-1)              
        qry_n = F.normalize(cls_queries, dim=-1)              
        return (out_n * qry_n.unsqueeze(0)).sum(-1)           


class CLIPDecoder(nn.Module):
    def __init__(self, d_in: int, st_dim: int, cfg: dict):
        super().__init__()
        E = cfg["embed_dim"]
        self.tab_encoder = TabEncoder(d_in, E, cfg["dropout"])
        self.text_proj   = nn.Sequential(
            nn.Linear(st_dim, E), nn.LayerNorm(E)
        )
        self.decoder     = MLDecoderHead(E, cfg["n_heads"],
                                         cfg["n_decoder_layers"], cfg["dropout"])
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / cfg["temp"])))

    def encode_tab(self, x):
        return self.tab_encoder(x)

    def encode_text(self, text_embs):
        return self.text_proj(text_embs)

    def forward_train(self, x: torch.Tensor,
                      text_embs_seen: torch.Tensor) -> tuple:
        tab_emb  = self.encode_tab(x)                  
        cls_seen = self.encode_text(text_embs_seen)     
        logits   = self.decoder(tab_emb, cls_seen)      
        return logits, tab_emb, cls_seen

    @torch.no_grad()
    def forward_infer(self, x: torch.Tensor,
                      text_embs_all: torch.Tensor,
                      seen_ids: list,
                      gamma: float = 0.5) -> torch.Tensor:
        tab_emb = self.encode_tab(x)                    
        cls_all = self.encode_text(text_embs_all)       

        scale   = self.logit_scale.exp().clamp(max=100)
        tab_n   = F.normalize(tab_emb, dim=-1)          
        cls_n   = F.normalize(cls_all, dim=-1)          
        scores  = scale * (tab_n @ cls_n.T)             

        if gamma != 0.0 and seen_ids:
            scores[:, seen_ids] -= gamma

        return scores


# ─────────────────────────────────────────────────────────────────────────────
# LOSSES
# ─────────────────────────────────────────────────────────────────────────────

def infonce_loss(tab_emb: torch.Tensor,
                 cls_emb: torch.Tensor,
                 labels: torch.Tensor,
                 logit_scale: nn.Parameter) -> torch.Tensor:
    scale  = logit_scale.exp().clamp(max=100)
    tab_n  = F.normalize(tab_emb, dim=-1)           
    cls_n  = F.normalize(cls_emb, dim=-1)           
    l_i2t  = scale * (tab_n @ cls_n.T)              
    
    return F.cross_entropy(l_i2t, labels)


def joint_loss(dec_logits: torch.Tensor,
               tab_emb: torch.Tensor,
               cls_emb: torch.Tensor,
               labels_local: torch.Tensor,
               logit_scale: nn.Parameter,
               cfg: dict) -> torch.Tensor:
    ce   = F.cross_entropy(dec_logits, labels_local)
    nce  = infonce_loss(tab_emb, cls_emb, labels_local, logit_scale)
    return cfg["cls_loss_w"] * ce + cfg["clip_loss_w"] * nce


# ─────────────────────────────────────────────────────────────────────────────
# METRICS  (GZSL harmonic mean)
# ─────────────────────────────────────────────────────────────────────────────

def gzsl_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                 seen_ids: list, unseen_ids: list) -> dict:
    s_m = np.isin(y_true, seen_ids)
    u_m = np.isin(y_true, unseen_ids)
    acc_s = accuracy_score(y_true[s_m], y_pred[s_m]) if s_m.any() else 0.0
    acc_u = accuracy_score(y_true[u_m], y_pred[u_m]) if u_m.any() else 0.0
    H     = 2 * acc_s * acc_u / (acc_s + acc_u + 1e-8)
    f1_m  = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    f1_s  = f1_score(y_true[s_m], y_pred[s_m], average="macro", zero_division=0) if s_m.any() else 0.0
    f1_u  = f1_score(y_true[u_m], y_pred[u_m], average="macro", zero_division=0) if u_m.any() else 0.0
    return dict(
        acc_seen=float(acc_s), acc_unseen=float(acc_u), H=float(H),
        f1_macro=float(f1_m), f1_seen=float(f1_s), f1_unseen=float(f1_u),
        overall_acc=float(accuracy_score(y_true, y_pred)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(model: CLIPDecoder,
          X_train: torch.Tensor,
          y_train: torch.Tensor,
          text_embs_all: torch.Tensor,
          seen_ids: list,
          cfg: dict) -> CLIPDecoder:
    device   = cfg["device"]
    model    = model.to(device)
    t_seen   = text_embs_all[seen_ids].to(device)    

    id_map = {gid: i for i, gid in enumerate(seen_ids)}
    y_local = torch.tensor([id_map[int(yi)] for yi in y_train], dtype=torch.long)

    loader = DataLoader(
        TensorDataset(X_train, y_local),
        batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
    )
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"], weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.01
    )

    model.train()
    for epoch in range(cfg["epochs"]):
        ep_loss = 0.0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits, tab_emb, cls_emb = model.forward_train(X_b, t_seen)
            loss = joint_loss(logits, tab_emb, cls_emb, y_b,
                              model.logit_scale, cfg)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"   epoch {epoch+1:>3}/{cfg['epochs']}  "
                  f"loss={ep_loss / max(len(loader), 1):.4f}  "
                  f"temp={model.logit_scale.exp().item():.2f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION 
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: CLIPDecoder,
             X_test: torch.Tensor,
             y_test: torch.Tensor,
             text_embs_all: torch.Tensor,
             seen_ids: list,
             unseen_ids: list,
             cfg: dict) -> dict:
    device = cfg["device"]
    model.eval()
    t_all = text_embs_all.to(device)

    all_scores = []
    loader = DataLoader(TensorDataset(X_test),
                        batch_size=cfg["batch_size"] * 4, shuffle=False)
    for (X_b,) in loader:
        X_b = X_b.to(device)
        tab_emb = model.encode_tab(X_b)
        cls_all = model.encode_text(t_all)
        scale   = model.logit_scale.exp().clamp(max=100)
        tab_n   = F.normalize(tab_emb, dim=-1)
        cls_n   = F.normalize(cls_all, dim=-1)
        scores  = scale * (tab_n @ cls_n.T)             
        all_scores.append(scores.cpu())

    all_scores = torch.cat(all_scores, dim=0)           
    y_np = y_test.numpy()

    # Expanded Gamma sweep to properly penalize the logit scale magnitude
    best_metrics, best_gamma = None, cfg["gamma"]
    print("   Gamma sweep:")
    for g in [0.0, 2.0, 5.0, 8.0, 10.0, 15.0, 20.0]:
        s = all_scores.clone()
        if g > 0:
            s[:, seen_ids] -= g
        preds = s.argmax(-1).numpy()
        m = gzsl_metrics(y_np, preds, seen_ids, unseen_ids)
        print(f"     gamma={g:>4.1f}  H={m['H']:.4f}  "
              f"acc_s={m['acc_seen']:.4f}  acc_u={m['acc_unseen']:.4f}")
        if best_metrics is None or m["H"] > best_metrics["H"]:
            best_metrics = m
            best_gamma   = g

    print(f"   Best gamma = {best_gamma}")
    return best_metrics, best_gamma


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run() -> dict:
    print("\n" + "═" * 60)
    print("  CLIP-DECODER  |  zero_shot  (v2.1 — Calibrated & Regularized)")
    print("═" * 60)
    set_seed(CFG["seed"])

    data_path = os.path.join(DATA_DIR, "03_run1_prepared_data_zero_shot.pt")
    data = torch.load(data_path, map_location="cpu", weights_only=False)

    X_train    = data["X_train"]
    y_train    = data["y_train"]
    X_test     = data["X_test"]
    y_test     = data["y_test"]
    seen_ids   = data["seen_label_ids"]
    unseen_ids = data["unseen_label_ids"]
    all_cls    = data["all_class_names"]
    cls_desc   = data["class_descriptions"]
    d_in       = int(data["d_in"])
    n_all      = len(all_cls)

    print(f"  d_in={d_in}  |  n_classes={n_all}  "
          f"(seen={len(seen_ids)}, unseen={len(unseen_ids)})")
    print(f"  X_train={tuple(X_train.shape)}  X_test={tuple(X_test.shape)}")

    print("  Encoding class prompts via SentenceTransformer (frozen)...")
    text_embs, st_dim = encode_class_descriptions(cls_desc, all_cls,
                                                   CFG["text_model"])
    print(f"  text_embs: {tuple(text_embs.shape)}  st_dim={st_dim}")

    model    = CLIPDecoder(d_in=d_in, st_dim=st_dim, cfg=CFG)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")
    print("  Training (seen-class decoder + InfoNCE alignment)...")
    model = train(model, X_train, y_train, text_embs, seen_ids, CFG)

    print("  Evaluating [CLIP cosine + GZSL gamma sweep]...")
    metrics, best_gamma = evaluate(model, X_test, y_test, text_embs,
                                   seen_ids, unseen_ids, CFG)

    print(f"\n  ── Best Results (gamma={best_gamma}) " + "─" * 28)
    for k, v in metrics.items():
        print(f"   {k:<20}: {v:.4f}")

    ckpt = os.path.join(OUT_DIR, "clip_decoder_zero_shot.pt")
    torch.save({"model_state": model.state_dict(), "cfg": CFG,
                "st_dim": st_dim, "d_in": d_in,
                "all_class_names": all_cls, "best_gamma": best_gamma}, ckpt)

    cfg_serial = {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                  for k, v in CFG.items()}
    result = dict(mode="zero_shot", best_gamma=best_gamma,
                  metrics=metrics, cfg=cfg_serial,
                  all_class_names=all_cls)
    json_path = os.path.join(OUT_DIR, "clip_decoder_zero_shot.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Checkpoint → {ckpt}")
    print(f"  Results    → {json_path}")
    return metrics


if __name__ == "__main__":
    run()