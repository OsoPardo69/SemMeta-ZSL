# train_supcon_debias.py
import os
CUDA_VISIBLE_DEVICES = 2

import json
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, default_data_collator, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from accelerate import Accelerator, DistributedDataParallelKwargs

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse

# Root of the smeta-zsl repo (two levels up from this script)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def parse_args():
    parser = argparse.ArgumentParser(description="AndMAL SupConDebias training")
    parser.add_argument(
        "--model", choices=["qwen", "qwen_coder","mistral_nemo","gemma", "llama3b", "llama8b"], default="llama8b",
        help="Select which model to use for fine-tuning."
    )
    parser.add_argument(
        "--unseen", nargs="+", default=None, metavar="CLASS",
        help="Custom unseen families (overrides default config)."
    )
    return parser.parse_args()
# --- 1. Configuration ---
class Config:
    def __init__(self, active_model="qwen"):
        self.MODELS = {
            "qwen": "Qwen/Qwen3-4B",
            "qwen_coder": "Qwen/Qwen2.5-Coder-14B-Instruct",
            "mistral_nemo": "mistralai/Mistral-Nemo-Base-2407",
            "gemma": "google/gemma-3-4b-it",
            "llama3b": "meta-llama/Llama-3.2-3B-Instruct",
            "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
        }
        self.ACTIVE_MODEL = active_model
        self.MODEL_ID = self.MODELS[self.ACTIVE_MODEL]

        self.DATA_FILE_PATH = os.path.join(_REPO_ROOT, "data", "andmal", "cti_reports.json")
        self.BASE_OUTPUT_DIR = os.path.join(_REPO_ROOT, "outputs", "andmal", "stage1")
        self.LOSS_FN = "SupConDebias"
        self.GAMMA = 0.2

        # --- Training ---
        self.NUM_EPOCHS = 10
        self.LEARNING_RATE = 1e-4
        self.BATCH_SIZE = 32  # Bumped to 32 to leverage server GPUs
        self.EFFECTIVE_B = 128  # Adjusted for the larger batch size
        self.WARMUP_RATIO = 0.06

        # --- Early Stopping ---
        self.EARLY_STOP_PATIENCE = 3

        # --- LoRA ---
        self.LORA_R = 16
        self.LORA_ALPHA = 32
        self.LORA_DROPOUT = 0.05

        # Dynamically map target modules
        if self.ACTIVE_MODEL == "gemma":
            self.LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]
        else:
            self.LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        self.UNSEEN_FAMILIES = ["airpush", "smsagent", "lotoor", "lockscreen", "skymobi", "kuguo"]
        self.MAX_LENGTH = 320

        self.RUN_NAME = (f"{self.ACTIVE_MODEL}_{self.LOSS_FN}_gamma{self.GAMMA}"
                         f"_epoc{self.NUM_EPOCHS}_lr{self.LEARNING_RATE}"
                         f"_bs{self.BATCH_SIZE}_r{self.LORA_R}")
        self.MODEL_OUTPUT_DIR = os.path.join(self.BASE_OUTPUT_DIR, self.RUN_NAME)


# --- 2. Loss Function ---
class SupConDebiasLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07, gamma=0.0):
        super().__init__()
        self.temperature      = temperature
        self.base_temperature = base_temperature
        self.gamma            = gamma

    def forward(self, features, labels):
        device        = features.device
        labels        = labels.to(device)
        batch_size    = features.shape[0]
        features_norm = F.normalize(features, p=2, dim=1)
        labels_view   = labels.contiguous().view(-1, 1)
        mask          = torch.eq(labels_view, labels_view.T).float().to(device)

        anchor_dot_contrast = torch.div(
            torch.matmul(features_norm, features_norm.T), self.temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits        = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size, device=device).view(-1, 1), 0
        )
        final_mask  = mask * logits_mask
        exp_logits  = torch.exp(logits) * logits_mask
        log_prob    = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        mask_sum    = final_mask.sum(1)
        mean_log_prob_pos = (final_mask * log_prob).sum(1) / (mask_sum + 1e-12)

        supcon_loss_per_sample = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        supcon_loss_per_sample = torch.nan_to_num(
            supcon_loss_per_sample, nan=0.0, posinf=0.0, neginf=0.0
        )
        supcon_loss  = supcon_loss_per_sample.mean()
        debias_loss  = self.debiasing_loss(features_norm) if self.gamma > 0 else torch.tensor(0.0, device=device)
        total_loss   = supcon_loss + self.gamma * debias_loss
        # Return component losses for logging
        return total_loss, supcon_loss.detach(), debias_loss.detach() if self.gamma > 0 else torch.tensor(0.0)

    def debiasing_loss(self, embeddings):
        mean_embedding   = torch.mean(embeddings, dim=0, keepdim=True)
        embeddings_norm  = F.normalize(embeddings, p=2, dim=1)
        mean_norm        = F.normalize(mean_embedding, p=2, dim=1)
        cosine_sim       = torch.matmul(embeddings_norm, mean_norm.t())
        return torch.mean(cosine_sim.pow(2))


# --- 3. Model Wrapper ---
class LlamaEmbedder(nn.Module):
    def __init__(self, llama_peft_model):
        super().__init__()
        self.llama_model = llama_peft_model

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.llama_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        if outputs.hidden_states is None:
            raise ValueError("No hidden_states.")
        last_hidden_state   = outputs.hidden_states[-1]
        batch_size          = last_hidden_state.shape[0]
        output_device       = last_hidden_state.device
        last_token_indices  = attention_mask.to(output_device).sum(dim=1) - 1
        embeddings = last_hidden_state[
            torch.arange(batch_size, device=output_device), last_token_indices
        ]
        return embeddings


# --- Helper: embedding extraction ---
def extract_embeddings(model, dataloader):
    all_embs, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop("label_id")
            embs   = model(**batch)
            all_embs.append(embs.cpu())
            all_labels.append(labels.cpu())
    return torch.cat(all_embs), torch.cat(all_labels)


# --- Helper: t-SNE plot ---
def plot_tsne_matplotlib(tsne_results, labels, title, label_encoder, save_path=None):
    unique_labels = np.unique(labels)
    num_classes   = len(unique_labels)
    cmap          = plt.cm.get_cmap('viridis', num_classes)
    fig, ax       = plt.subplots(figsize=(12, 8))
    for i, label_id in enumerate(unique_labels):
        indices     = np.where(labels == label_id)[0]
        family_name = label_encoder.inverse_transform([label_id])[0]
        ax.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
                   color=cmap(i / num_classes), label=family_name, alpha=0.7, s=50)
    ax.set_title(title)
    ax.set_xlabel('t-SNE Dim 1')
    ax.set_ylabel('t-SNE Dim 2')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    if save_path:
        plt.savefig(save_path)
        print(f"t-SNE plot saved to {save_path}")
    plt.close(fig)


# --- 4. Main ---
def main():
    args = parse_args()
    config = Config(active_model=args.model)

    # Allow CLI to override unseen families
    if args.unseen is not None:
        config.UNSEEN_FAMILIES = args.unseen

    # --- REPLACE THE OLD LINE WITH THESE TWO ---
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    if accelerator.is_main_process:
        print(f"--- Experiment: {config.RUN_NAME} ---")
        print(f"LR: {config.LEARNING_RATE} | Epochs: {config.NUM_EPOCHS} | "
              f"Early stop patience: {config.EARLY_STOP_PATIENCE} | "
              f"Warmup ratio: {config.WARMUP_RATIO}")
        os.makedirs(config.MODEL_OUTPUT_DIR, exist_ok=True)

    # Ministral tokenizer_config.json ships extra_special_tokens as a list
    # instead of a dict — patch the base class method to handle both
    import transformers.tokenization_utils_base as _tub
    _orig = _tub.PreTrainedTokenizerBase._set_model_specific_special_tokens

    def _patched(self, special_tokens=None):
        if isinstance(special_tokens, list):
            special_tokens = {t: t for t in special_tokens}
        return _orig(self, special_tokens)

    _tub.PreTrainedTokenizerBase._set_model_specific_special_tokens = _patched

    # Now load normally — also still need to override the class name
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
    except ValueError as e:
        if "TokenizersBackend" in str(e):
            from transformers import LlamaTokenizerFast
            tokenizer = LlamaTokenizerFast.from_pretrained(config.MODEL_ID)
        else:
            raise
    # --- Tokenizer ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
    except ValueError as e:
        if "TokenizersBackend" in str(e):
            # Ministral ships with a broken tokenizer_class in its config.
            # Fall back to LlamaTokenizerFast which shares the same SentencePiece vocab.
            from transformers import LlamaTokenizerFast
            tokenizer = LlamaTokenizerFast.from_pretrained(config.MODEL_ID)
        else:
            raise
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- Data ---
    with open(config.DATA_FILE_PATH, 'r') as f:
        all_data = json.load(f)
    df       = pd.DataFrame(all_data)
    seen_df  = df[~df['label'].isin(config.UNSEEN_FAMILIES)].copy()
    label_encoder = LabelEncoder()
    seen_df.loc[:, 'label_id'] = label_encoder.fit_transform(seen_df['label'])
    train_df, val_df = train_test_split(
        seen_df, test_size=0.1, random_state=42, stratify=seen_df['label_id']
    )
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset   = Dataset.from_pandas(val_df)

    def tokenize_function(examples):
        out = tokenizer(examples["description"], truncation=True,
                        padding="max_length", max_length=config.MAX_LENGTH)
        out["label_id"] = examples["label_id"]
        return out

    with accelerator.main_process_first():
        train_tok = train_dataset.map(tokenize_function, batched=True,
                                      remove_columns=train_dataset.column_names)
        val_tok   = val_dataset.map(tokenize_function, batched=True,
                                    remove_columns=val_dataset.column_names)
    train_tok.set_format("torch")
    val_tok.set_format("torch")

    # --- Gradient accumulation ---
    grad_accum_steps = config.EFFECTIVE_B // (config.BATCH_SIZE * accelerator.num_processes)
    accelerator.gradient_accumulation_steps = grad_accum_steps

    train_dataloader = DataLoader(train_tok, batch_size=config.BATCH_SIZE,
                                  collate_fn=default_data_collator,
                                  shuffle=True, pin_memory=True)
    val_dataloader   = DataLoader(val_tok, batch_size=config.BATCH_SIZE * 2,
                                  collate_fn=default_data_collator, pin_memory=True)

    # --- Model ---  CHANGE: added flash_attention_2 + max_memory to restrict to GPU 1+2
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        quantization_config=bnb_config,
        attn_implementation="sdpa",
        device_map={"": accelerator.device},
        #max_memory={1: "45GiB", 2: "45GiB", 0: "0GiB"},
        trust_remote_code=True
    )
    base_model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}  # ← key fix
    )
    lora_config = LoraConfig(
        r=config.LORA_R, lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT, bias="none", task_type="CAUSAL_LM"
    )
    peft_model    = get_peft_model(base_model, lora_config)
    model_wrapper = LlamaEmbedder(peft_model)
    loss_fn       = SupConDebiasLoss(gamma=config.GAMMA)

    # CHANGE: added weight_decay for regularization
    optimizer = torch.optim.AdamW(
        model_wrapper.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=0.01
    )

    # NEW: cosine LR scheduler with linear warmup — biggest single fix for flat training loss
    num_update_steps   = (len(train_dataloader) // grad_accum_steps) * config.NUM_EPOCHS
    num_warmup_steps   = int(num_update_steps * config.WARMUP_RATIO)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_update_steps
    )

    (model_prepared, optimizer,
     train_dataloader, val_dataloader,
     loss_fn, lr_scheduler) = accelerator.prepare(
        model_wrapper, optimizer, train_dataloader, val_dataloader, loss_fn, lr_scheduler
    )

    # --- Training loop ---
    best_val_loss     = float('inf')
    epochs_no_improve = 0           # NEW: early stopping counter
    global_step       = 0
    all_train_losses  = []          # (step, total, supcon, debias)
    all_val_losses    = []          # (epoch, val_loss)
    epoch_boundary_steps = []       # track which step each epoch ends at

    progress_bar = tqdm(
        range(num_update_steps),
        disable=not accelerator.is_local_main_process,
        desc="Overall Progress"
    )

    for epoch in range(config.NUM_EPOCHS):
        accelerator.print(f"\n--- Starting Epoch {epoch+1}/{config.NUM_EPOCHS} ---")
        model_prepared.train()
        epoch_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training",
                         disable=not accelerator.is_local_main_process, leave=False)

        for step, batch in enumerate(epoch_bar):
            with accelerator.accumulate(model_prepared):
                labels = batch.pop("label_id")
                embs   = model_prepared(**batch)
                total_loss, supcon_loss, debias_loss = loss_fn(embs, labels)
                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    # CHANGE: gradient clipping prevents exploding gradients
                    accelerator.clip_grad_norm_(model_prepared.parameters(), max_norm=1.0)
                    optimizer.step()
                    lr_scheduler.step()     # NEW: advance scheduler each update
                    optimizer.zero_grad()
                    global_step += 1
                    progress_bar.update(1)

                    if global_step % 10 == 0:
                        gathered = accelerator.gather(total_loss.detach().clone())
                        avg_t    = gathered.mean().item()
                        avg_s    = accelerator.gather(supcon_loss.clone()).mean().item()
                        avg_d    = accelerator.gather(debias_loss.clone()).mean().item() if config.GAMMA > 0 else 0.0
                        current_lr = lr_scheduler.get_last_lr()[0]
                        all_train_losses.append((global_step, avg_t, avg_s, avg_d))
                        # CHANGE: log supcon and debias separately for better diagnostics
                        accelerator.print(
                            f"Step {global_step}: loss={avg_t:.4f} "
                            f"(supcon={avg_s:.4f}, debias={avg_d:.4f}) lr={current_lr:.2e}"
                        )

        epoch_boundary_steps.append(global_step)

        # --- Validation ---
        accelerator.print(f"--- End of Epoch {epoch+1}, starting validation... ---")
        model_prepared.eval()
        epoch_val_losses = []
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validating", leave=False,
                              disable=not accelerator.is_local_main_process):
                labels   = batch.pop("label_id")
                embs     = model_prepared(**batch)
                vl, _, _ = loss_fn(embs, labels)
                epoch_val_losses.append(vl.detach())

        avg_val = accelerator.gather(torch.stack(epoch_val_losses)).mean().item()
        all_val_losses.append((epoch + 1, avg_val))
        accelerator.print(f"Epoch {epoch+1}: Val Loss = {avg_val:.4f} | Best = {best_val_loss:.4f}")

        if avg_val < best_val_loss:
            best_val_loss     = avg_val
            epochs_no_improve = 0
            accelerator.print(f"  ✅ New best val_loss: {best_val_loss:.4f}. Saving model...")
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model_prepared)
                unwrapped.llama_model.save_pretrained(config.MODEL_OUTPUT_DIR)
                tokenizer.save_pretrained(config.MODEL_OUTPUT_DIR)
                accelerator.print(f"  Model saved to {config.MODEL_OUTPUT_DIR}")
        else:
            epochs_no_improve += 1
            accelerator.print(f"  ⚠️  No improvement for {epochs_no_improve}/{config.EARLY_STOP_PATIENCE} epochs.")
            # NEW: early stopping
            if epochs_no_improve >= config.EARLY_STOP_PATIENCE:
                accelerator.print(
                    f"\n🛑 Early stopping triggered at epoch {epoch+1}. "
                    f"Best val loss: {best_val_loss:.4f}"
                )
                break

        accelerator.wait_for_everyone()

    accelerator.print("\n--- Training finished. ---")

    # --- Plots and artifacts ---
    if accelerator.is_main_process:
        # CHANGE: fixed x-axis bug — val losses now plotted at actual epoch boundary steps
        steps_list  = [r[0] for r in all_train_losses]
        total_list  = [r[1] for r in all_train_losses]
        supcon_list = [r[2] for r in all_train_losses]
        debias_list = [r[3] for r in all_train_losses]
        val_steps   = epoch_boundary_steps[:len(all_val_losses)]
        val_vals    = [r[1] for r in all_val_losses]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

        # Top: total + val loss
        ax1.plot(steps_list, total_list, 'b-', alpha=0.7, label='Train Total Loss')
        ax1.plot(val_steps, val_vals, 'r-s', linewidth=2, label='Val Loss (epoch end)')
        ax1.axhline(y=best_val_loss, color='green', linestyle='--', label=f'Best val={best_val_loss:.4f}')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'Loss Curves — {config.RUN_NAME}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bottom: supcon vs debias components
        ax2.plot(steps_list, supcon_list, 'b-', alpha=0.7, label='SupCon Loss')
        ax2.plot(steps_list, debias_list, 'orange', alpha=0.7, label='Debias Loss')
        ax2.set_xlabel('Global Step')
        ax2.set_ylabel('Component Loss')
        ax2.set_title('SupCon vs Debias Loss Components')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        loss_path = os.path.join(config.MODEL_OUTPUT_DIR, "loss_curve.png")
        plt.savefig(loss_path, dpi=150)
        plt.close(fig)
        print(f"Loss curve saved to {loss_path}")

        # Save loss history as JSON for later analysis
        history = {
            "train": [{"step": r[0], "total": r[1], "supcon": r[2], "debias": r[3]}
                      for r in all_train_losses],
            "val":   [{"epoch": r[0], "val_loss": r[1]} for r in all_val_losses],
            "best_val_loss": best_val_loss
        }
        with open(os.path.join(config.MODEL_OUTPUT_DIR, "loss_history.json"), 'w') as f:
            json.dump(history, f, indent=2)
        print("Loss history saved to loss_history.json")

        # t-SNE
        print("\nExtracting embeddings for t-SNE...")
        ft_embs, ft_labels = extract_embeddings(model_prepared, val_dataloader)
        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000,  # CHANGE: 300→1000 for stable layout
                    random_state=42, metric='cosine')
        tsne_results = tsne.fit_transform(ft_embs.numpy())
        plot_tsne_matplotlib(
            tsne_results, ft_labels.numpy(),
            f't-SNE — Fine-Tuned Embeddings (gamma={config.GAMMA})',
            label_encoder,
            save_path=os.path.join(config.MODEL_OUTPUT_DIR, "tsne_finetuned.png")
        )


if __name__ == "__main__":
    main()
