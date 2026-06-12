#!/usr/bin/env python3
"""
train_hybrid.py — Finetuning del modello ibrido LLaMA + 2-Simplicial su C4.

Pipeline:
1. Carica modello da HuggingFace
2. Calcola baseline PPL runtime (su C4)
3. Converte in ibrido con convert_llama_to_hybrid()
4. Crea AdamW con 4 gruppi di parametri (frozen, standard, k1v1, k2v2)
5. Training loop manuale con:
   - C4 streaming dataset
   - WandB logging
   - Validation ogni 500 step
   - Early stopping se PPL > soglia
   - Checkpoint ogni 1000 step
6. Salva modello finale e checkpoint

Usage:
    python finetuning/train_hybrid.py                           # default
    python finetuning/train_hybrid.py --config custom.yaml       # config custom
    python finetuning/train_hybrid.py --max-steps 5000          # override singoli parametri
    python finetuning/train_hybrid.py --attention-type gram_det # GramDet
"""

import os
import sys
import math
import time
import gc
import glob
import shutil
import argparse
import yaml
from typing import Dict, Optional

import torch
import torch.nn.functional as F

# Aggiungi root del progetto al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.modeling.convert_to_hybrid import convert_llama_to_hybrid
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
from datasets import load_dataset

from finetuning.utils.data import make_c4_train_loader, prepare_validation_batch, prepare_c4_validation_batch
from finetuning.utils.optimizer import create_optimizer_groups
from finetuning.utils.metrics import evaluate_validation
from finetuning.utils.wandb_utils import init_wandb, log_metrics, finish_wandb


# ==========================================================================
# Configurazione
# ==========================================================================

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

# Baseline Monte Carlo per varianza geodesica su Gr(2, d)
# Calcolata con scripts/grassmann_baseline.py --dim <d> --runs 10
GRASSMANN_BASELINE = {128: 4.09, 64: 3.85}

# Colori ANSI per output
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
NC = "\033[0m"


def load_config(config_path: str = DEFAULT_CONFIG_PATH, overrides: dict = None) -> dict:
    """Carica config YAML e applica override da CLI."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if overrides:
        config.update(overrides)
    return config


def compute_baseline_ppl(
    model,
    tokenizer,
    device: str = "cuda",
    seq_length: int = 512,
    num_samples: int = 10,
) -> float:
    """
    Calcola la PPL baseline su C4 per il modello appena caricato (forward pass only).

    Args:
        model: modello LLaMA fresco (non convertito)
        tokenizer: tokenizer
        device: device
        seq_length: lunghezza sequenza
        num_samples: numero di campioni C4

    Returns:
        PPL media
    """
    try:
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    except Exception:
        print(f"  {YELLOW}[WARN]{NC} C4 validation non disponibile, uso wikitext-2")
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test", streaming=True)

    total_nll = 0.0
    total_tokens = 0
    count = 0

    model.eval()
    for example in ds:
        if count >= num_samples:
            break
        text = example.get("text", "")
        if not text.strip():
            continue
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_length)
        if enc["input_ids"].shape[1] < 10:
            continue
        input_ids = enc["input_ids"].to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            nll = outputs.loss.item() * (input_ids.shape[1] - 1)
            total_nll += nll
            total_tokens += input_ids.shape[1] - 1
        count += 1

    model.train()
    avg_nll = total_nll / max(total_tokens, 1)
    ppl = math.exp(avg_nll)
    return ppl


def check_env_vars(wandb_active: bool, model_name: str):
    """Verifica variabili d'ambiente necessarie."""
    hf_token = os.environ.get("HF_TOKEN")
    wandb_key = os.environ.get("WANDB_API_KEY")

    print(f"\n  {'─'*40}")
    print("  PREREQUISITI:")
    print(f"  {'─'*40}")

    # HF_TOKEN
    if hf_token:
        print(f"  {GREEN}[OK]{NC} HF_TOKEN impostato")
    else:
        print(f"  {YELLOW}[WARN]{NC} HF_TOKEN non impostato")
        print(f"  {YELLOW}      Il download del modello da HuggingFace richiede:{NC}")
        print(f"  {YELLOW}      1. Accettare licenza su https://hf.co/{model_name}{NC}")
        print(f"      2. Generare token su https://hf.co/settings/tokens")
        print(f"      3. export HF_TOKEN=hf_yourtoken")
        print(f"      Oppure assicurati di essere loggato con huggingface-cli login")

    # WANDB_API_KEY
    if wandb_active:
        if wandb_key or os.path.exists(os.path.expanduser("~/.netrc")):
            print(f"  {GREEN}[OK]{NC} WANDB_API_KEY impostato")
        else:
            print(f"  {YELLOW}[WARN]{NC} WANDB_API_KEY non impostato e .netrc non trovato")
            print(f"  {YELLOW}      WandB non potra' autenticarsi.{NC}")
            print(f"      Per attivare: export WANDB_API_KEY=your_wandb_key")
            print(f"      Oppure: wandb login")
    else:
        print(f"  {YELLOW}[INFO]{NC} WandB disabilitato (non installato o --no-wandb)")
        print(f"      Logging solo su stdout.")

    print(f"  {'─'*40}\n")


def train(config: dict):
    """
    Esegue il finetuning del modello ibrido.

    Args:
        config: dict con tutti gli iperparametri
    """
    # --- Init ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  Training ibrido LLaMA + 2-Simplicial")
    print(f"  Device: {device}")
    print(f"  Attention type: {config['attention_type']}")
    print(f"  Indici layer simpliciali: {config['simplicial_indices']}")
    print(f"{'='*60}")

    # --- WandB + check prerequisites ---
    wandb_active = init_wandb(config)
    if wandb_active:
        import wandb
        wandb.config.update(config, allow_val_change=True)

    check_env_vars(wandb_active, config["model_name"])

    # --- Carica modello ---
    print(f"\n[1/5] Caricamento modello: {config['model_name']}")
    hf_token = os.environ.get("HF_TOKEN")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        token=hf_token,
    )
    model.train()
    print("  OK")

    # --- Deriva architettura dal model.config ---
    head_dim = getattr(model.config, "head_dim", None) or (
        model.config.hidden_size // model.config.num_attention_heads
    )
    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    num_q_heads = model.config.num_attention_heads
    num_kv_heads = getattr(model.config, "num_key_value_heads", num_q_heads)
    num_repeats = num_q_heads // num_kv_heads if num_kv_heads > 0 else 1

    print(f"  Architettura derivata: hidden={hidden_size}, layers={num_layers}, "
          f"heads={num_q_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    print("  Tokenizer OK")

    # --- Baseline PPL runtime (stesso dominio del training) ---
    baseline_ppl = config.get("baseline_ppl", None)
    if baseline_ppl is None:
        print(f"\n  Calcolo baseline PPL su C4 ({config.get('val_samples', 50)} campioni)...")
        baseline_ppl = compute_baseline_ppl(
            model, tokenizer, device=device,
            seq_length=config["seq_length"],
            num_samples=config.get("val_samples", 50),
        )
        print(f"  Baseline LLaMA su C4: PPL = {baseline_ppl:.2f}")
    else:
        print(f"  Baseline LLaMA su C4: PPL = {baseline_ppl:.2f} (da config)")

    print(f"\n[2/5] Batch di validazione su C4 ({config['val_samples']} campioni)...")
    val_batch_c4 = prepare_c4_validation_batch(
        tokenizer,
        seq_length=config["seq_length"],
        num_samples=config["val_samples"],
        device=device,
    )
    print(f"  Batch C4: {val_batch_c4['input_ids'].shape}")

    # --- Converti in ibrido ---
    print(f"\n[3/5] Conversione in ibrido ({config['attention_type']})...")
    model, converted = convert_llama_to_hybrid(
        model,
        simplicial_indices=config["simplicial_indices"],
        alpha=config["alpha"],
        w1=config["w1"],
        w2=config["w2"],
        attention_type=config["attention_type"],
        gram_window=config.get("gram_window", 8),
    )
    print(f"  Layer convertiti: {converted}")

    # --- Ottimizzatore ---
    print(f"\n[4/5] Creazione ottimizzatore...")
    lr_standard = config.get("lr_standard", 5e-6)
    param_groups = create_optimizer_groups(
        model,
        simplicial_indices=config["simplicial_indices"],
        lr_k2v2=config["lr_k2v2"],
        lr_k1v1=config["lr_k1v1"],
        lr_standard=lr_standard,
        weight_decay=config["weight_decay"],
        attention_type=config["attention_type"],
    )
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(config["beta1"], config["beta2"]),
    )

    # --- Scheduler (warmup lineare) ---
    def get_lr(step):
        if step < config["warmup_steps"]:
            return step / config["warmup_steps"]
        return 1.0 - (step - config["warmup_steps"]) / max(config["max_steps"] - config["warmup_steps"], 1)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)

    # --- Dataset ---
    print(f"\n[5/5] Dataset: {config['dataset_name']}/{config['dataset_config']} (streaming)")
    train_loader = make_c4_train_loader(
        tokenizer,
        seq_length=config["seq_length"],
    )

    # --- Checkpoint dir ---
    checkpoint_dir = config["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ======================================================================
    # RESUME: Carica pesi + optimizer da checkpoint precedente
    # ======================================================================
    resume_path = config.get("resume", None)
    resume_step = 0
    if resume_path is not None:
        from safetensors.torch import load_file as safetensors_load

        print(f"\n  RESUME da {resume_path}...")
        
        # 1. Carica state_dict dal checkpoint
        safetensor_files = glob.glob(os.path.join(resume_path, "*.safetensors"))
        if not safetensor_files:
            idx = os.path.join(resume_path, "model.safetensors.index.json")
            if os.path.exists(idx):
                import json
                with open(idx) as f:
                    ix = json.load(f)
                safetensor_files = list(set(ix["weight_map"].values()))
                safetensor_files = [os.path.join(resume_path, f) for f in safetensor_files]
        
        state_dict = {}
        for sf in safetensor_files:
            state_dict.update(safetensors_load(sf))
        
        # 2. Sovrascrivi SOLO i pesi dei layer simpliciali
        loaded = 0
        for name, param in model.named_parameters():
            if name in state_dict and any(f"layers.{i}." in name for i in config["simplicial_indices"]):
                param.data.copy_(state_dict[name].to(param.device))
                loaded += 1
        
        print(f"  Caricati {loaded} pesi simpliciali dal checkpoint.")
        
        # 3. Carica training_state.pt (optimizer, scheduler, step, best_ppl)
        state_path = os.path.join(resume_path, "training_state.pt")
        if os.path.exists(state_path):
            training_state = torch.load(state_path, map_location=device)
            optimizer.load_state_dict(training_state["optimizer"])
            scheduler.load_state_dict(training_state["scheduler"])
            resume_step = training_state["step"]
            best_ppl = training_state.get("best_ppl", float('inf'))
            print(f"  Riprendo da step {resume_step} (best PPL: {best_ppl:.2f})")
        else:
            resume_step = int(os.path.basename(resume_path).replace("checkpoint-", ""))
            best_ppl = float('inf')
            print(f"  Nessun training_state.pt, riprendo da step {resume_step}")
        
        del state_dict
        gc.collect()

    # ======================================================================
    # Training loop
    # ======================================================================

    print(f"\n{'='*60}")
    print(f"  TRAINING: {config['max_steps']} steps")
    print(f"  Batch effettivo: {config['per_device_batch_size']} * {config['gradient_accumulation_steps']} = "
          f"{config['per_device_batch_size'] * config['gradient_accumulation_steps']}")
    print(f"  LR K2/V2: {config['lr_k2v2']}, LR K1/V1: {config['lr_k1v1']}, LR Standard: {lr_standard}")
    print(f"{'='*60}\n")

    global_step = resume_step  # 0 se fresh, N se resume
    cumulative_loss = 0.0
    best_ppl = float('inf')
    early_stopped = False
    patience_counter = 0
    max_patience = 3
    early_stop_ppl = config.get("early_stop_ppl", None)

    for batch in train_loader:
        if global_step >= config["max_steps"]:
            break

        # Forward
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss / config["gradient_accumulation_steps"]
        cumulative_loss += loss.item()

        # Backward
        loss.backward()

        if (global_step + 1) % config["gradient_accumulation_steps"] == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Log training loss
            avg_loss = cumulative_loss
            lr_k2v2 = optimizer.param_groups[-1]["lr"] if len(optimizer.param_groups) > 2 else 0.0

            log_metrics({
                "train/loss": avg_loss,
                "train/pprox": math.exp(avg_loss) if avg_loss < 20 else float('inf'),
                "train/lr": lr_k2v2,
            }, global_step, wandb_active)

            cumulative_loss = 0.0

        global_step += 1

        # ==================================================================
        # Validation
        # ==================================================================
        if global_step % config["val_every"] == 0:
            print(f"\n{'─'*50}")
            print(f"  Validazione a step {global_step}")
            print(f"{'─'*50}")

            # Validazione su C4 (stesso dominio del training)
            val_metrics = evaluate_validation(
                model, val_batch_c4, config["simplicial_indices"],
                attention_type=config["attention_type"],
            )
            log_metrics(val_metrics, global_step, wandb_active)

            ppl = val_metrics["val/perplexity"]
            delta = ppl - baseline_ppl

            print(f"  PPL: {ppl:.2f} (baseline: {baseline_ppl:.2f}, delta: {delta:+.2f})")
            k1k2 = val_metrics.get('val/l2_k1k2_mean', "N/A")
            v1v2 = val_metrics.get('val/l2_v1v2_mean', "N/A")
            l2_k1k2_str = f"{k1k2:.6f}" if isinstance(k1k2, float) else str(k1k2)
            l2_v1v2_str = f"{v1v2:.6f}" if isinstance(v1v2, float) else str(v1v2)
            print(f"  L2 K1/K2: {l2_k1k2_str}")
            print(f"  L2 V1/V2: {l2_v1v2_str}")

            # Early stopping per PPL assoluta
            if early_stop_ppl is not None and ppl > early_stop_ppl:
                print(f"\n{'='*50}")
                print(f"  EARLY STOPPING: PPL={ppl:.2f} > {early_stop_ppl}")
                print(f"{'='*50}\n")
                early_stopped = True
                log_metrics({"train/early_stopped_ppl_at": global_step}, global_step, wandb_active)
                break

            # Early stopping basato su patience
            if ppl >= best_ppl:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{max_patience}")
            else:
                patience_counter = 0
                best_ppl = ppl
                print(f"  Nuova best PPL: {ppl:.2f} (patience resettata)")

            if patience_counter >= max_patience:
                print(f"\n{'='*50}")
                print(f"  EARLY STOPPING: PPL non migliora da {max_patience} validation ({best_ppl:.2f})")
                print(f"{'='*50}\n")
                early_stopped = True
                log_metrics({"train/early_stopped_at": global_step}, global_step, wandb_active)
                break

        # ==================================================================
        # Checkpoint
        # ==================================================================
        if global_step % config["save_every"] == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint-{global_step}")
            print(f"\n  Checkpoint saved: {ckpt_path}")
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            torch.save({
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": global_step,
                "best_ppl": best_ppl,
            }, os.path.join(ckpt_path, "training_state.pt"))
            log_metrics({"train/checkpoint_saved": global_step}, global_step, wandb_active)

            # Auto-pulizia: cancella checkpoint piu' vecchi (tieni solo ultimi 2)
            all_ckpts = sorted(
                glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")),
                key=lambda p: int(os.path.basename(p).split("-")[-1]),
            )
            for old_ckpt in all_ckpts[:-2]:
                print(f"  Spazio: cancellazione checkpoint vecchio {old_ckpt}")
                shutil.rmtree(old_ckpt)

    # ======================================================================
    # Fine training
    # ======================================================================

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETATO")
    print(f"  Steps: {global_step}/{config['max_steps']}")
    print(f"  Best PPL: {best_ppl:.2f}")
    print(f"  Early stopped: {early_stopped}")
    print(f"{'='*60}\n")

    # Salva modello finale
    final_path = os.path.join(checkpoint_dir, "final")
    print(f"  Salvando modello finale in {final_path}...")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print("  OK")

    finish_wandb(wandb_active)
    print("\nFatto!\n")


# ==========================================================================
# CLI
# ==========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Finetuning ibrido LLaMA + 2-Simplicial su C4")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH,
                        help=f"Path config YAML (default: {DEFAULT_CONFIG_PATH})")
    parser.add_argument("--attention-type", type=str, choices=["simplicial", "gram_det"])
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--lr-k2v2", type=float)
    parser.add_argument("--lr-k1v1", type=float)
    parser.add_argument("--lr-standard", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--checkpoint-dir", type=str)
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disabilita WandB")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path a checkpoint da cui riprendere training")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Carica config + override CLI
    overrides = {k: v for k, v in vars(args).items()
                 if v is not None and k != "config"}
    if args.no_wandb:
        overrides["wandb_project"] = None

    config = load_config(args.config, overrides=overrides)
    train(config)