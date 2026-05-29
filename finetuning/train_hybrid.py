#!/usr/bin/env python3
"""
train_hybrid.py — Finetuning del modello ibrido LLaMA + 2-Simplicial su C4.

Pipeline:
1. Carica modello LLaMA 3.1 8B da HuggingFace
2. Calcola baseline PPL su Wikitext-2 (per early stopping)
3. Converte in ibrido con convert_llama_to_hybrid()
4. Crea AdamW con 3 gruppi di parametri
5. Training loop manuale con:
   - C4 streaming dataset
   - WandB logging
   - Validation ogni 500 step
   - Early stopping se PPL gap > 0.5
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

from utils.data import make_c4_train_loader, prepare_validation_batch
from utils.optimizer import create_optimizer_groups
from utils.metrics import evaluate_validation
from utils.wandb_utils import init_wandb, log_metrics, finish_wandb


# ==========================================================================
# Configurazione
# ==========================================================================

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

# Perplexity baseline di LLaMA 3.1 8B su Wikitext-2 (seq_len=512)
# Valore dalla literature, usato per early stopping
LLAMA_BASELINE_PPL = 8.2

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


# ==========================================================================
# Training loop
# ==========================================================================

def check_env_vars(wandb_active: bool):
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
        print(f"  {YELLOW}      1. Accettare licenza su https://hf.co/meta-llama/Llama-3.1-8B{NC}")
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

    check_env_vars(wandb_active)

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

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    print("  Tokenizer OK")

    # --- Baseline perplexity (per early stopping) ---
    print(f"\n[2/5] Baseline PPL: {LLAMA_BASELINE_PPL} (Wikitext-2, seq_len=512)")
    val_batch = prepare_validation_batch(
        tokenizer,
        seq_length=config["seq_length"],
        num_samples=config["val_samples"],
        device=device,
    )
    print(f"  Batch di validazione: {val_batch['input_ids'].shape}")

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
    param_groups = create_optimizer_groups(
        model,
        simplicial_indices=config["simplicial_indices"],
        lr_k2v2=config["lr_k2v2"],
        lr_k1v1=config["lr_k1v1"],
        weight_decay=config["weight_decay"],
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
    # Training loop
    # ======================================================================

    print(f"\n{'='*60}")
    print(f"  TRAINING: {config['max_steps']} steps")
    print(f"  Batch effettivo: {config['per_device_batch_size']} * {config['gradient_accumulation_steps']} = "
          f"{config['per_device_batch_size'] * config['gradient_accumulation_steps']}")
    print(f"  LR K2/V2: {config['lr_k2v2']}, LR K1/V1: {config['lr_k1v1']}")
    print(f"{'='*60}\n")

    global_step = 0
    cumulative_loss = 0.0
    best_ppl = float('inf')
    early_stopped = False

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

            # Log training loss (accumulata)
            avg_loss = cumulative_loss / config["gradient_accumulation_steps"]
            lr_k2v2 = optimizer.param_groups[2]["lr"] if len(optimizer.param_groups) > 2 else 0.0

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

            val_metrics = evaluate_validation(model, val_batch, config["simplicial_indices"])
            log_metrics(val_metrics, global_step, wandb_active)

            ppl = val_metrics["val/perplexity"]
            delta = ppl - LLAMA_BASELINE_PPL

            print(f"  PPL: {ppl:.2f} (baseline: {LLAMA_BASELINE_PPL}, delta: {delta:+.2f})")
            print(f"  L2 K1/K2: {val_metrics.get('val/l2_k1k2_mean', 0):.6f}")
            print(f"  L2 V1/V2: {val_metrics.get('val/l2_v1v2_mean', 0):.6f}")

            # Check miglior PPL
            if ppl < best_ppl:
                best_ppl = ppl
                print(f"  Nuova best PPL: {ppl:.2f}")

            # Early stopping
            if delta > config["max_perplexity_gap"]:
                print(f"\n{'='*50}")
                print(f"  EARLY STOPPING: PPL gap {delta:.2f} > {config['max_perplexity_gap']}")
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
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--checkpoint-dir", type=str)
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disabilita WandB")
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