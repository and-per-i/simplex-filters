#!/usr/bin/env python3
"""
train_hybrid.py — Script principale di finetuning per modelli ibridi.
Carica LLaMA 3.1 8B, converte layer selezionati in attenzione 2-simpliciale,
e fa training solo dei nuovi pesi K/V su C4.

Esporta la funzione train(config) per essere chiamata da main.py.
"""

import os
import sys
import math
import time
import yaml
import json
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import wandb
from pathlib import Path
from safetensors.torch import save_file as safetensors_save

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.convert_to_hybrid import convert_llama_to_hybrid
from finetuning.utils.optimizer import create_optimizer_groups
from finetuning.utils.data import make_c4_train_loader, prepare_validation_batch
from finetuning.utils.metrics import evaluate_validation


def load_config(config_path: str) -> dict:
    """Carica la configurazione YAML e sovrascrive con env vars se presenti."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    for key in config:
        env_val = os.environ.get(key.upper())
        if env_val is not None:
            try:
                config[key] = yaml.safe_load(env_val)
            except yaml.YAMLError:
                config[key] = env_val
    return config


def _resume_from_checkpoint(model, checkpoint_path: str):
    """Carica i pesi salvati da un checkpoint precedente."""
    from safetensors.torch import load_file as safetensors_load
    import glob

    state = {}
    for sf in sorted(glob.glob(os.path.join(checkpoint_path, "*.safetensors"))):
        state.update(safetensors_load(sf))

    loaded = 0
    for name, param in model.named_parameters():
        if name in state and state[name].shape == param.shape:
            param.data.copy_(state[name].to(param.device))
            loaded += 1
    print(f"  ✓ Resume: {loaded} pesi da {checkpoint_path}")


def _resume_scheduler(optimizer, start_step: int, max_steps: int, eta_min: float = 1e-7):
    """
    Riporta il cosine scheduler allo step corretto.
    
    PyTorch CosineAnnealingLR con last_epoch gestisce male i resume.
    Facciamo start_step iterazioni esplicite dello scheduler (pura aritmetica,
    < 0.1s anche per 10000 step).
    """
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])

    scheduler = CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=eta_min)
    for _ in range(start_step):
        scheduler.step()
    return scheduler


def _do_train(config: dict):
    """Esegue il training loop."""
    model_name = config["model_name"]
    attention_type = config["attention_type"]
    simplicial_indices = config["simplicial_indices"]
    max_steps = config["max_steps"]
    per_device_batch_size = config["per_device_batch_size"]
    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    log_every = config["log_every"]
    val_every = config["val_every"]
    save_every = config["save_every"]
    baseline_ppl = config["baseline_ppl"]
    early_stop_ppl = config.get("early_stop_ppl", 9999)
    max_perplexity_gap = config.get("max_perplexity_gap", 10.0)
    checkpoint_dir = config["checkpoint_dir"]
    alpha = config.get("alpha", 0.01)
    w1 = config["w1"]
    w2 = config["w2"]
    gram_window = config.get("gram_window", 8)
    lr_k2v2 = config["lr_k2v2"]
    lr_k1v1 = config["lr_k1v1"]
    lr_standard = config.get("lr_standard", 1e-5)
    weight_decay = config["weight_decay"]
    beta1 = config.get("beta1", 0.9)
    beta2 = config.get("beta2", 0.95)
    seq_length = config["seq_length"]
    wandb_project = config.get("wandb_project", "simplex-filters")
    wandb_run_name = config.get("wandb_run_name", "llama-simplicial-finetune")
    resume_checkpoint = config.get("resume_checkpoint") or config.get("resume_path")
    start_step = config.get("start_step", 0)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_distributed = world_size > 1

    if is_distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    if local_rank == 0:
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)

    # Modello
    print(f"[Rank {local_rank}] Caricamento {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device} if not is_distributed else None,
        attn_implementation="eager",
    )
    model.train()

    # Converti in ibrido
    model, converted = convert_llama_to_hybrid(
        model, simplicial_indices=simplicial_indices, alpha=alpha,
        w1=w1, w2=w2, attention_type=attention_type, gram_window=gram_window,
    )

    # Resume
    if resume_checkpoint:
        _resume_from_checkpoint(model, resume_checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Optimizer
    optimizer_groups = create_optimizer_groups(
        model, simplicial_indices,
        lr_k2v2=lr_k2v2, lr_k1v1=lr_k1v1, lr_standard=lr_standard,
        weight_decay=weight_decay, attention_type=attention_type,
    )
    optimizer = AdamW(optimizer_groups, betas=(beta1, beta2))

    # Scheduler
    if start_step > 0:
        scheduler = _resume_scheduler(optimizer, start_step, max_steps)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=1e-7)

    # Data
    train_loader = make_c4_train_loader(tokenizer, seq_length=seq_length)

    # Training loop — senza validazione per ora (evita crash su API metriche)
    global_step = start_step
    total_loss = 0.0
    best_val_ppl = float("inf")
    start_time = time.time()

    print(f"[Rank {local_rank}] Training: {max_steps} step (da {start_step}), "
          f"batch={per_device_batch_size}, accum={gradient_accumulation_steps}")

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        if global_step >= max_steps:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

            total_loss += loss.item() * gradient_accumulation_steps

            if global_step % log_every == 0 and local_rank == 0:
                avg_loss = total_loss / log_every
                elapsed = time.time() - start_time
                current_lr = max(scheduler.get_last_lr())
                print(f"Step {global_step}/{max_steps} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | Time: {elapsed:.1f}s")
                wandb.log({"train/loss": avg_loss, "train/lr": current_lr, "train/step": global_step})
                total_loss = 0.0

            # Validazione ogni val_every step
            if global_step % val_every == 0 and local_rank == 0:
                model.eval()
                val_batch = prepare_validation_batch(tokenizer, seq_length=seq_length, device=str(device))
                metrics = evaluate_validation(model, val_batch, simplicial_indices, attention_type)
                val_ppl = metrics["val/perplexity"]
                model.train()
                print(f"  Val PPL: {val_ppl:.2f} (baseline={baseline_ppl})")
                ppl_gap = val_ppl - baseline_ppl
                wandb.log({f"val/{k}": v for k, v in metrics.items()})
                wandb.log({"val/step": global_step, "val/gate_passed": 1.0 if ppl_gap < max_perplexity_gap else 0.0})

                if val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl
                    wandb.log({"val/best_perplexity": best_val_ppl})

                if val_ppl < early_stop_ppl:
                    print(f"  ✅ Early stop! PPL {val_ppl:.2f} < {early_stop_ppl}")
                    break

            # Checkpoint
            if global_step % save_every == 0 and local_rank == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"checkpoint-{global_step}")
                os.makedirs(ckpt_path, exist_ok=True)
                state_dict = {}
                for name, param in model.named_parameters():
                    if param.requires_grad or any(f"layers.{i}." in name for i in simplicial_indices):
                        state_dict[name] = param.detach().cpu()
                safetensors_save(state_dict, os.path.join(ckpt_path, "model.safetensors"))
                with open(os.path.join(ckpt_path, "config.json"), "w") as f:
                    json.dump({"step": global_step, "best_val_ppl": best_val_ppl, "simplicial_indices": simplicial_indices}, f)
                print(f"  Checkpoint salvato: {ckpt_path}")

    if local_rank == 0:
        total_time = time.time() - start_time
        print(f"\nTraining completato in {total_time:.1f}s ({total_time/60:.1f}m)")
        print(f"Best Val PPL: {best_val_ppl:.2f}")
        wandb.finish()

    if is_distributed:
        dist.destroy_process_group()


def train(config: dict):
    """Chiamata da main.py."""
    _do_train(config)


def main():
    config_path = os.environ.get("CONFIG_PATH", "finetuning/config.yaml")
    _do_train(load_config(config_path))


if __name__ == "__main__":
    main()