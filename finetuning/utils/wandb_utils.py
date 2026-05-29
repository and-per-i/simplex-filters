"""
Weights & Biases utilities per logging durante il finetuning.

Setup automatico: se wandb non è installato o non c'e' login, logging silenzioso.
"""

import os
import math
from typing import Dict, Optional


def init_wandb(config: dict) -> bool:
    """
    Inizializza una run Weights & Biases.

    Args:
        config: dict con configurazione del training
            (almeno wandb_project, wandb_run_name)

    Returns:
        True se WandB e' attivo, False altrimenti
    """
    try:
        import wandb
    except ImportError:
        print("[wandb] non installato. Logging su stdout.")
        return False

    project = config.get("wandb_project", "simplex-filters")
    run_name = config.get("wandb_run_name", "llama-simplicial-finetune")

    try:
        wandb.init(
            project=project,
            name=run_name,
            config=config,
            settings=wandb.Settings(init_timeout=30),
        )
        print(f"[wandb] Run '{run_name}' inizializzata su progetto '{project}'")
        return True
    except Exception as e:
        print(f"[wandb] Init fallito: {e}. Logging su stdout.")
        return False


def log_metrics(metrics: Dict[str, float], step: int, wandb_active: bool):
    """
    Logga metriche su WandB e/o stdout.

    Args:
        metrics: dict nome -> valore
        step: step corrente
        wandb_active: True se WandB e' attivo
    """
    if wandb_active:
        import wandb
        wandb.log(metrics, step=step)

    # Log su stdout ogni volta (WandB potrebbe fallire silenziosamente)
    metric_str = " | ".join(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items())
    print(f"step {step:5d} | {metric_str}")


def finish_wandb(wandb_active: bool):
    """Chiude la run WandB se attiva."""
    if wandb_active:
        import wandb
        wandb.finish()
        print("[wandb] Run completata.")