"""
Ottimizzatore con 3 gruppi di parametri per training ibrido.

Gruppi:
1. Frozen (lr=0): tutto tranne i layer simpliciali
2. K1/V1 (lr piccolo): k1_proj, v1_proj nei layer simpliciali
3. K2/V2 (lr normale): k2_proj, v2_proj nei layer simpliciali
"""

import torch
from typing import List


def create_optimizer_groups(
    model,
    simplicial_indices: List[int],
    lr_k2v2: float = 2e-4,
    lr_k1v1: float = 2e-5,
    weight_decay: float = 0.01,
):
    """
    Crea i 3 gruppi di parametri per AdamW.

    Args:
        model: modello ibrido
        simplicial_indices: [16, 20, 24, 28]
        lr_k2v2: learning rate per K2/V2
        lr_k1v1: learning rate per K1/V1
        weight_decay: weight decay

    Returns:
        lista di dict per AdamW
    """
    frozen_params = []
    k1v1_params = []
    k2v2_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            frozen_params.append(param)
            continue

        in_simplicial = any(f"layers.{idx}." in name for idx in simplicial_indices)
        if not in_simplicial:
            param.requires_grad = False
            frozen_params.append(param)
        elif "k2_proj" in name or "v2_proj" in name:
            k2v2_params.append(param)
        elif "k1_proj" in name or "v1_proj" in name:
            k1v1_params.append(param)
        else:
            param.requires_grad = False
            frozen_params.append(param)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"[optimizer] Gruppi creati:")
    print(f"  Frozen:        {sum(p.numel() for p in frozen_params):>10,} params")
    print(f"  K1/V1 (lr={lr_k1v1}): {sum(p.numel() for p in k1v1_params):>10,} params")
    print(f"  K2/V2 (lr={lr_k2v2}): {sum(p.numel() for p in k2v2_params):>10,} params")
    print(f"  Trainable: {trainable:,}/{total:,} ({100*trainable/total:.2f}%)")

    return [
        {"params": frozen_params, "lr": 0.0, "weight_decay": 0.0},
        {"params": k1v1_params, "lr": lr_k1v1, "weight_decay": weight_decay},
        {"params": k2v2_params, "lr": lr_k2v2, "weight_decay": weight_decay},
    ]