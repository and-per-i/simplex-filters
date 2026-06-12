"""
Ottimizzatore con 4 gruppi di parametri per training ibrido.

Gruppi:
1. Frozen (lr=0): embedding, lm_head, e tutto tranne i layer simpliciali e standard
2. Standard (lr=lr_standard): parametri dei layer non-simpliciali (incluse FFN, norm)
3. K1/V1 (lr piccolo): k1_proj, v1_proj nei layer simpliciali
4. K2/V2 (lr normale): k2_proj, v2_proj nei layer simpliciali

Per GramDet:
1. Frozen (lr=0): embedding, lm_head, tutto tranne simpliciali
2. Standard (lr=lr_standard): parametri dei layer non-simpliciali
3. GramDet (lr=lr_k2v2): q/k/v/o dei layer simpliciali
"""

import torch
from typing import List


def create_optimizer_groups(
    model,
    simplicial_indices: List[int],
    lr_k2v2: float = 2e-4,
    lr_k1v1: float = 2e-5,
    lr_standard: float = 1e-5,
    weight_decay: float = 0.01,
    attention_type: str = "simplicial",
):
    """
    Crea i gruppi di parametri per AdamW.

    Per "simplicial": K2/V2 con LR alto, K1/V1 con LR basso, standard con LR medio, frozen.
    Per "gram_det": singolo gruppo GramDet, standard, frozen.

    Logica esatta per ogni parametro (in ordine):
    a) not requires_grad → frozen
    b) "embed" in name or "lm_head" in name → requires_grad=False, frozen
    c) simplicial layer + k1v1 → k1v1_params
    d) simplicial layer + k2v2 → k2v2_params
    e) simplicial layer + gram_det → gram_det_params
    f) simplicial layer + non-attention → frozen (norm, ecc.)
    g) altrimenti → standard_params

    Args:
        model: modello ibrido
        simplicial_indices: [16, 20, 24, 28]
        lr_k2v2: learning rate per K2/V2 (simplicial) o GramDet
        lr_k1v1: learning rate per K1/V1 (simplicial)
        lr_standard: learning rate per layer non-simpliciali (standard)
        weight_decay: weight decay
        attention_type: "simplicial" o "gram_det"

    Returns:
        lista di dict per AdamW
    """
    frozen_params = []
    standard_params = []
    k1v1_params = []
    k2v2_params = []
    gram_det_params = []

    for name, param in model.named_parameters():
        # a) Parametri già frozen
        if not param.requires_grad:
            frozen_params.append(param)
            continue

        # b) Embedding e lm_head → freeze esplicitamente
        if "embed" in name or "lm_head" in name:
            param.requires_grad = False
            frozen_params.append(param)
            continue

        in_simplicial = any(f"layers.{idx}." in name for idx in simplicial_indices)

        if not in_simplicial:
            # g) Layer non-simpliciale → standard (include FFN, norm, ecc.)
            standard_params.append(param)
        elif attention_type == "gram_det":
            # GramDet: solo q/k/v/o sono trainable
            if "q_proj" in name or "k_proj" in name or "v_proj" in name or "o_proj" in name:
                gram_det_params.append(param)
            else:
                # Norm, ecc. nei layer simpliciali → frozen
                param.requires_grad = False
                frozen_params.append(param)
        elif "k2_proj" in name or "v2_proj" in name:
            # d) Simplicial + K2/V2
            k2v2_params.append(param)
        elif "k1_proj" in name or "v1_proj" in name:
            # c) Simplicial + K1/V1
            k1v1_params.append(param)
        else:
            # f) Simplicial ma non attenzione (norm, ecc.) → frozen
            param.requires_grad = False
            frozen_params.append(param)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if attention_type == "gram_det":
        gram_det_count = sum(p.numel() for p in gram_det_params)
        standard_count = sum(p.numel() for p in standard_params)
        print(f"[optimizer] Gruppi creati (GramDet):")
        print(f"  Frozen:   {sum(p.numel() for p in frozen_params):>10,} params")
        print(f"  Standard: {standard_count:>10,} params (lr={lr_standard})")
        print(f"  GramDet:  {gram_det_count:>10,} params (lr={lr_k2v2})")
        print(f"  Trainable: {trainable:,}/{total:,} ({100*trainable/total:.2f}%)")
        return [
            {"params": frozen_params, "lr": 0.0, "weight_decay": 0.0},
            {"params": standard_params, "lr": lr_standard, "weight_decay": weight_decay},
            {"params": gram_det_params, "lr": lr_k2v2, "weight_decay": weight_decay},
        ]
    else:
        print(f"[optimizer] Gruppi creati:")
        print(f"  Frozen:        {sum(p.numel() for p in frozen_params):>10,} params")
        print(f"  Standard:      {sum(p.numel() for p in standard_params):>10,} params (lr={lr_standard})")
        print(f"  K1/V1 (lr={lr_k1v1}): {sum(p.numel() for p in k1v1_params):>10,} params")
        print(f"  K2/V2 (lr={lr_k2v2}): {sum(p.numel() for p in k2v2_params):>10,} params")
        print(f"  Trainable: {trainable:,}/{total:,} ({100*trainable/total:.2f}%)")
        return [
            {"params": frozen_params, "lr": 0.0, "weight_decay": 0.0},
            {"params": standard_params, "lr": lr_standard, "weight_decay": weight_decay},
            {"params": k1v1_params, "lr": lr_k1v1, "weight_decay": weight_decay},
            {"params": k2v2_params, "lr": lr_k2v2, "weight_decay": weight_decay},
        ]