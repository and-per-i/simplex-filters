"""
Metriche per monitoring durante il finetuning.

Fornisce:
- Perplexity (PPL = exp(loss))
- Distanza L2 media tra k1_proj.weight e k2_proj.weight
- Distanza L2 media tra v1_proj.weight e v2_proj.weight
"""

import math
import torch
import torch.nn.functional as F
from typing import List, Dict


def compute_perplexity(loss: float) -> float:
    """Calcola perplexity da una loss media."""
    return math.exp(loss)


def compute_k1k2_distances(
    model,
    simplicial_indices: List[int],
) -> Dict[str, float]:
    """
    Calcola la distanza L2 media tra K2/K1 e V2/V1 per ogni layer simpliciale.

    Args:
        model: modello ibrido
        simplicial_indices: [16, 20, 24, 28]

    Returns:
        dict con l2_k1k2_mean, l2_v1v2_mean, e metriche per layer
    """
    total_l2_k = 0.0
    total_l2_v = 0.0
    layer_metrics = {}

    for idx in simplicial_indices:
        attn = model.model.layers[idx].self_attn

        # Verifica che abbia k1_proj e k2_proj
        if not hasattr(attn, 'k1_proj') or not hasattr(attn, 'k2_proj'):
            continue

        with torch.no_grad():
            diff_k = (attn.k2_proj.weight - attn.k1_proj.weight).norm(p=2, dim=-1).mean().item()
            diff_v = (attn.v2_proj.weight - attn.v1_proj.weight).norm(p=2, dim=-1).mean().item()

        total_l2_k += diff_k
        total_l2_v += diff_v
        layer_metrics[f"l2_k1k2_layer_{idx}"] = diff_k
        layer_metrics[f"l2_v1v2_layer_{idx}"] = diff_v

    num_layers = len(simplicial_indices)
    results = {
        "l2_k1k2_mean": total_l2_k / num_layers if num_layers > 0 else 0.0,
        "l2_v1v2_mean": total_l2_v / num_layers if num_layers > 0 else 0.0,
        **layer_metrics,
    }
    return results


@torch.no_grad()
def evaluate_loss(
    model,
    val_batch: Dict[str, torch.Tensor],
) -> float:
    """
    Calcola la loss media su un batch di validazione.

    Args:
        model: modello ibrido
        val_batch: dict con input_ids, labels, attention_mask

    Returns:
        loss media (float)
    """
    model.eval()
    outputs = model(
        input_ids=val_batch["input_ids"],
        labels=val_batch["labels"],
    )
    loss = outputs.loss.item()
    model.train()
    return loss


@torch.no_grad()
def evaluate_validation(
    model,
    val_batch: Dict[str, torch.Tensor],
    simplicial_indices: List[int],
) -> Dict[str, float]:
    """
    Valutazione completa: loss + perplexity + distanza K1/K2.

    Args:
        model: modello ibrido
        val_batch: batch di validazione
        simplicial_indices: [16, 20, 24, 28]

    Returns:
        dict con metriche
    """
    loss = evaluate_loss(model, val_batch)
    ppl = compute_perplexity(loss)

    metrics = {
        "val/loss": loss,
        "val/perplexity": ppl,
    }

    distances = compute_k1k2_distances(model, simplicial_indices)
    for k, v in distances.items():
        metrics[f"val/{k}"] = v

    return metrics