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
    chunk_size: int = 8,
) -> float:
    """
    Calcola la loss media su un batch di validazione, processando a chunk.

    Val_batch puo' contenere centinaia di campioni (es. 500). Processarli tutti
    in un unico forward causa OOM su GPU da 48 GB (matrici S×S per ogni campione).
    
    Dividiamo in chunk di chunk_size campioni, calcoliamo la loss per ogni chunk,
    e facciamo la media pesata per numero di token.
    
    Args:
        model: modello ibrido
        val_batch: dict con input_ids, labels, attention_mask
        chunk_size: campioni per chunk (default: 8)

    Returns:
        loss media (float)
    """
    model.eval()
    
    input_ids = val_batch["input_ids"]
    labels = val_batch["labels"]
    attention_mask = val_batch.get("attention_mask")
    
    N = input_ids.shape[0]
    total_loss = 0.0
    total_tokens = 0
    
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk_inputs = input_ids[start:end]
        chunk_labels = labels[start:end]
        
        outputs = model(
            input_ids=chunk_inputs,
            labels=chunk_labels,
        )
        
        # outputs.loss e' gia' la media per token
        # Pesiamo per il numero di token nel chunk
        loss_val = outputs.loss.item()
        n_tokens = (chunk_labels != -100).sum().item() if hasattr(chunk_labels, 'sum') else chunk_labels.numel()
        
        total_loss += loss_val * n_tokens
        total_tokens += n_tokens
    
    avg_loss = total_loss / max(total_tokens, 1)
    model.train()
    return avg_loss


@torch.no_grad()
def evaluate_validation(
    model,
    val_batch: Dict[str, torch.Tensor],
    simplicial_indices: List[int],
    attention_type: str = "simplicial",
) -> Dict[str, float]:
    """
    Valutazione completa: loss + perplexity + distanza K1/K2.

    Args:
        model: modello ibrido
        val_batch: batch di validazione
        simplicial_indices: [16, 20, 24, 28]
        attention_type: "simplicial" o "gram_det" (per chunk_size)

    Returns:
        dict con metriche
    """
    # GramDet materializza [B, H, N, P, d] → usa chunk piu' piccoli per evitare OOM
    chunk_size = 2 if attention_type == "gram_det" else 8
    loss = evaluate_loss(model, val_batch, chunk_size=chunk_size)
    ppl = compute_perplexity(loss)

    metrics = {
        "val/loss": loss,
        "val/perplexity": ppl,
    }

    distances = compute_k1k2_distances(model, simplicial_indices)
    for k, v in distances.items():
        metrics[f"val/{k}"] = v

    return metrics
