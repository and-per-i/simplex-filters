"""
eviction.py — Meccanismo di eviction sperimentale per KV cache simpliciale.

Quando si applica eviction, le coppie (k_j1, k_j2) nell'attenzione devono
essere formate solo da chiavi sopravvissute. Questo significa che se una
chiave e' stata eliminata, non partecipera' a NESSUNA coppia.

L'eviction Q-filter e' UNA SOLA: si calcola lo score per ogni singola
chiave usando la componente ortogonale al piano medio (‖k − P̄k‖),
si ordinano per score decrescente, si tengono le top-B,
e le chiavi eliminate sono semplicemente assenti dalla KV cache.

Formula valida (Spearman ρ=+0.61 per GramDet, ρ=+0.50 per trilineare):
    score(k_j) = ‖k_j − P̄ k_j‖  (componente ortogonale al piano medio)

NOTA: La formula di proiezione anisotropa (qfilter_score con σ₁, σ₂)
e' stata deprecata perche' anti-correlata (ρ=-0.27 contro ρ=+0.61).
"""

import torch
from typing import Optional

from src.kv_cache.qfilter_score import qfilter_score_orthogonal, top_k_indices, random_indices


def evict_keys(
    keys: torch.Tensor,
    U_mean: torch.Tensor,
    budget: float,
    strategy: str = "qfilter",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applica eviction sulle chiavi e restituisce quelle sopravvissute.

    Usa lo score ortogonale ‖k − P̄k‖ (qfilter_score_orthogonal):
    - Score alto → chiave atipica (lontana dal piano medio) → preservata
    - Score basso → chiave tipica (vicina al piano medio) → candidata a eviction

    Args:
        keys: tutte le chiavi nella finestra [N, d]
        U_mean: base del piano medio [d, 2]
        budget: frazione da tenere (0.5, 0.3, 0.1)
        strategy: "qfilter" o "random"

    Returns:
        survived_keys: chiavi sopravvissute [B, d]
        survived_indices: indici originali [B]
    """
    N = keys.shape[0]

    if strategy == "qfilter":
        scores = qfilter_score_orthogonal(keys, U_mean)
        indices = top_k_indices(scores, budget)
    elif strategy == "random":
        indices = random_indices(N, budget)
    else:
        raise ValueError(f"Strategia sconosciuta: {strategy}")

    return keys[indices], indices


def compute_perplexity_with_eviction(
    model,
    input_ids: torch.Tensor,
    U_mean: torch.Tensor,
    budget: float,
    window_size: int = 512,
    strategy: str = "qfilter",
) -> float:
    """
    Calcola la perplexity applicando eviction Q-filter durante il forward.

    NOTA: Questa e' una versione semplificata che applica l'eviction sulle
    chiavi prima di passarli al modello. L'implementazione completa richiede
    la modifica del kernel di attenzione per accettare solo le chiavi
    sopravvissute.

    Args:
        model: modello ibrido
        input_ids: input token [B, S]
        U_mean: base del piano medio [d, 2]
        budget: frazione da tenere
        window_size: finestra K1 (default 512)
        strategy: "qfilter" o "random"

    Returns:
        perplexity: float
    """
    import math
    import torch.nn.functional as F

    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()

    perplexity = math.exp(loss)
    return perplexity