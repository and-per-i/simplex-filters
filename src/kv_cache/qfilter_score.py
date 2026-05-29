"""
qfilter_score.py — Calcolo dello score di eviction Q-filter.

Formula:
    Score(k_j) = sqrt(σ₁² · ⟨k_j, e₂⟩² + σ₂² · ⟨k_j, e₁⟩²)

Dove:
    σ₁, σ₂ = valori singolari della distribuzione query nel piano medio (anisotropia)
    e₁, e₂ = vettori della base del piano medio (colonne di U_mean ∈ R^{d×2})
    
Interpretazione:
    Lo score pesa la proiezione della chiave k_j sul piano medio usando i
    valori singolari delle query. Piu' lo score e' alto, piu' quella chiave
    contribuisce al volume del parallelepipedo (q, k1, k2).
"""

import torch


def qfilter_score(
    k: torch.Tensor,
    sigma1: float,
    sigma2: float,
    U_mean: torch.Tensor,
) -> torch.Tensor:
    """
    Calcola il Q-filter score per ogni chiave.

    Args:
        k: chiavi [N, d] (batch di vettori d-dimensionali)
        sigma1: σ₁ dall'analisi geometrica (scalar)
        sigma2: σ₂ dall'analisi geometrica (scalar)
        U_mean: base del piano medio [d, 2] (colonne e₁, e₂)

    Returns:
        scores: [N] score per ogni chiave
    """
    # Proietta k sul piano medio: [N, 2] = k @ U_mean
    k_proj = k @ U_mean  # [N, 2]

    # Separa le componenti lungo e₁ e e₂
    k_e1 = k_proj[:, 0]  # [N]
    k_e2 = k_proj[:, 1]  # [N]

    # Score pesato: sqrt(σ₁² · k_e2² + σ₂² · k_e1²)
    scores = torch.sqrt(sigma1**2 * k_e2**2 + sigma2**2 * k_e1**2)

    return scores


def qfilter_score_single(
    k: torch.Tensor,
    sigma1: float,
    sigma2: float,
    e1: torch.Tensor,
    e2: torch.Tensor,
) -> torch.Tensor:
    """
    Versione per singola chiave con vettori espliciti.

    Args:
        k: chiave [d]
        sigma1, sigma2: valori singolari
        e1, e2: vettori base del piano [d]

    Returns:
        score: scalar
    """
    k_e1 = (k * e1).sum()
    k_e2 = (k * e2).sum()
    score = torch.sqrt(sigma1**2 * k_e2**2 + sigma2**2 * k_e1**2)
    return score


def top_k_indices(
    scores: torch.Tensor,
    budget: float,
) -> torch.Tensor:
    """
    Seleziona gli indici delle top-B chiavi per budget B.

    Args:
        scores: [N] score per ogni chiave
        budget: frazione da tenere (0.5 = 50%, 0.3 = 30%, 0.1 = 10%)

    Returns:
        indices: [B] indici delle chiavi selezionate, ordinate per score decrescente
    """
    N = scores.shape[0]
    B = max(1, int(N * budget))

    # Ordina per score decrescente
    indices = torch.argsort(scores, descending=True)[:B]
    return indices


def random_indices(
    N: int,
    budget: float,
) -> torch.Tensor:
    """
    Seleziona indici casuali come baseline (random eviction).

    Args:
        N: numero totale di chiavi
        budget: frazione da tenere

    Returns:
        indices: [B] indici casuali
    """
    B = max(1, int(N * budget))
    indices = torch.randperm(N)[:B]
    return indices