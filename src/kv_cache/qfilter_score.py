"""
qfilter_score.py — Calcolo dello score di eviction Q-filter.

FORMULA PRIMARIA (raccomandata):
    Score(k_j) = ‖k_j - P̄ k_j‖  (componente ortogonale al piano medio)
    Usata da: qfilter_score_orthogonal()
    Validazione: Spearman ρ=+0.61 (GramDet), ρ=+0.50 (trilineare)

    Premia le chiavi ATIPICHE (ortogonali al piano medio). Cio' significa
    che le chiavi piu' informative sono quelle che si discostano dal piano
    medio — coerentemente con l'idea che il piano medio catturi il
    comportamento "tipico" e le chiavi atipiche portino informazione nuova.

FORMULA SECONDARIA (DEPRECATA, solo per confronto):
    Score(k_j) = sqrt(σ₁² · ⟨k_j, e₂⟩² + σ₂² · ⟨k_j, e₁⟩²)
    Usata da: qfilter_score()
    Validazione: Spearman ρ=-0.27 — ANTI-CORRELATA, NON USARE.

    Questa formula premia le chiavi vicine al piano medio (tipiche), ma
    la validazione empirica mostra che e' anti-correlata al vero ordine
    di importanza. Mantenuta solo per riproducibilita' dei risultati.

Dove:
    σ₁, σ₂ = valori singolari della distribuzione query nel piano medio (anisotropia)
    e₁, e₂ = vettori della base del piano medio (colonne di U_mean ∈ R^{d×2})
    P̄ = proiettore ortogonale sul piano medio (U_mean @ U_mean^T)

Riferimento:
    validate_proxy.py — script di validazione empirica che ha prodotto r=+0.61.
"""

import torch


def qfilter_score(
    k: torch.Tensor,
    sigma1: float,
    sigma2: float,
    U_mean: torch.Tensor,
) -> torch.Tensor:
    """
    Q-filter score per attenzione TRILINEARE (geometria per-token).
    
    Pesa la proiezione della chiave sul piano medio usando i valori singolari
    delle query. Score alto = chiave vicina al piano medio (tipica).
    Formula: sqrt(σ₁² · ⟨k, e₂⟩² + σ₂² · ⟨k, e₁⟩²)

    Args:
        k: chiavi [N, d]
        sigma1: σ₁ dall'analisi geometrica
        sigma2: σ₂ dall'analisi geometrica
        U_mean: base del piano medio [d, 2]

    Returns:
        scores: [N] score per ogni chiave
    """
    k_proj = k @ U_mean  # [N, 2]
    k_e1 = k_proj[:, 0]  # [N]
    k_e2 = k_proj[:, 1]  # [N]
    scores = torch.sqrt(sigma1**2 * k_e2**2 + sigma2**2 * k_e1**2)
    return scores


def qfilter_score_orthogonal(
    k: torch.Tensor,
    U_mean: torch.Tensor,
) -> torch.Tensor:
    """
    Q-filter score per attenzione GRAMDET (geometria per-coppia).
    
    Calcola la componente ortogonale di ogni chiave al piano medio.
    Score alto = chiave atipica (lontana dal piano tipico).
    Formula: ‖k - P̄ k‖ dove P̄ = U_mean @ U_mean^T

    Validato empiricamente: correlazione Spearman +0.61 vs ground truth
    (vs -0.27 della formula classica di proiezione).

    Args:
        k: chiavi [N, d]
        U_mean: base del piano medio [d, 2]

    Returns:
        scores: [N] score per ogni chiave
    """
    # Proiettore sul piano medio
    P = U_mean @ U_mean.T  # [d, d]
    # Componente sul piano: k_proj = k @ P
    k_proj = k @ P  # [N, d]
    # Componente ortogonale: k_orth = k - k_proj
    k_orth = k - k_proj  # [N, d]
    # Score = norma della componente ortogonale
    scores = torch.norm(k_orth, dim=-1)  # [N]
    return scores


def qfilter_score_single(
    k: torch.Tensor,
    sigma1: float,
    sigma2: float,
    e1: torch.Tensor,
    e2: torch.Tensor,
) -> torch.Tensor:
    """
    Versione per singola chiave con vettori espliciti (TRILINEARE).

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
