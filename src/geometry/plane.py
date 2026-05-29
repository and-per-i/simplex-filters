"""
plane.py — Costruzione e manipolazione di piani sulla Grassmanniana Gr(2,d).

Un piano P ⊂ R^d e' definito da due vettori (k1, k2) linearmente indipendenti.
La rappresentazione canonica e' il proiettore ortogonale P = U U^T,
dove U ∈ R^{d×2} e' la base ortonormale ottenuta via SVD.

Formule:
- A = [k1 | k2] ∈ R^{d×2}
- SVD(A) = U Σ V^T  →  U = prime 2 colonne sinistre
- Proiettore: P = U U^T ∈ R^{d×d} (simmetrico, idempotente, rango 2)
"""

import torch
import torch.nn.functional as F


def plane_projector(k1: torch.Tensor, k2: torch.Tensor) -> torch.Tensor:
    """
    Costruisce il proiettore ortogonale sul piano span(k1, k2).

    Args:
        k1: vettore [d] o batch [B, d]
        k2: vettore [d] o batch [B, d]

    Returns:
        Proiettore P = U U^T, [d, d] o [B, d, d]
    """
    # Gestisci batch o singolo
    if k1.dim() == 1:
        k1 = k1.unsqueeze(0)
        k2 = k2.unsqueeze(0)
        batched = False
    else:
        batched = True

    # Impila: A = [k1, k2]  →  [B, d, 2]
    A = torch.stack([k1, k2], dim=-1)  # [B, d, 2]

    # SVD
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    # U: [B, d, 2] — base ortonormale del piano
    # S: [B, 2] — valori singolari
    # Vh: [B, 2, 2] — rotazioni

    # Proiettore: P = U U^T  →  [B, d, d]
    P = U @ U.transpose(-2, -1)

    if not batched:
        P = P.squeeze(0)

    return P


def plane_projector_and_basis(k1: torch.Tensor, k2: torch.Tensor):
    """
    Come plane_projector, ma restituisce anche la base ortonormale U.

    Returns:
        P: proiettore [d, d] o [B, d, d]
        U: base ortonormale [d, 2] o [B, d, 2]
        S: valori singolari [2] o [B, 2]
    """
    if k1.dim() == 1:
        k1 = k1.unsqueeze(0)
        k2 = k2.unsqueeze(0)
        batched = False
    else:
        batched = True

    A = torch.stack([k1, k2], dim=-1)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    P = U @ U.transpose(-2, -1)

    if not batched:
        U = U.squeeze(0)
        S = S.squeeze(0)
        P = P.squeeze(0)

    return P, U, S


def principal_angles(U1: torch.Tensor, U2: torch.Tensor) -> torch.Tensor:
    """
    Calcola gli angoli principali tra due piani con basi U1, U2 ∈ R^{d×2}.

    Formula:
        Σ = U1^T U2 ∈ R^{2×2}
        σ_i = svdvals(Σ)  ∈ [0, 1]
        θ_i = arccos(clamp(σ_i, 0, 1))

    Args:
        U1: base ortonormale [d, 2] o [B, d, 2]
        U2: base ortonormale [d, 2] o [B, d, 2]

    Returns:
        angoli principali [2] o [B, 2] in radianti
    """
    # Matrice di correlazione
    Sigma = U1.transpose(-2, -1) @ U2  # [B, 2, 2] o [2, 2]

    # SVD
    _, s, _ = torch.linalg.svd(Sigma, full_matrices=False)

    # Clamp e arccos
    s = torch.clamp(s, -1.0, 1.0)
    angles = torch.acos(s)  # [2]

    return angles


def geodesic_distance(U1: torch.Tensor, U2: torch.Tensor) -> torch.Tensor:
    """
    Distanza geodesica tra due piani sulla Grassmanniana.

    d_g = sqrt(θ1^2 + θ2^2)

    Args:
        U1: base [d, 2] o [B, d, 2]
        U2: base [d, 2] o [B, d, 2]

    Returns:
        distanza scalare o [B]
    """
    angles = principal_angles(U1, U2)
    dist = torch.sqrt((angles ** 2).sum(dim=-1))
    return dist


def project_on_plane(P: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Proietta il vettore v sul piano definito dal proiettore P.

    v_proj = P @ v

    Args:
        P: proiettore [d, d] o [B, d, d]
        v: vettore [d] o [B, d]

    Returns:
        vettore proiettato [d] o [B, d]
    """
    if P.dim() == 2:
        return P @ v
    return torch.bmm(P, v.unsqueeze(-1)).squeeze(-1)


def projection_norm(P: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Norma della proiezione di v sul piano P.
    Se prossima a 0, v e' ortogonale al piano → volume massimizzato.

    ||P v||_2

    Args:
        P: proiettore [d, d] o [B, d, d]
        v: vettore [d] o [B, d]

    Returns:
        norma scalare o [B]
    """
    proj = project_on_plane(P, v)
    return torch.norm(proj, dim=-1)