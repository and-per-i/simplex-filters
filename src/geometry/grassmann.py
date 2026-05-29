"""
grassmann.py — Statistica sulla Grassmanniana Gr(2,d).

Fornisce:
- Media di Frechet iterativa per piani (proiettori)
- Varianza geodesica
- Query mean (via SVD su ipersfera)
- Relazione query-piano (proiezione)

Formule (Media di Frechet):
    M^{0} = (1/N) Σ_i P_i                    # media euclidea
    M^{0} = (M^{0} + (M^{0})^T) / 2          # simmetrizza
    U, Σ, V = SVD(M^{0})                     # mantieni top-2
    P_mean = U_{:,:2} @ U_{:,:2}^T           # proietta su Gr(2,d)
    Ripeti 5-10 iterazioni per convergenza

Formule (distanza geodesica):
    Σ = U_i^T U_mean ∈ R^{2×2}
    σ_1, σ_2 = svdvals(Σ)
    θ_j = arccos(clamp(σ_j, 0, 1))
    d_g = sqrt(θ_1^2 + θ_2^2)
"""

import torch


def frechet_mean_planes(
    U_list: torch.Tensor,
    n_iter: int = 10,
    tol: float = 1e-8,
) -> torch.Tensor:
    """
    Calcola la media di Frechet sulla Grassmanniana Gr(2,d).
    
    Dati N piani rappresentati dalle loro basi ortonormali U_i ∈ R^{d×2},
    trova il piano medio che minimizza Σ_i d_g(P_i, P_mean)^2.

    Args:
        U_list: basi ortonormali [N, d, 2]
        n_iter: numero di iterazioni (default: 10)
        tol:  tolleranza per early stopping

    Returns:
        U_mean: base ortonormale del piano medio [d, 2]
        P_mean: proiettore  [d, d]
    """
    N, d, k = U_list.shape  # k=2

    # Inizializzazione: media euclidea dei proiettori
    # P_i = U_i @ U_i^T
    P_sum = torch.zeros(d, d, device=U_list.device, dtype=U_list.dtype)
    for i in range(N):
        P_sum += U_list[i] @ U_list[i].T
    P_mean = P_sum / N
    
    # ---- Ciclo iterativo (Frechet mean via proiezione) ----
    for iteration in range(n_iter):
        P_old = P_mean.clone()
        
        # 1. Simmetrizza
        P_mean = (P_mean + P_mean.T) / 2.0
        
        # 2. SVD e proiezione su Gr(2,d): top-2 autovettori
        U_mean, S, Vh = torch.linalg.svd(P_mean, full_matrices=False)
        U_mean = U_mean[:, :2]  # [d, 2]
        P_mean = U_mean @ U_mean.T  # [d, d]
        
        # Convergenza
        diff = (P_mean - P_old).norm().item()
        if diff < tol:
            break

    return U_mean, P_mean


def geodesic_variance(
    U_list: torch.Tensor,
    U_mean: torch.Tensor,
) -> torch.Tensor:
    """
    Calcola la varianza geodesica: media delle distanze geodesiche al quadrato
    tra ogni piano e il piano medio.

    Var = (1/N) Σ_i d_g(U_i, U_mean)^2

    Args:
        U_list: basi [N, d, 2]
        U_mean: base media [d, 2]

    Returns:
        varianza: scalare
        distances: [N] distanze individuali
    """
    from src.geometry.plane import geodesic_distance, principal_angles
    
    N = U_list.shape[0]
    distances = torch.zeros(N, device=U_list.device, dtype=U_list.dtype)
    
    for i in range(N):
        distances[i] = geodesic_distance(U_list[i], U_mean)
    
    variance = (distances ** 2).mean()
    return variance, distances


def frechet_mean_queries(
    Q_list: torch.Tensor,
    n_iter: int = 10,
    tol: float = 1e-8,
) -> torch.Tensor:
    """
    Calcola la media di Frechet sull'ipersfera S^{d-1} (vettori unitari).
    
    Dati N vettori normalizzati q̄_i = q_i / ||q_i||, trova il vettore medio.
    
    Formula:
        q_mean = top-1 autovettore di (1/N) Σ_i q̄_i q̄_i^T
    
    (Equivalente alla prima componente principale della matrice
     di Gram delle query normalizzate)

    Args:
        Q_list: vettori query [N, d] (non normalizzati, la funzione normalizza)
        n_iter: iterazioni (non usato, SVD chiusa)
        tol: tolleranza

    Returns:
        q_mean: vettore medio normalizzato [d]
    """
    # Normalizza
    Q_norm = Q_list / torch.norm(Q_list, dim=-1, keepdim=True)
    
    # Matrice di Gram (o scatter matrix)
    # M = (1/N) Σ_i q_i q_i^T
    M = (Q_norm.T @ Q_norm) / Q_norm.shape[0]
    
    # SVD → top-1 autovettore = query media
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    q_mean = U[:, 0]
    
    return q_mean


def query_plane_relation(
    q: torch.Tensor,
    U_plane: torch.Tensor,
) -> torch.Tensor:
    """
    Calcola la relazione tra un vettore query e un piano.
    
    proj_norm = ||P q||_2   dove P = U U^T
    Se proj_norm ≈ 0, q e' ortogonale al piano → volume massimizzato.
    
    Il volume del parallelepipedo e':
        vol = det(Gram(q, k1, k2)) = ||q|| · ||k1|| · ||k2|| · sin(θ)
    dove θ e' l'angolo tra q e il piano.

    Args:
        q: vettore query [d] o [B, d]
        U_plane: base del piano [d, 2]

    Returns:
        proj_norm: norma della proiezione (scalare o [B])
        angle: angolo tra q e il piano (radianti) = arcsin(proj_norm / ||q||)
    """
    from src.geometry.plane import projection_norm
    
    if q.dim() == 1:
        q = q.unsqueeze(0)
        batched = True
    else:
        batched = False
    
    # Norma proiezione
    P = U_plane @ U_plane.T  # [d, d]
    proj_norm = projection_norm(P.unsqueeze(0), q)  # [B]
    
    # Angolo: sin(θ) = ||proj|| / ||q||
    q_norm = torch.norm(q, dim=-1)
    # Evita divisione per zero
    q_norm = torch.clamp(q_norm, min=1e-10)
    sin_theta = proj_norm / q_norm
    sin_theta = torch.clamp(sin_theta, 0.0, 1.0)
    angle = torch.asin(sin_theta)
    
    if not batched:
        proj_norm = proj_norm.squeeze(0)
        angle = angle.squeeze(0)
    
    return proj_norm, angle