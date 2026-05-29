"""
Test per il pacchetto geometry (Grassmanniana, piani, media di Frechet).

Verifica su dati sintetici:
1. Costruzione piano via SVD (ortonormalita', idempotenza)
2. Angoli principali (correttezza per casi noti)
3. Distanza geodesica (non-negativa, simmetria, disuguaglianza triangolare)
4. Media di Frechet (convergenza, caso noto)
5. Varianza geodesica
6. Query media su ipersfera
7. Relazione query-piano (ortogonalita' → volume = 0)
"""

import pytest
import torch
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from geometry.plane import (
    plane_projector,
    plane_projector_and_basis,
    principal_angles,
    geodesic_distance,
    project_on_plane,
    projection_norm,
)
from geometry.grassmann import (
    frechet_mean_planes,
    geodesic_variance,
    frechet_mean_queries,
    q_filters_query_mean,
    query_plane_relation,
)
from geometry.analyzer import analyze_query_distribution


# ======================================================================
# Test: Costruzione piano via SVD
# ======================================================================

class TestPlaneProjector:
    """Verifica che il proiettore sul piano sia costruito correttamente."""

    def test_plane_projector_shape(self):
        """Proiettore per due vettori d-dimensionali → [d, d]."""
        k1 = torch.randn(64)
        k2 = torch.randn(64)
        P = plane_projector(k1, k2)
        assert P.shape == (64, 64)

    def test_plane_projector_symmetric(self):
        """Proiettore simmetrico: P = P^T."""
        k1 = torch.randn(64)
        k2 = torch.randn(64)
        P = plane_projector(k1, k2)
        diff = (P - P.T).abs().max().item()
        assert diff < 1e-5, f"Proiettore non simmetrico: diff={diff}"

    def test_plane_projector_idempotent(self):
        """Proiettore idempotente: P @ P = P (rango 2)."""
        k1 = torch.randn(64)
        k2 = torch.randn(64)
        P = plane_projector(k1, k2)
        PP = P @ P
        diff = (P - PP).abs().max().item()
        assert diff < 1e-5, f"Proiettore non idempotente: diff={diff}"

    def test_plane_projector_rank2(self):
        """Proiettore ha rango 2."""
        k1 = torch.randn(64)
        k2 = torch.randn(64)
        P = plane_projector(k1, k2)
        U, S, Vh = torch.linalg.svd(P)
        rank = (S > 1e-5).sum().item()
        assert rank == 2, f"Proiettore ha rango {rank}, atteso 2"

    def test_plane_projector_trace2(self):
        """Traccia del proiettore = 2 (dimensione del piano)."""
        k1 = torch.randn(64)
        k2 = torch.randn(64)
        P = plane_projector(k1, k2)
        tr = P.trace().item()
        assert abs(tr - 2) < 1e-5, f"Traccia = {tr}, attesa 2"

    def test_plane_projector_batch(self):
        """Batch: [B, d] → [B, d, d]."""
        k1 = torch.randn(8, 64)
        k2 = torch.randn(8, 64)
        P = plane_projector(k1, k2)
        assert P.shape == (8, 64, 64)

    def test_project_on_plane(self):
        """Proiettare k1 sul piano span(k1, k2) restituisce k1 stesso."""
        k1 = torch.randn(64)
        k2 = torch.randn(64)
        P = plane_projector(k1, k2)
        proj = project_on_plane(P, k1)
        diff = (proj - k1).abs().max().item()
        assert diff < 1e-5, f"Proiezione di k1 non restituisce k1: diff={diff}"

    def test_project_orthogonal_vector(self):
        """Vettore ortogonale al piano ha proiezione ~0."""
        k1 = torch.randn(64)
        k2 = torch.randn(64)
        P = plane_projector(k1, k2)
        
        # Crea vettore ortogonale al piano
        # P e' il proiettore, quindi (I - P) e' il proiettore ortogonale
        I = torch.eye(64)
        v_orth = (I - P) @ torch.randn(64)  # vettore ortogonale al piano
        v_orth = v_orth / v_orth.norm()
        
        norm = projection_norm(P, v_orth).item()
        assert norm < 1e-5, f"Vettore ortogonale ha proiezione {norm}"


# ======================================================================
# Test: Angoli principali e distanza geodesica
# ======================================================================

class TestPrincipalAngles:
    """Verifica calcolo degli angoli principali."""

    def test_parallel_planes_zero_angle(self):
        """Piani paralleli (stesso U) → angoli = 0."""
        k1 = torch.randn(64)
        k2 = torch.randn(64)
        _, U, _ = plane_projector_and_basis(k1, k2)
        angles = principal_angles(U, U)
        assert angles.abs().max().item() < 1e-5, f"Angoli non nulli per piani paralleli: {angles}"

    def test_orthogonal_planes(self):
        """Piani ortogonali hanno angolo = pi/2."""
        # Crea base U1
        U1 = torch.zeros(64, 2)
        U1[0, 0] = 1.0
        U1[1, 1] = 1.0
        
        # Crea base U2 ortogonale (usa dimensioni 2,3)
        U2 = torch.zeros(64, 2)
        U2[2, 0] = 1.0
        U2[3, 1] = 1.0
        
        angles = principal_angles(U1, U2)
        # Almeno un angolo dovrebbe essere ~pi/2
        assert angles.max().item() > 1.5, f"Angoli per piani ortogonali: {angles}"

    def test_principal_angles_symmetry(self):
        """principal_angles(U1, U2) ≈ principal_angles(U2, U1)."""
        k1 = torch.randn(64)
        k2 = torch.randn(64)
        k3 = torch.randn(64)
        k4 = torch.randn(64)
        _, U1, _ = plane_projector_and_basis(k1, k2)
        _, U2, _ = plane_projector_and_basis(k3, k4)
        
        a12 = principal_angles(U1, U2)
        a21 = principal_angles(U2, U1)
        diff = (a12 - a21).abs().max().item()
        assert diff < 1e-5, f"Angoli non simmetrici: diff={diff}"


class TestGeodesicDistance:
    """Verifica proprieta' della distanza geodesica."""

    def test_distance_non_negative(self):
        """Distanza geodesica non negativa."""
        k1 = torch.randn(64)
        k2 = torch.randn(64)
        k3 = torch.randn(64)
        k4 = torch.randn(64)
        _, U1, _ = plane_projector_and_basis(k1, k2)
        _, U2, _ = plane_projector_and_basis(k3, k4)
        
        d = geodesic_distance(U1, U2).item()
        assert d >= 0, f"Distanza negativa: {d}"

    def test_distance_self_zero(self):
        """Distanza da se' stessi = 0."""
        k1 = torch.randn(64)
        k2 = torch.randn(64)
        _, U, _ = plane_projector_and_basis(k1, k2)
        d = geodesic_distance(U, U).item()
        assert d < 1e-5, f"Distanza da se' stesso non nulla: {d}"

    def test_distance_symmetry(self):
        """Distanza simmetrica: d(U1, U2) = d(U2, U1)."""
        k1 = torch.randn(64)
        k2 = torch.randn(64)
        k3 = torch.randn(64)
        k4 = torch.randn(64)
        _, U1, _ = plane_projector_and_basis(k1, k2)
        _, U2, _ = plane_projector_and_basis(k3, k4)
        
        d12 = geodesic_distance(U1, U2).item()
        d21 = geodesic_distance(U2, U1).item()
        assert abs(d12 - d21) < 1e-5, f"Distanza non simmetrica: {d12} vs {d21}"

    def test_triangle_inequality(self):
        """Disuguaglianza triangolare: d(U1, U3) ≤ d(U1, U2) + d(U2, U3)."""
        k1 = torch.randn(64)
        k2 = torch.randn(64)
        k3 = torch.randn(64)
        k4 = torch.randn(64)
        k5 = torch.randn(64)
        k6 = torch.randn(64)
        _, U1, _ = plane_projector_and_basis(k1, k2)
        _, U2, _ = plane_projector_and_basis(k3, k4)
        _, U3, _ = plane_projector_and_basis(k5, k6)
        
        d13 = geodesic_distance(U1, U3).item()
        d12 = geodesic_distance(U1, U2).item()
        d23 = geodesic_distance(U2, U3).item()
        
        assert d13 <= d12 + d23 + 1e-5, \
            f"Triangolo violato: {d13} > {d12} + {d23}"


# ======================================================================
# Test: Media di Frechet
# ======================================================================

class TestFrechetMeanPlanes:
    """Verifica media di Frechet sulla Grassmanniana."""

    def test_frechet_mean_converges(self):
        """Media converge per piani identici → restituisce lo stesso piano."""
        k1 = torch.randn(32)
        k2 = torch.randn(32)
        _, U_ref, _ = plane_projector_and_basis(k1, k2)
        
        # Crea 20 piani identici
        U_list = U_ref.unsqueeze(0).repeat(20, 1, 1)
        
        U_mean, P_mean = frechet_mean_planes(U_list, n_iter=10)
        
        diff = (U_mean - U_ref).abs().max().item()
        assert diff < 1e-5, f"Media di piani identici non restituisce lo stesso piano: diff={diff}"

    def test_frechet_mean_trace(self):
        """Piano medio ha traccia = 2."""
        U_list = torch.randn(20, 32, 2)
        # Ortogonalizza ogni base
        for i in range(20):
            U_list[i], _, _ = torch.linalg.svd(U_list[i], full_matrices=False)
        
        U_mean, P_mean = frechet_mean_planes(U_list, n_iter=10)
        tr = P_mean.trace().item()
        assert abs(tr - 2) < 1e-5, f"Traccia media = {tr}, attesa 2"

    def test_frechet_mean_invariant_to_basis(self):
        """Media invariante per rotazione delle basi individuali."""
        U_list = torch.randn(20, 32, 2)
        for i in range(20):
            U_list[i], _, _ = torch.linalg.svd(U_list[i], full_matrices=False)
        
        U_mean1, _ = frechet_mean_planes(U_list, n_iter=10)
        
        # Ruota ogni base di una rotazione casuale 2x2
        U_list_rot = U_list.clone()
        for i in range(20):
            R, _ = torch.linalg.qr(torch.randn(2, 2))
            U_list_rot[i] = U_list[i] @ R
        
        U_mean2, _ = frechet_mean_planes(U_list_rot, n_iter=10)
        
        # Le due medie dovrebbero essere vicine (stesso piano)
        dot = (U_mean1.T @ U_mean2).abs().sum().item()
        assert dot > 1.5, f"Medie diverse dopo rotazione basi: dot={dot}"


# ======================================================================
# Test: Varianza geodesica
# ======================================================================

class TestGeodesicVariance:
    """Verifica varianza geodesica."""

    def test_variance_zero_for_identical(self):
        """Varianza zero per piani identici."""
        k1 = torch.randn(32)
        k2 = torch.randn(32)
        _, U_ref, _ = plane_projector_and_basis(k1, k2)
        U_list = U_ref.unsqueeze(0).repeat(10, 1, 1)
        
        U_mean, _ = frechet_mean_planes(U_list)
        var, _ = geodesic_variance(U_list, U_mean)
        assert var.item() < 1e-5, f"Varianza non nulla per piani identici: {var}"


# ======================================================================
# Test: Query mean
# ======================================================================

class TestQueryMean:
    """Verifica media delle query su ipersfera."""

    def test_query_mean_identical(self):
        """Query identiche → media = stessa query."""
        q = torch.randn(64)
        q = q / q.norm()
        Q_list = q.unsqueeze(0).repeat(20, 1)
        
        q_mean = frechet_mean_queries(Q_list)
        dot = (q_mean * q).sum().item()
        assert abs(dot - 1) < 1e-5, f"Media query identiche: dot={dot}"

    def test_query_mean_unit_norm(self):
        """Media normalizzata = 1."""
        Q_list = torch.randn(50, 64)
        q_mean = frechet_mean_queries(Q_list)
        norm = q_mean.norm().item()
        assert abs(norm - 1) < 1e-5, f"Norma media = {norm}, attesa 1"


# ======================================================================
# Test: Relazione query-piano
# ======================================================================

class TestQueryPlaneRelation:
    """Verifica relazione query-piano."""

    def test_query_in_plane_max_proj(self):
        """Query nel piano → proiezione massima."""
        k1 = torch.randn(32)
        k2 = torch.randn(32)
        _, U_plane, _ = plane_projector_and_basis(k1, k2)
        
        # Query = combinazione lineare di k1, k2 (sta nel piano)
        q = 0.7 * k1 + 0.3 * k2
        
        proj, angle = query_plane_relation(q, U_plane)
        
        # ||P q|| dovrebbe essere ~||q||
        q_norm = q.norm().item()
        assert proj.item() > 0.9 * q_norm, f"Query nel piano: proj={proj}, q_norm={q_norm}"

    def test_query_orthogonal_to_plane_zero_proj(self):
        """Query ortogonale al piano → proiezione ~0."""
        k1 = torch.randn(32)
        k2 = torch.randn(32)
        _, U_plane, _ = plane_projector_and_basis(k1, k2)
        
        # Crea query ortogonale al piano
        P = U_plane @ U_plane.T
        I = torch.eye(32)
        q = (I - P) @ torch.randn(32)
        q = q / q.norm()
        
        proj, angle = query_plane_relation(q, U_plane)
        assert proj.item() < 1e-5, f"Query ortogonale: proj={proj}"

    def test_angle_orthogonal_is_pi_half(self):
        """Query ortogonale al piano → angolo = pi/2."""
        k1 = torch.randn(32)
        k2 = torch.randn(32)
        _, U_plane, _ = plane_projector_and_basis(k1, k2)
        
        P = U_plane @ U_plane.T
        I = torch.eye(32)
        q = (I - P) @ torch.randn(32)
        q = q / q.norm()
        
        _, angle = query_plane_relation(q, U_plane)
        assert abs(angle.item() - 3.14159 / 2) < 0.1, \
            f"Angolo per query ortogonale = {angle}, atteso ~pi/2"


# ======================================================================
# Test: Q-filters query mean
# ======================================================================

class TestQFiltersQueryMean:
    """Verifica che q_filters_query_mean() usi il metodo Q-filters."""

    def test_q_filters_shape(self):
        """SVD su Q [N, d] → vettore medio [d]."""
        Q = torch.randn(100, 64)
        q_mean = q_filters_query_mean(Q)
        assert q_mean.shape == (64,), f"Shape: {q_mean.shape}"

    def test_q_filters_identical_queries(self):
        """Query identiche → media coincidente."""
        q = torch.randn(64)
        Q = q.unsqueeze(0).repeat(50, 1)
        q_mean = q_filters_query_mean(Q)
        # Il vettore medio dovrebbe essere collineare con q
        dot = (q_mean * q).abs().sum().item()
        norm_q = q.norm().item()
        norm_mean = q_mean.norm().item()
        cos_sim = dot / (norm_q * norm_mean + 1e-10)
        assert cos_sim > 0.99, f"cos_sim = {cos_sim}"

    def test_q_filters_different_from_frechet(self):
        """Q-filters su raw vs Frechet su normalizzati danno risultati diversi."""
        Q = torch.randn(100, 64)
        q_mean_qf = q_filters_query_mean(Q)
        q_mean_fr = frechet_mean_queries(Q)
        # Dovrebbero essere diversi (norma diversa, direzione potenzialmente diversa)
        diff = (q_mean_qf - q_mean_fr).norm().item()
        assert diff > 1e-6, f"Q-filters e Frechet identici: diff={diff}"


# ======================================================================
# Test: analyze_query_distribution (anisotropia delle query nel piano)
# ======================================================================

class TestQueryDistribution:
    """Verifica analisi distribuzione query nel piano medio."""

    def test_distribution_isotropic(self):
        """Distribuzione uniforme → σ1 ≈ σ2 (ratio vicino a 1)."""
        U_mean = torch.eye(32)[:, :2]  # piano (e1, e2)
        q_all = torch.randn(200, 32)    # query casuali → isotropiche
        
        result = analyze_query_distribution(q_all, U_mean)
        ratio = result["query_anisotropy_ratio"]
        # Con distribuzione uniforme, ratio dovrebbe essere ≈ 1
        assert 0.5 < ratio < 2.0, f"Anisotropia per distribuzione isotropica: {ratio}"

    def test_distribution_anisotropic(self):
        """Distribuzione concentrata lungo e1 → σ1 >> σ2."""
        U_mean = torch.eye(32)[:, :2]  # piano (e1, e2)
        
        # Query tutte allineate lungo e1 (molto rumore su e2, ma dominante e1)
        e1 = U_mean[:, 0]  # [32]
        noise = torch.randn(200, 32) * 0.01
        q_all = e1.unsqueeze(0).expand(200, 32) + noise
        
        result = analyze_query_distribution(q_all, U_mean)
        ratio = result["query_anisotropy_ratio"]
        assert ratio > 3.0, f"Anisotropia non rilevata: σ1/σ2 = {ratio}"

    def test_distribution_output_keys(self):
        """Output contiene sigma1, sigma2, anisotropy_ratio."""
        U_mean = torch.eye(16)[:, :2]
        q_all = torch.randn(50, 16)
        result = analyze_query_distribution(q_all, U_mean)
        assert "query_sigma1" in result
        assert "query_sigma2" in result
        assert "query_anisotropy_ratio" in result
        assert result["query_sigma1"] >= result["query_sigma2"], \
            "sigma1 dovrebbe essere >= sigma2"

    def test_distribution_symmetric_in_plane(self):
        """Distribuzione simmetrica nel piano → σ1 e σ2 comparabili."""
        U_mean = torch.eye(32)[:, :2]
        
        # Query distribuite uniformemente in un cerchio 2D dentro R^32
        angles = torch.rand(200) * 2 * 3.14159
        radii = torch.rand(200)
        
        q_2d = torch.stack([radii * torch.cos(angles), radii * torch.sin(angles)], dim=-1)
        # Proietta nello spazio 32D
        q_all = q_2d @ U_mean.T  # [200, 32] nel piano
        
        result = analyze_query_distribution(q_all, U_mean)
        ratio = result["query_anisotropy_ratio"]
        # Distribuzione simmetrica → ratio ≈ 1
        assert 0.5 < ratio < 2.0, f"Asimmetria in distribuzione simmetrica: {ratio}"
