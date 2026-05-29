"""
Test per il pacchetto kv_cache (Q-filter, eviction, benchmark).

Verifica su dati sintetici (CPU, senza modello reale):
1. Q-filter score: formula corretta, monotonicità
2. top_k_indices: shape, ordinamento
3. random_indices: shape, riproducibilità
4. evict_keys: shape, strategia
5. BenchmarkResult: summary() produce output valido
"""

import pytest
import torch
import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kv_cache.qfilter_score import (
    qfilter_score,
    qfilter_score_single,
    top_k_indices,
    random_indices,
)
from kv_cache.eviction import evict_keys
from kv_cache.benchmark import BenchmarkResult


# ======================================================================
# Test: Q-filter score
# ======================================================================

class TestQFilterScore:
    """Verifica che il Q-filter score sia calcolato correttamente."""

    def test_qfilter_score_shape(self):
        """Input [N, d] → output [N]."""
        k = torch.randn(100, 32)
        sigma1, sigma2 = 2.0, 1.0
        U_mean = torch.eye(32)[:, :2]
        scores = qfilter_score(k, sigma1, sigma2, U_mean)
        assert scores.shape == (100,), f"Shape: {scores.shape}"

    def test_qfilter_score_non_negative(self):
        """Score non negativo."""
        k = torch.randn(50, 16)
        U_mean = torch.eye(16)[:, :2]
        scores = qfilter_score(k, 1.0, 1.0, U_mean)
        assert (scores >= 0).all(), "Score negativo!"

    def test_qfilter_score_sigma_weight(self):
        """σ₁ > σ₂ → componente e2 pesata di più."""
        U_mean = torch.eye(32)[:, :2]
        e1 = U_mean[:, 0]  # [32]
        e2 = U_mean[:, 1]  # [32]

        # Crea chiave allineata con e1
        k1 = e1.clone()
        k2 = e2.clone()

        # Con σ₁=10, σ₂=1: k2 (allineata con e2) ha score maggiore
        score_k1 = qfilter_score_single(k1, 10.0, 1.0, e1, e2)
        score_k2 = qfilter_score_single(k2, 10.0, 1.0, e1, e2)

        # k2 ha proiezione su e2 alta, pesata da σ₁²
        assert score_k2 > score_k1, f"k2={score_k2} dovrebbe > k1={score_k1}"

    def test_qfilter_score_orthogonal(self):
        """Chiave ortogonale al piano → score ≈ 0."""
        U_mean = torch.eye(32)[:, :2]
        e1 = U_mean[:, 0]
        e2 = U_mean[:, 1]

        # Crea vettore ortogonale a e1, e2
        k_orth = torch.zeros(32)
        k_orth[2] = 1.0  # usa dimensione 3 (fuori dal piano)

        score = qfilter_score_single(k_orth, 1.0, 1.0, e1, e2)
        assert score.item() < 1e-5, f"Score per vettore ortogonale = {score}"


# ======================================================================
# Test: top_k_indices e random_indices
# ======================================================================

class TestSelectIndices:
    """Verifica la selezione degli indici."""

    def test_top_k_shape(self):
        """top_k_indices restituisce B indici."""
        scores = torch.randn(100)
        indices = top_k_indices(scores, budget=0.3)
        assert indices.shape == (30,), f"Shape: {indices.shape}"

    def test_top_k_ordering(self):
        """top_k_indices ordina per score decrescente."""
        scores = torch.tensor([0.1, 0.5, 0.9, 0.3, 0.7])
        indices = top_k_indices(scores, budget=0.6)
        # I 3 migliori: 2 (0.9), 4 (0.7), 1 (0.5)
        expected = torch.tensor([2, 4, 1])
        assert torch.allclose(indices, expected), f"Indici: {indices}"

    def test_random_shape(self):
        """random_indices restituisce B indici."""
        indices = random_indices(100, budget=0.5)
        assert indices.shape == (50,), f"Shape: {indices.shape}"

    def test_random_unique(self):
        """random_indices non ha duplicati."""
        indices = random_indices(200, budget=0.3)
        assert len(indices) == len(set(indices.tolist())), "Duplicati!"

    def test_budget_min_1(self):
        """Budget molto piccolo restituisce almeno 1 indice."""
        indices = top_k_indices(torch.randn(100), budget=0.001)
        assert indices.shape[0] >= 1, "Nessun indice!"


# ======================================================================
# Test: evict_keys
# ======================================================================

class TestEvictKeys:
    """Verifica la funzione di eviction."""

    def test_evict_qfilter_shape(self):
        """Eviction restituisce [B, d] e [B]."""
        keys = torch.randn(100, 32)
        U_mean = torch.eye(32)[:, :2]
        survived, indices = evict_keys(keys, 1.0, 1.0, U_mean, budget=0.3)
        assert survived.shape == (30, 32), f"Shape keys: {survived.shape}"
        assert indices.shape == (30,), f"Shape indices: {indices.shape}"

    def test_evict_random_shape(self):
        """Eviction random restituisce [B, d]."""
        keys = torch.randn(100, 16)
        U_mean = torch.eye(16)[:, :2]
        survived, indices = evict_keys(keys, 1.0, 1.0, U_mean, budget=0.5,
                                        strategy="random")
        assert survived.shape == (50, 16)

    def test_evict_strategy_error(self):
        """Strategia sconosciuta → errore."""
        keys = torch.randn(10, 4)
        U_mean = torch.eye(4)[:, :2]
        with pytest.raises(ValueError):
            evict_keys(keys, 1.0, 1.0, U_mean, budget=0.5, strategy="unknown")


# ======================================================================
# Test: BenchmarkResult
# ======================================================================

class TestBenchmarkResult:
    """Verifica il dataclass dei risultati."""

    def test_summary_no_data(self):
        """Summary senza dati non crasha."""
        result = BenchmarkResult(model_name="test")
        s = result.summary()
        assert "test" in s

    def test_summary_with_data(self):
        """Summary con dati produce tabella."""
        result = BenchmarkResult(model_name="test")
        result.ppl_qfilter[0.5] = 10.0
        result.ppl_qfilter[0.3] = 15.0
        result.ppl_random[0.5] = 12.0
        result.ppl_random[0.3] = 20.0
        s = result.summary()
        assert "50%" in s or "50" in s
        assert "10.00" in s or "10.0" in s

    def test_delta_positive(self):
        """Random > Q-filter → delta positivo (Q-filter migliore)."""
        result = BenchmarkResult(model_name="test")
        result.ppl_qfilter[0.3] = 12.0
        result.ppl_random[0.3] = 18.0
        # Delta = random - qfilter = 6 > 0
        s = result.summary()
        assert "+" in s or "6" in s, "Delta non visibile"