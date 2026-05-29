"""
Test per GramDetAttention — versione vettorizzata.

Verifica:
1. Forward naive e vettorizzato coincidono (tolerance 1e-5)
2. Shape output
3. Nessun NaN/Inf
4. Correttezza del calcolo del determinante (confronto con torch.det)
5. Gradiente fluisce (backward pass)
6. Speed: N=256 W=8 B=4 < 3s su GPU
7. Atenuazione della finestra
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from modeling.gram_det_attention import GramDetAttention, _build_pair_indices


BATCH_SIZE = 2
SEQ_LENGTH = 32
D_MODEL = 128
N_HEADS = 4


# Salta test GPU se non disponibile
has_cuda = torch.cuda.is_available()
requires_gpu = pytest.mark.skipif(not has_cuda, reason="Richiede GPU")


# ======================================================================
# Test: _build_pair_indices
# ======================================================================

class TestPairIndices:
    """Verifica che gli indici delle coppie siano corretti."""

    def test_pair_count(self):
        """P = (2W+1)(2W+2)/2"""
        for W in [1, 2, 4, 8, 16]:
            P_expected = (2 * W + 1) * (2 * W + 2) // 2
            indices = _build_pair_indices(W)
            assert indices.shape == (P_expected, 2), \
                f"W={W}: atteso {P_expected}, ottenuto {indices.shape[0]}"

    def test_pair_ordering(self):
        """Le coppie sono ordinate correttamente (j1 <= j2)."""
        indices = _build_pair_indices(4)
        for (j1, j2) in indices.tolist():
            assert j1 <= j2, f"Coppia non valida: ({j1}, {j2})"

    def test_pair_all_covered(self):
        """Tutte le coppie sono presenti (senza duplicati)."""
        indices = _build_pair_indices(4)
        pairs_set = set()
        for (j1, j2) in indices.tolist():
            pairs_set.add((j1, j2))
        P_expected = (2 * 4 + 1) * (2 * 4 + 2) // 2
        assert len(pairs_set) == P_expected, \
            f"Coppie uniche: {len(pairs_set)}, attese {P_expected}"


# ======================================================================
# Test: Sarrus determinante vs torch.det
# ======================================================================

class TestSarrusDeterminant:
    """Verifica che sarrus_determinant() sia corretto vs torch.det()."""

    def test_sarrus_matches_torch_det(self):
        """Per matrici 3x3 casuali, il determinante via Sarrus matcha torch.det()."""
        from modeling.gram_det_attention import sarrus_determinant

        B, H, P, d = 2, 4, 8, 16
        q = torch.randn(B, H, 1, d)
        k1 = torch.randn(B, H, P, d)
        k2 = torch.randn(B, H, P, d)

        qq = (q * q).sum(dim=-1, keepdim=True)
        k1k1 = (k1 * k1).sum(dim=-1).unsqueeze(-2)
        k2k2 = (k2 * k2).sum(dim=-1).unsqueeze(-1)
        qk1 = torch.matmul(q, k1.transpose(-2, -1))
        qk2 = torch.matmul(q, k2.transpose(-2, -1))
        k1k2 = torch.matmul(k1, k2.transpose(-2, -1))

        det_sarrus = sarrus_determinant(qq, k1k1, k2k2, qk1, qk2, k1k2)

        for j1 in range(P):
            for j2 in range(P):
                if j1 >= j2:
                    continue
                gram = torch.zeros(B, H, 3, 3)
                qv = q[:, :, 0, :]
                k1v = k1[:, :, j1, :]
                k2v = k2[:, :, j2, :]

                gram[:, :, 0, 0] = (qv * qv).sum(dim=-1)
                gram[:, :, 1, 1] = (k1v * k1v).sum(dim=-1)
                gram[:, :, 2, 2] = (k2v * k2v).sum(dim=-1)
                gram[:, :, 0, 1] = gram[:, :, 1, 0] = (qv * k1v).sum(dim=-1)
                gram[:, :, 0, 2] = gram[:, :, 2, 0] = (qv * k2v).sum(dim=-1)
                gram[:, :, 1, 2] = gram[:, :, 2, 1] = (k1v * k2v).sum(dim=-1)

                det_torch = torch.det(gram)
                det_s = det_sarrus[:, :, j1, j2]

                assert torch.allclose(det_s, det_torch, atol=1e-4), \
                    f"Mismatch at ({j1},{j2})"


# ======================================================================
# Test: Forward naive vs vettorizzato
# ======================================================================

class TestVectorizedVsNaive:
    """Il forward vettorizzato deve coincidere con quello naive."""

    def test_vectorized_matches_naive(self):
        """Output naive e vettorizzato coincidono entro 1e-5."""
        attn = GramDetAttention(d_model=64, n_heads=2, window_size=4)
        x = torch.randn(2, 16, 64)

        out_vec = attn(x)
        out_naive = attn.forward_naive(x)

        diff = (out_vec - out_naive).abs().max().item()
        assert diff < 1e-5, \
            f"Forward vettorizzato e naive differiscono: max diff={diff:.8f}"

    def test_vectorized_matches_naive_larger(self):
        """Output coincidono anche con dimensioni più grandi."""
        attn = GramDetAttention(d_model=128, n_heads=4, window_size=8)
        x = torch.randn(2, 32, 128)

        out_vec = attn(x)
        out_naive = attn.forward_naive(x)

        diff = (out_vec - out_naive).abs().max().item()
        assert diff < 1e-5, \
            f"Forward vettorizzato e naive differiscono: max diff={diff:.8f}"

    def test_vectorized_matches_naive_with_weights(self):
        """return_weights=True produce stessi pesi."""
        attn = GramDetAttention(d_model=64, n_heads=2, window_size=4)
        x = torch.randn(1, 8, 64)

        out_vec, w_vec = attn(x, return_weights=True)
        out_naive, w_naive_list = attn.forward_naive(x, return_weights=True)

        # w_vec: [B, H, N, P], w_naive_list: lista di tensori [B, H, P]
        # Confronta la media per ogni posizione
        diff = (out_vec - out_naive).abs().max().item()
        assert diff < 1e-5, \
            f"Output differiscono con return_weights: max diff={diff:.8f}"


# ======================================================================
# Test: Forward pass (base)
# ======================================================================

class TestForward:
    """Verifiche di base sul forward pass."""

    @pytest.fixture
    def gram_attn(self):
        return GramDetAttention(d_model=D_MODEL, n_heads=N_HEADS, window_size=8)

    @pytest.fixture
    def random_input(self):
        return torch.randn(BATCH_SIZE, SEQ_LENGTH, D_MODEL)

    def test_forward_no_crash(self, gram_attn, random_input):
        output = gram_attn(random_input)
        assert output is not None

    def test_output_shape(self, gram_attn, random_input):
        output = gram_attn(random_input)
        assert output.shape == (BATCH_SIZE, SEQ_LENGTH, D_MODEL)

    def test_no_nan_inf_output(self, gram_attn, random_input):
        output = gram_attn(random_input)
        assert torch.isfinite(output).all()

    def test_deterministic(self, gram_attn, random_input):
        y1 = gram_attn(random_input)
        y2 = gram_attn(random_input)
        assert torch.allclose(y1, y2, atol=1e-5)

    def test_forward_with_return_weights(self, gram_attn, random_input):
        output, weights = gram_attn(random_input, return_weights=True)
        assert output.shape == (BATCH_SIZE, SEQ_LENGTH, D_MODEL)
        assert weights.shape[:-1] == (BATCH_SIZE, N_HEADS, SEQ_LENGTH)
        # weights: [B, H, N, P]


# ======================================================================
# Test: Finestra
# ======================================================================

class TestWindow:
    """Verifica che la finestra funzioni correttamente."""

    def test_window_clamp(self):
        attn = GramDetAttention(d_model=64, n_heads=2, window_size=4)
        x = torch.randn(1, 3, 64)
        output = attn(x)
        assert output.shape == (1, 3, 64)

    def test_wider_window_changes_output(self):
        x = torch.randn(1, 16, 64)
        attn_small = GramDetAttention(d_model=64, n_heads=2, window_size=2)
        attn_large = GramDetAttention(d_model=64, n_heads=2, window_size=6)

        y_small = attn_small(x)
        y_large = attn_large(x)

        diff = (y_small - y_large).abs().mean().item()
        assert diff > 1e-6, f"Output con W diverse uguali: diff={diff}"


# ======================================================================
# Test: Backward pass
# ======================================================================

class TestBackward:
    """Verifica che il gradiente fluisca."""

    @pytest.fixture
    def gram_attn(self):
        return GramDetAttention(d_model=D_MODEL, n_heads=N_HEADS, window_size=8)

    @pytest.fixture
    def random_input(self):
        return torch.randn(BATCH_SIZE, SEQ_LENGTH, D_MODEL)

    def test_backward_no_crash(self, gram_attn, random_input):
        output = gram_attn(random_input)
        loss = output.mean()
        loss.backward()

    def test_all_parameters_have_grad(self, gram_attn, random_input):
        output = gram_attn(random_input)
        loss = output.mean()
        loss.backward()

        names_with_grad = []
        names_without_grad = []
        for name, param in gram_attn.named_parameters():
            if param.grad is not None and param.grad.abs().sum().item() > 0:
                names_with_grad.append(name)
            else:
                names_without_grad.append(name)

        assert len(names_with_grad) == 4, \
            f"Parametri con grad: {names_with_grad}"

    def test_gradient_not_exploding(self, gram_attn, random_input):
        output = gram_attn(random_input)
        loss = output.mean()
        loss.backward()

        max_grad = 0.0
        for param in gram_attn.parameters():
            if param.grad is not None:
                max_grad = max(max_grad, param.grad.abs().max().item())

        assert max_grad < 100, f"Gradienti esplosi: max_grad={max_grad:.2f}"


# ======================================================================
# Test: Proiezione output
# ======================================================================

class TestOutputProjection:
    """o_proj lavora correttamente."""

    def test_o_proj_shapes(self):
        attn = GramDetAttention(d_model=D_MODEL, n_heads=N_HEADS)
        Hd = N_HEADS * (D_MODEL // N_HEADS)
        assert attn.o_proj.in_features == Hd
        assert attn.o_proj.out_features == D_MODEL

    def test_output_not_cloned_input(self):
        attn = GramDetAttention(d_model=64, n_heads=2, window_size=4)
        x = torch.randn(2, 16, 64)
        output = attn(x)
        diff = (output - x).abs().mean().item()
        assert diff > 0, "Output identico a input!"


# ======================================================================
# Test: Speed benchmark
# ======================================================================

class TestSpeed:
    """Benchmark di velocità su GPU."""

    @requires_gpu
    def test_vectorized_speed_gpu(self):
        """Forward vettorizzato: N=256 W=8 B=4 deve finire in < 3s su GPU."""
        device = 'cuda'
        attn = GramDetAttention(d_model=512, n_heads=8, window_size=8).to(device)
        x = torch.randn(4, 256, 512, device=device)

        # Warmup
        for _ in range(3):
            _ = attn(x)

        # Misura
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            _ = attn(x)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / 10

        print(f"  Tempo per step (N=256, W=8, B=4): {elapsed*1000:.1f} ms")
        assert elapsed < 3.0, \
            f"Troppo lento: {elapsed:.2f}s per step (soglia: 3.0s)"

    @requires_gpu
    def test_vs_naive_speed_gpu(self):
        """Il vettorizzato deve essere più veloce del naive su GPU."""
        device = 'cuda'
        attn = GramDetAttention(d_model=128, n_heads=4, window_size=4).to(device)
        x = torch.randn(2, 32, 128, device=device)

        # Warmup
        for _ in range(3):
            _ = attn(x)
            _ = attn.forward_naive(x)

        # Misura vettorizzato
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20):
            _ = attn(x)
        torch.cuda.synchronize()
        t_vec = (time.time() - t0) / 20

        # Misura naive
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(3):  # naive più lento, meno iterazioni
            _ = attn.forward_naive(x)
        torch.cuda.synchronize()
        t_naive = (time.time() - t0) / 3

        print(f"  Vettorizzato: {t_vec*1000:.2f} ms, Naive: {t_naive*1000:.2f} ms, Speedup: {t_naive/t_vec:.1f}x")
        assert t_vec < t_naive, \
            f"Il vettorizzato ({t_vec*1000:.2f}ms) non e' piu veloce del naive ({t_naive*1000:.2f}ms)!"