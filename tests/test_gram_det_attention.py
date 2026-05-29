"""
Test per GramDetAttention.

Verifica:
1. Forward pass senza crash su input fittizi
2. Shape output
3. Nessun NaN/Inf
4. Correttezza del calcolo del determinante (confronto con torch.det)
5. Gradiente fluisce (backward pass)
6. Simmetria della finestra
7. Confronto qualitativo con attenzione standard
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from modeling.gram_det_attention import GramDetAttention, sarrus_determinant


BATCH_SIZE = 2
SEQ_LENGTH = 32
D_MODEL = 128
N_HEADS = 4


@pytest.fixture
def gram_attn():
    """GramDetAttention con default W=8."""
    return GramDetAttention(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        window_size=8,
    )


@pytest.fixture
def random_input():
    """Input random [B, N, D]."""
    return torch.randn(BATCH_SIZE, SEQ_LENGTH, D_MODEL)


# ======================================================================
# Test: Sarrus determinante vs torch.det
# ======================================================================

class TestSarrusDeterminant:
    """Verifica che sarrus_determinant() sia corretto vs torch.det()."""

    def test_sarrus_matches_torch_det(self):
        """Per matrici 3×3 casuali, il determinante via Sarrus matcha torch.det()."""
        B, H = 2, 4

        # Crea vettori casuali d-dimensionali
        d = 16
        q = torch.randn(B, H, 1, d)
        k1 = torch.randn(B, H, 8, d)
        k2 = torch.randn(B, H, 8, d)

        # Costruisci matrici di Gram 3×3 per ogni tripletta (j1,j2)
        # Gram[j1, j2] = sum_d( [q, k_j1, k_j2]^T @ [q, k_j1, k_j2] )
        # Ma per testare sarrus_determinant, confrontiamo i singoli dot products

        qq = (q * q).sum(dim=-1, keepdim=True)               # [B, H, 1, 1]
        k1k1 = (k1 * k1).sum(dim=-1).unsqueeze(-2)           # [B, H, 1, P]
        k2k2 = (k2 * k2).sum(dim=-1).unsqueeze(-1)           # [B, H, P, 1]
        qk1 = torch.matmul(q, k1.transpose(-2, -1))           # [B, H, 1, P]
        qk2 = torch.matmul(q, k2.transpose(-2, -1))           # [B, H, 1, P]
        k1k2 = torch.matmul(k1, k2.transpose(-2, -1))         # [B, H, P, P]

        # Calcola via Sarrus
        det_sarrus = sarrus_determinant(qq, k1k1, k2k2, qk1, qk2, k1k2)

        # Calcola via torch.det per alcune coppie specifiche
        P = 8
        for j1 in range(P):
            for j2 in range(P):
                if j1 >= j2:
                    continue
                # Costruisci Gram 3×3 esplicita per (q, k1_j1, k2_j2)
                gram = torch.zeros(B, H, 3, 3)
                qv = q[:, :, 0, :]    # [B, H, d]
                k1v = k1[:, :, j1, :]  # [B, H, d]
                k2v = k2[:, :, j2, :]  # [B, H, d]

                gram[:, :, 0, 0] = (qv * qv).sum(dim=-1)
                gram[:, :, 1, 1] = (k1v * k1v).sum(dim=-1)
                gram[:, :, 2, 2] = (k2v * k2v).sum(dim=-1)
                gram[:, :, 0, 1] = gram[:, :, 1, 0] = (qv * k1v).sum(dim=-1)
                gram[:, :, 0, 2] = gram[:, :, 2, 0] = (qv * k2v).sum(dim=-1)
                gram[:, :, 1, 2] = gram[:, :, 2, 1] = (k1v * k2v).sum(dim=-1)

                det_torch = torch.det(gram)  # [B, H]
                det_s = det_sarrus[:, :, j1, j2]  # [B, H]

                assert torch.allclose(det_s, det_torch, atol=1e-4), \
                    f"Mismatch at ({j1},{j2}): sarrus={det_s[0,0].item():.6f}, torch={det_torch[0,0].item():.6f}"


# ======================================================================
# Test: Forward pass
# ======================================================================

class TestForward:
    """Verifiche di base sul forward pass."""

    def test_forward_no_crash(self, gram_attn, random_input):
        """Forward pass senza eccezioni."""
        output = gram_attn(random_input)
        assert output is not None

    def test_output_shape(self, gram_attn, random_input):
        """Output [B, N, D] uguale all'input."""
        output = gram_attn(random_input)
        assert output.shape == (BATCH_SIZE, SEQ_LENGTH, D_MODEL), \
            f"Output shape {output.shape}"

    def test_no_nan_inf_output(self, gram_attn, random_input):
        """Output senza NaN o Inf."""
        output = gram_attn(random_input)
        assert torch.isfinite(output).all(), "Output contiene NaN o Inf!"

    def test_different_inputs_different_outputs(self, gram_attn):
        """Input diversi producono output diversi."""
        x1 = torch.randn(1, 5, D_MODEL)
        x2 = torch.randn(1, 5, D_MODEL)
        y1 = gram_attn(x1)
        y2 = gram_attn(x2)
        diff = (y1 - y2).abs().mean().item()
        assert diff > 1e-6, f"Output troppo simili per input diversi: diff={diff}"

    def test_deterministic(self, gram_attn, random_input):
        """Stesso input produce stesso output."""
        y1 = gram_attn(random_input)
        y2 = gram_attn(random_input)
        assert torch.allclose(y1, y2, atol=1e-5), "Non deterministico!"

    def test_forward_with_return_weights(self, gram_attn, random_input):
        """Forward con return_weights=True produce output + lista pesi."""
        output, weights = gram_attn(random_input, return_weights=True)
        assert output.shape == (BATCH_SIZE, SEQ_LENGTH, D_MODEL)
        assert len(weights) == SEQ_LENGTH  # un peso per ogni posizione
        for w in weights:
            assert w.shape[-2] == w.shape[-1]  # matrice quadrata P×P


# ======================================================================
# Test: Atenuazione della finestra
# ======================================================================

class TestWindowBehavior:
    """Verifica che la finestra funzioni correttamente."""

    def test_window_size_property(self):
        """Il parametro window_size è accessibile."""
        attn = GramDetAttention(d_model=64, n_heads=2, window_size=4)
        assert attn.window_size == 4

    def test_window_clamp(self):
        """Posizioni vicine ai bordi non causano errori."""
        attn = GramDetAttention(d_model=64, n_heads=2, window_size=4)
        x = torch.randn(1, 3, 64)  # sequenza molto corta, W=4
        output = attn(x)
        assert output.shape == (1, 3, 64)

    def test_wider_window_changes_output(self):
        """W più grande produce output diverso (più coppie considerate)."""
        x = torch.randn(1, 16, 64)
        attn_small = GramDetAttention(d_model=64, n_heads=2, window_size=2)
        attn_large = GramDetAttention(d_model=64, n_heads=2, window_size=6)

        y_small = attn_small(x)
        y_large = attn_large(x)

        diff = (y_small - y_large).abs().mean().item()
        assert diff > 1e-6, \
            f"Output troppo simili con W diverse: diff={diff}"


# ======================================================================
# Test: Backward pass
# ======================================================================

class TestBackward:
    """Verifica che il gradiente fluisca correttamente."""

    def test_backward_no_crash(self, gram_attn, random_input):
        """Backward pass senza eccezioni."""
        output = gram_attn(random_input)
        loss = output.mean()
        loss.backward()
        # Successo se arriva qui

    def test_all_parameters_have_grad(self, gram_attn, random_input):
        """Tutti i parametri hanno gradiente non-nullo dopo backward."""
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
        if names_without_grad:
            print(f"  Parametri senza grad: {names_without_grad}")

    def test_gradient_not_exploding(self, gram_attn, random_input):
        """Nessun gradiente esploso (>100)."""
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
    """Verifica che o_proj lavori correttamente."""

    def test_o_proj_shape(self, gram_attn):
        """o_proj è nn.Linear(H*d, D)."""
        Hd = gram_attn.n_heads * gram_attn.head_dim
        assert gram_attn.o_proj.in_features == Hd
        assert gram_attn.o_proj.out_features == gram_attn.d_model

    def test_output_not_cloned_input(self, gram_attn, random_input):
        """Output è diverso da input (l'attenzione ha effetto)."""
        output = gram_attn(random_input)
        # Verifica che l'output non sia semplicemente l'input clonato
        diff = (output - random_input).abs().mean().item()
        assert diff > 0, "Output identico a input — l'attenzione non fa nulla!"


# ======================================================================
# Test: Maschera coppie j1 < j2
# ======================================================================

class TestPairMasking:
    """Verifica che vengano usate solo coppie j1 < j2."""

    def test_pair_mask_symmetry(self, gram_attn, random_input):
        """Con return_weights=True, le matrici peso sono triangolari superiori."""
        _, weights = gram_attn(random_input, return_weights=True)

        for w in weights:
            # w è [B, H, P, P] — deve essere triangolare superiore con diag=0
            P = w.shape[-1]
            upper = torch.triu(torch.ones(P, P), diagonal=1)
            lower = torch.tril(torch.ones(P, P), diagonal=0)
            # Check: tutti i valori triangolari inferiori e diagonale devono essere ~0
            assert w[:, :, lower.bool()].abs().max().item() < 1e-5, \
                "Pesi d'attenzione non-zero in posizioni j1 >= j2!"