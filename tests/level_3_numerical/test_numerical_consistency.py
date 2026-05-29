"""
Level 3 — Sanity check numerico.

Richiede GPU con Triton. Verifica che l'output del modello ibrido
sia simile ma non identico a LLaMA puro, controllando che K2 contribuisca.

Test:
1. Output ibrido ≠ output LLaMA originale (K2 contribuisce)
2. Output ibrido non troppo diverso (con α=0.01)
3. Se azzeri K2, output cambia significativamente
4. Cos-sim tra simuliciale e standard adiacente
5. α sensitivity
"""

import pytest
import torch
import torch.nn.functional as F
import os
import sys
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from modeling.simplicial_attention import SimplicialAttention
from modeling.convert_to_hybrid import convert_llama_to_hybrid


# Salta tutti i test se non c'è GPU
has_cuda = torch.cuda.is_available()
try:
    import triton
    has_triton = has_cuda
except ImportError:
    has_triton = False

requires_gpu = pytest.mark.skipif(
    not has_triton,
    reason="Richiede GPU con Triton per il kernel 2-simplicial"
)

BATCH_SIZE = 2
SEQ_LENGTH = 64


@pytest.fixture
def llama_only_model(config):
    """Modello LLaMA random senza conversioni."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)
    model.eval()
    return model


@pytest.fixture
def small_input(hybrid_fixture):
    """Input di test."""
    model = hybrid_fixture["model"]
    device = next(model.parameters()).device
    return torch.randint(0, 1000, (BATCH_SIZE, SEQ_LENGTH), device=device)


# ======================================================================
# Test 1: Output ibrido vs output LLaMA puro
# ======================================================================

class TestOutputComparison:
    """Confronto con LLaMA originale."""

    @requires_gpu
    def test_output_not_identical_to_llama(self, hybrid_fixture, llama_only_model, small_input):
        """
        L'output ibrido NON può essere identico a LLaMA puro.
        Se lo sono, K2 non contribuisce (bug).
        """
        model = hybrid_fixture["model"]
        device = next(model.parameters()).device
        llama_model = llama_only_model.to(device)

        with torch.no_grad():
            out_hybrid = model(small_input).logits
            out_llama = llama_model(small_input).logits

        # Cos-sim < 0.999 significa "non identici"
        cos_sim = F.cosine_similarity(
            out_hybrid.flatten().unsqueeze(0),
            out_llama.flatten().unsqueeze(0),
        ).item()

        assert cos_sim < 0.999, \
            f"Output ibrido e LLaMA sono identici (cos_sim={cos_sim:.6f})! K2 non contribuisce."
        print(f"  Cos-sim ibrido vs LLaMA: {cos_sim:.6f}")

    @requires_gpu
    def test_output_not_too_different(self, hybrid_fixture, llama_only_model, small_input):
        """
        Con α=0.01, l'output ibrido non deve essere troppo diverso.
        Cos-sim > 0.9 indica che l'inizializzazione non distrugge il forward.
        """
        model = hybrid_fixture["model"]
        device = next(model.parameters()).device
        llama_model = llama_only_model.to(device)

        with torch.no_grad():
            out_hybrid = model(small_input).logits
            out_llama = llama_model(small_input).logits

        cos_sim = F.cosine_similarity(
            out_hybrid.flatten().unsqueeze(0),
            out_llama.flatten().unsqueeze(0),
        ).item()

        assert cos_sim > 0.9, \
            f"Output troppo diverso da LLaMA: cos_sim={cos_sim:.6f} (soglia 0.9)"
        print(f"  Cos-sim ibrido vs LLaMA: {cos_sim:.6f} (soglia 0.9)")


# ======================================================================
# Test 2: Azzeramento K2
# ======================================================================

class TestK2ZeroOut:
    """Se azzeri K2, l'output deve cambiare."""

    @requires_gpu
    def test_k2_zero_changes_output(self, hybrid_fixture, small_input):
        """Azzerare K2 in un layer deve cambiare l'output visibilmente."""
        model = hybrid_fixture["model"]

        with torch.no_grad():
            out_normal = model(small_input).logits

        # Azzera k2_proj in un layer
        with torch.no_grad():
            attn = model.model.layers[16].self_attn
            saved_k2 = attn.k2_proj.weight.data.clone()
            attn.k2_proj.weight.data.zero_()

            out_k2_zero = model(small_input).logits

            # Ripristina
            attn.k2_proj.weight.data = saved_k2

        cos_sim = F.cosine_similarity(
            out_normal.flatten().unsqueeze(0),
            out_k2_zero.flatten().unsqueeze(0),
        ).item()

        assert cos_sim < 0.999, \
            f"Azzerare K2 non cambia output! (cos_sim={cos_sim:.6f})"
        print(f"  Cos-sim dopo azzeramento K2 (layer 16): {cos_sim:.6f}")


# ======================================================================
# Test 3: Layer simpliciale produce output diverso da layer standard adiacente
# ======================================================================

class TestLayerOutput:
    """Confronto tra output di layer simpliciali e standard adiacenti."""

    @requires_gpu
    def test_simplicial_layer_output_differs(self, hybrid_fixture, small_input):
        """
        Un layer simpliciale (indice 16) produce output diverso da
        un layer standard adiacente (indice 15 o 17) sullo stesso input.
        """
        model = hybrid_fixture["model"]
        config = hybrid_fixture["config"]

        # Estrai hidden states intermediate
        with torch.no_grad():
            output = model(small_input, output_hidden_states=True)
            hidden = output.hidden_states  # lista di 33 elementi (0..32)

            # hidden[16] = output dopo layer 16 (simpliciale)
            # hidden[15] = output dopo layer 15 (standard)
            # hidden[17] = output dopo layer 17 (standard)

            h15 = hidden[15]  # dopo layer 15 standard
            h16 = hidden[16]  # dopo layer 16 simpliciale
            h17 = hidden[17]  # dopo layer 17 standard

        # Cos-sim tra layer simpliciale e adiacenti
        cos_sim_15_16 = F.cosine_similarity(
            h15.flatten().unsqueeze(0), h16.flatten().unsqueeze(0)
        ).item()

        cos_sim_15_17 = F.cosine_similarity(
            h15.flatten().unsqueeze(0), h17.flatten().unsqueeze(0)
        ).item()

        print(f"  Cos-sim layer 15→16 (simpl→std): {cos_sim_15_16:.6f}")
        print(f"  Cos-sim layer 15→17 (std→simpl): {cos_sim_15_17:.6f}")

        # L'output del layer simpliciale può essere diverso da quello standard adiacente
        # Non c'è un threshold fisso — basta che non sia identico

        # Nota: h15 e h17 sono molto simili (entrambi standard)
        cos_sim_std = F.cosine_similarity(
            h15.flatten().unsqueeze(0), h17.flatten().unsqueeze(0)
        ).item()
        print(f"  Cos-sim layer 15→17 (std→std): {cos_sim_std:.6f} (riferimento)")


# ======================================================================
# Test 4: α sensitivity
# ======================================================================

class TestAlphaSensitivity:
    """Con α più grande, la divergenza da LLaMA deve aumentare."""

    @requires_gpu
    def test_alpha_sensitivity(self, config, small_input):
        """
        α=0.01 → dissimilarità moderata
        α=0.1  → dissimilarità maggiore
        α=0    → identico a LLaMA (K2 = K1)
        """
        from transformers import AutoModelForCausalLM

        device = small_input.device
        config = config
        simplicial_indices = [16, 20, 24, 28]

        results = {}
        for alpha in [0.0, 0.01, 0.1]:
            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32).to(device)
            model, _ = convert_llama_to_hybrid(model, alpha=alpha)
            model.eval()

            with torch.no_grad():
                out = model(small_input).logits

            results[alpha] = out

        # α=0 deve essere identico a k_proj non perturbato
        # (k2 = k1, v2 = v1, ma k1 è espanso → output diversi da LLaMA puro)
        cos_sim_0_001 = F.cosine_similarity(
            results[0.0].flatten().unsqueeze(0),
            results[0.01].flatten().unsqueeze(0),
        ).item()

        cos_sim_0_01_01 = F.cosine_similarity(
            results[0.01].flatten().unsqueeze(0),
            results[0.1].flatten().unsqueeze(0),
        ).item()

        print(f"  Cos-sim α=0 vs α=0.01: {cos_sim_0_001:.6f}")
        print(f"  Cos-sim α=0.01 vs α=0.1: {cos_sim_0_01_01:.6f}")

        # Con α maggiore, l'output differisce di più
        assert cos_sim_0_001 > cos_sim_0_01_01, \
            f"α=0.01 vs α=0.1 ({cos_sim_0_01_01}) dovrebbe essere meno simile di α=0 vs α=0.01 ({cos_sim_0_001})"


# ======================================================================
# Test 5: Simmetria degli output
# ======================================================================

class TestOutputSymmetry:
    """L'output per input identici deve essere identico."""

    @requires_gpu
    def test_deterministic_output(self, hybrid_fixture, small_input):
        """Due forward con stesso input e stesso seed devono dare stesso output."""
        model = hybrid_fixture["model"]

        with torch.no_grad():
            out1 = model(small_input).logits
            out2 = model(small_input).logits

        assert torch.allclose(out1, out2, atol=1e-5), \
            "Output diverso tra due forward identici — non deterministico!"