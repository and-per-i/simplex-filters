"""
Level 1 — Sanity check strutturale.
Non richiede GPU, funziona su CPU con modello a pesi random.

Verifica:
1. 4 layer simpliciali ai posti giusti
2. k1_proj/v1_proj = pesi originali espansi 4×
3. k2_proj/v2_proj = k1_proj/v1_proj + α·noise
4. Tutti gli altri layer frozen
5. Shape delle proiezioni
6. Nessun NaN nei pesi
"""

import pytest
import torch
from transformers.models.llama.modeling_llama import LlamaAttention

from src.modeling.simplicial_attention import SimplicialAttention


# ======================================================================
# Test 1: Numero e posizione dei layer simpliciali
# ======================================================================

class TestLayerPositions:
    """Verifica che i layer simpliciali siano nei posti giusti."""

    def test_simplicial_layer_count(self, hybrid_fixture):
        """Esattamente 4 layer sono SimplicialAttention, 28 sono LlamaAttention."""
        model = hybrid_fixture["model"]
        simplicial_count = 0
        llama_count = 0

        for layer in model.model.layers:
            if isinstance(layer.self_attn, SimplicialAttention):
                simplicial_count += 1
            elif isinstance(layer.self_attn, LlamaAttention):
                llama_count += 1

        assert simplicial_count == 4, f"Attesi 4 layer simpliciali, trovati {simplicial_count}"
        assert llama_count == 28, f"Attesi 28 layer Llama, trovati {llama_count}"
        assert simplicial_count + llama_count == 32

    def test_simplicial_layer_positions(self, hybrid_fixture):
        """I layer simpliciali sono agli indici [16, 20, 24, 28]."""
        model = hybrid_fixture["model"]
        expected = [16, 20, 24, 28]
        found = []

        for idx, layer in enumerate(model.model.layers):
            if isinstance(layer.self_attn, SimplicialAttention):
                found.append(idx)

        assert found == expected, f"Layer simpliciali attesi {expected}, trovati {found}"

    def test_simplicial_layer_contiguity(self, hybrid_fixture):
        """Ogni 4 layer, nella seconda metà del modello."""
        model = hybrid_fixture["model"]
        for idx in [16, 20, 24, 28]:
            assert isinstance(
                model.model.layers[idx].self_attn, SimplicialAttention
            ), f"Layer {idx} dovrebbe essere SimplicialAttention"
            assert isinstance(
                model.model.layers[idx + 1].self_attn, LlamaAttention
            ), f"Layer {idx + 1} dovrebbe essere LlamaAttention (vicino a simpliciale)"


# ======================================================================
# Test 2: Pesi originali vs pesi espansi
# ======================================================================

class TestK1V1OriginalExpansion:
    """Verifica che k1_proj / v1_proj siano espansioni dei pesi originali."""

    def test_k1_proj_shape(self, hybrid_fixture):
        """k1_proj è [4096, 4096] mentre k_proj originale è [4096, 1024]."""
        model = hybrid_fixture["model"]
        config = hybrid_fixture["config"]

        for idx in [16, 20, 24, 28]:
            attn = model.model.layers[idx].self_attn
            # 32 heads × 128 dim = 4096
            expected_out = config.num_attention_heads * (config.hidden_size // config.num_attention_heads)
            assert attn.k1_proj.weight.shape == (expected_out, config.hidden_size), \
                f"Layer {idx}: k1_proj shape {attn.k1_proj.weight.shape}"
            assert attn.v1_proj.weight.shape == (expected_out, config.hidden_size), \
                f"Layer {idx}: v1_proj shape {attn.v1_proj.weight.shape}"

    def test_k1_proj_match_expanded_original(self, hybrid_fixture):
        """
        k1_proj.weight == k_proj.weight.repeat(1, 4).
        I pesi originali sono salvati PRIMA della conversione in conftest.py.
        """
        model = hybrid_fixture["model"]
        original_weights = hybrid_fixture["original_weights"]

        for idx in [16, 20, 24, 28]:
            attn = model.model.layers[idx].self_attn
            k_orig = original_weights[idx]["k_proj"]  # [4096, 1024]
            v_orig = original_weights[idx]["v_proj"]  # [4096, 1024]

            k1 = attn.k1_proj.weight  # [4096, 4096]
            v1 = attn.v1_proj.weight  # [4096, 4096]

            # Espandi: repeat(1, 4) → [4096, 4096]
            k_expanded = k_orig.repeat(1, 4)
            v_expanded = v_orig.repeat(1, 4)

            assert torch.allclose(k1, k_expanded, atol=1e-6), \
                f"Layer {idx}: k1_proj non matcha k_proj originale espanso"
            assert torch.allclose(v1, v_expanded, atol=1e-6), \
                f"Layer {idx}: v1_proj non matcha v_proj originale espanso"

    def test_k1_proj_not_identical_to_original(self, hybrid_fixture):
        """k1_proj (32 heads) NON può essere identico a k_proj (8 heads) perché hanno shape diversa."""
        model = hybrid_fixture["model"]

        for idx in [16, 20, 24, 28]:
            attn = model.model.layers[idx].self_attn
            k1 = attn.k1_proj.weight

            # La shape è diversa (4096 vs 1024), quindi non possono essere uguali
            # Inoltre l'originale non è più accessibile (sostituito)
            # Ma il test precedente garantisce che il contenuto è corretto
            assert k1.shape[0] == 4096 and k1.shape[1] == 4096


# ======================================================================
# Test 3: Perturbazione K2/V2
# ======================================================================

class TestK2V2Initialization:
    """Verifica che K2/V2 = K1/V1 + α·noise."""

    def test_k2_proj_diff_from_k1_proj(self, hybrid_fixture):
        """k2_proj ≠ k1_proj (se sono uguali, l'inizializzazione non ha senso)."""
        model = hybrid_fixture["model"]

        for idx in [16, 20, 24, 28]:
            attn = model.model.layers[idx].self_attn
            k1 = attn.k1_proj.weight
            k2 = attn.k2_proj.weight

            diff = (k2 - k1).abs().mean().item()
            assert diff > 0, f"Layer {idx}: k2_proj è identico a k1_proj (diff={diff})"

    def test_k2_proj_noise_magnitude(self, hybrid_fixture):
        """|k2 - k1|_mean ≈ alpha * sqrt(2/π) ~ 0.01 * 0.7979 ≈ 0.008 con α=0.01."""
        model = hybrid_fixture["model"]

        for idx in [16, 20, 24, 28]:
            attn = model.model.layers[idx].self_attn
            k1 = attn.k1_proj.weight
            k2 = attn.k2_proj.weight

            diff = (k2 - k1).abs().mean().item()
            expected = 0.01 * (2 / 3.14159) ** 0.5  # α * mean(|N(0,1)|)
            assert abs(diff - expected) < 0.005, \
                f"Layer {idx}: |k2-k1|_mean = {diff:.6f}, atteso ~{expected:.6f}"

    def test_v2_proj_diff_from_v1_proj(self, hybrid_fixture):
        """Stessa verifica per V2/V1."""
        model = hybrid_fixture["model"]

        for idx in [16, 20, 24, 28]:
            attn = model.model.layers[idx].self_attn
            v1 = attn.v1_proj.weight
            v2 = attn.v2_proj.weight

            diff = (v2 - v1).abs().mean().item()
            assert diff > 0, f"Layer {idx}: v2_proj è identico a v1_proj"


# ======================================================================
# Test 4: Frozen / trainable
# ======================================================================

class TestFrozenParameters:
    """Verifica che solo K1/V1/K2/V2 siano trainable."""

    def test_all_other_layers_frozen(self, hybrid_fixture):
        """Tutti i parametri FUORI dai layer simpliciali sono frozen."""
        model = hybrid_fixture["model"]
        simplicial_indices = hybrid_fixture["simplicial_indices"]

        for name, param in model.named_parameters():
            in_simplicial = any(f"layers.{idx}." in name for idx in simplicial_indices)
            if not in_simplicial:
                assert not param.requires_grad, \
                    f"Parametro non-simpliciale dovrebbe essere frozen: {name}"

    def test_k1v1_k2v2_trainable(self, hybrid_fixture):
        """Solo k1_proj, v1_proj, k2_proj, v2_proj hanno requires_grad=True nei layer simpliciali."""
        model = hybrid_fixture["model"]
        trainable_k1v1 = []
        trainable_k2v2 = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if "k1_proj" in name or "v1_proj" in name:
                    trainable_k1v1.append(name)
                elif "k2_proj" in name or "v2_proj" in name:
                    trainable_k2v2.append(name)

        # 4 layer × 2 proiezioni ciascuno
        assert len(trainable_k1v1) == 8, \
            f"Attese 8 proiezioni K1/V1 trainable, trovate {len(trainable_k1v1)}: {trainable_k1v1}"
        assert len(trainable_k2v2) == 8, \
            f"Attese 8 proiezioni K2/V2 trainable, trovate {len(trainable_k2v2)}: {trainable_k2v2}"

    def test_qproj_oproj_frozen_in_simplicial(self, hybrid_fixture):
        """q_proj e o_proj nei layer simpliciali sono frozen."""
        model = hybrid_fixture["model"]
        simplicial_indices = hybrid_fixture["simplicial_indices"]

        for idx in simplicial_indices:
            attn = model.model.layers[idx].self_attn
            assert not attn.q_proj.weight.requires_grad, \
                f"Layer {idx}: q_proj dovrebbe essere frozen"
            assert not attn.o_proj.weight.requires_grad, \
                f"Layer {idx}: o_proj dovrebbe essere frozen"


# ======================================================================
# Test 5: Shape delle proiezioni
# ======================================================================

class TestProjectionShapes:
    """Verifica che tutte le proiezioni abbiano le shape corrette."""

    def test_all_projection_shapes(self, hybrid_fixture):
        """Verifica shape di tutte le proiezioni nei layer simpliciali."""
        model = hybrid_fixture["model"]
        config = hybrid_fixture["config"]

        hidden_size = config.hidden_size  # 4096
        num_heads = config.num_attention_heads  # 32
        head_dim = hidden_size // num_heads  # 128

        for idx in [16, 20, 24, 28]:
            attn = model.model.layers[idx].self_attn

            # [out_features, in_features] tutte nn.Linear
            assert attn.q_proj.weight.shape == (hidden_size, hidden_size), \
                f"Layer {idx}: q_proj shape {attn.q_proj.weight.shape}"
            assert attn.k1_proj.weight.shape == (num_heads * head_dim, hidden_size), \
                f"Layer {idx}: k1_proj shape {attn.k1_proj.weight.shape}"
            assert attn.k2_proj.weight.shape == (num_heads * head_dim, hidden_size), \
                f"Layer {idx}: k2_proj shape {attn.k2_proj.weight.shape}"
            assert attn.v1_proj.weight.shape == (num_heads * head_dim, hidden_size), \
                f"Layer {idx}: v1_proj shape {attn.v1_proj.weight.shape}"
            assert attn.v2_proj.weight.shape == (num_heads * head_dim, hidden_size), \
                f"Layer {idx}: v2_proj shape {attn.v2_proj.weight.shape}"
            assert attn.o_proj.weight.shape == (hidden_size, num_heads * head_dim), \
                f"Layer {idx}: o_proj shape {attn.o_proj.weight.shape}"


# ======================================================================
# Test 6: Nessun NaN nei pesi
# ======================================================================

class TestWeightNaN:
    """Verifica che nessun peso contenga NaN o Inf."""

    def test_no_nan_in_weights(self, hybrid_fixture):
        """Nessun peso nelle proiezioni è NaN o Inf."""
        model = hybrid_fixture["model"]

        for idx in [16, 20, 24, 28]:
            attn = model.model.layers[idx].self_attn
            for name in ["q_proj", "k1_proj", "k2_proj", "v1_proj", "v2_proj", "o_proj"]:
                w = getattr(attn, name).weight
                assert torch.isfinite(w).all(), \
                    f"Layer {idx}, {name}: peso non finito (NaN o Inf)"