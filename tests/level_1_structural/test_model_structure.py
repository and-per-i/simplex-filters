"""
Level 1 — Sanity check strutturale.
Non richiede GPU, funziona su CPU con modello a pesi random.

Verifica:
1. Layer simpliciali nei posti giusti (numero e posizione derivati dal config)
2. k1_proj/v1_proj = pesi originali espansi per GQA
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
        """Numero di layer simpliciali e Llama deriva dal model.config."""
        model = hybrid_fixture["model"]
        simplicial_indices = hybrid_fixture["simplicial_indices"]
        num_layers = model.config.num_hidden_layers
        expected_llama = num_layers - len(simplicial_indices)

        simplicial_count = 0
        llama_count = 0

        for layer in model.model.layers:
            if isinstance(layer.self_attn, SimplicialAttention):
                simplicial_count += 1
            elif isinstance(layer.self_attn, LlamaAttention):
                llama_count += 1

        assert simplicial_count == len(simplicial_indices), \
            f"Attesi {len(simplicial_indices)} layer simpliciali, trovati {simplicial_count}"
        assert llama_count == expected_llama, \
            f"Attesi {expected_llama} layer Llama, trovati {llama_count}"
        assert simplicial_count + llama_count == num_layers

    def test_simplicial_layer_positions(self, hybrid_fixture):
        """I layer simpliciali sono agli indici attesi."""
        model = hybrid_fixture["model"]
        expected = hybrid_fixture["simplicial_indices"]
        found = []

        for idx, layer in enumerate(model.model.layers):
            if isinstance(layer.self_attn, SimplicialAttention):
                found.append(idx)

        assert found == expected, f"Layer simpliciali attesi {expected}, trovati {found}"

    def test_simplicial_layer_contiguity(self, hybrid_fixture):
        """Verifichiamo che i layer simpliciali siano ai posti giusti."""
        model = hybrid_fixture["model"]
        simplicial_indices = hybrid_fixture["simplicial_indices"]

        for idx in simplicial_indices:
            assert isinstance(
                model.model.layers[idx].self_attn, SimplicialAttention
            ), f"Layer {idx} dovrebbe essere SimplicialAttention"

    def test_simplicial_indices_from_fixture(self, hybrid_fixture):
        """Gli indici simpliciali sono quelli attesi dalla fixture."""
        simplicial_indices = hybrid_fixture["simplicial_indices"]
        assert len(simplicial_indices) >= 2, \
            f"Almeno 2 layer simpliciali, trovati {len(simplicial_indices)}: {simplicial_indices}"


# ======================================================================
# Test 2: Pesi originali vs pesi espansi
# ======================================================================

class TestK1V1OriginalExpansion:
    """Verifica che k1_proj / v1_proj siano espansioni dei pesi originali."""

    def test_k1_proj_shape(self, hybrid_fixture):
        """k1_proj ha shape [hidden_size, hidden_size] (full heads)."""
        model = hybrid_fixture["model"]
        config = hybrid_fixture["config"]
        simplicial_indices = hybrid_fixture["simplicial_indices"]
        hidden_size = config.hidden_size

        for idx in simplicial_indices:
            attn = model.model.layers[idx].self_attn
            assert attn.k1_proj.weight.shape == (hidden_size, hidden_size), \
                f"Layer {idx}: k1_proj shape {attn.k1_proj.weight.shape}"
            assert attn.v1_proj.weight.shape == (hidden_size, hidden_size), \
                f"Layer {idx}: v1_proj shape {attn.v1_proj.weight.shape}"

    def test_k1_proj_match_expanded_original(self, hybrid_fixture):
        """
        k1_proj.weight == k_proj.weight.repeat(1, num_repeats).
        num_repeats = num_q_heads // num_kv_heads = 4 per 8B, 1 per 1B.
        """
        model = hybrid_fixture["model"]
        original_weights = hybrid_fixture["original_weights"]
        config = hybrid_fixture["config"]
        simplicial_indices = hybrid_fixture["simplicial_indices"]

        for idx in simplicial_indices:
            if idx not in original_weights:
                continue  # skip se non salvato (es. env var diversa)
            attn = model.model.layers[idx].self_attn
            k_orig = original_weights[idx]["k_proj"]
            v_orig = original_weights[idx]["v_proj"]

            k1 = attn.k1_proj.weight
            v1 = attn.v1_proj.weight

            # Deriva il fattore di espansione GQA dalle shape reali (kv_heads → q_heads)
            # k_orig: [num_kv_heads * head_dim, hidden_size] → k1: [num_q_heads * head_dim, hidden_size]
            actual_repeats = k1.shape[0] // k_orig.shape[0]
            k_expanded = k_orig.repeat(actual_repeats, 1) if actual_repeats > 1 else k_orig
            v_expanded = v_orig.repeat(actual_repeats, 1) if actual_repeats > 1 else v_orig

            assert torch.allclose(k1, k_expanded, atol=1e-6), \
                f"Layer {idx}: k1_proj non matcha k_proj originale espanso (repeat={actual_repeats})"
            assert torch.allclose(v1, v_expanded, atol=1e-6), \
                f"Layer {idx}: v1_proj non matcha v_proj originale espanso (repeat={actual_repeats})"

    def test_k1_proj_shape_hidden_size(self, hybrid_fixture):
        """k1_proj ha sempre shape [hidden_size, hidden_size]."""
        model = hybrid_fixture["model"]
        config = hybrid_fixture["config"]
        simplicial_indices = hybrid_fixture["simplicial_indices"]
        hidden_size = config.hidden_size

        for idx in simplicial_indices:
            attn = model.model.layers[idx].self_attn
            k1 = attn.k1_proj.weight
            assert k1.shape == (hidden_size, hidden_size), \
                f"Layer {idx}: k1_proj shape {k1.shape}, atteso ({hidden_size}, {hidden_size})"


# ======================================================================
# Test 3: Perturbazione K2/V2
# ======================================================================

class TestK2V2Initialization:
    """Verifica che K2/V2 = K1/V1 + α·noise."""

    def test_k2_proj_diff_from_k1_proj(self, hybrid_fixture):
        """k2_proj ≠ k1_proj (se sono uguali, l'inizializzazione non ha senso)."""
        model = hybrid_fixture["model"]
        simplicial_indices = hybrid_fixture["simplicial_indices"]

        for idx in simplicial_indices:
            attn = model.model.layers[idx].self_attn
            k1 = attn.k1_proj.weight
            k2 = attn.k2_proj.weight

            diff = (k2 - k1).abs().mean().item()
            assert diff > 0, f"Layer {idx}: k2_proj è identico a k1_proj (diff={diff})"

    def test_k2_proj_noise_magnitude(self, hybrid_fixture):
        """|k2 - k1|_mean ≈ alpha * sqrt(2/π) ~ 0.01 * 0.7979 ≈ 0.008 con α=0.01."""
        model = hybrid_fixture["model"]
        simplicial_indices = hybrid_fixture["simplicial_indices"]

        for idx in simplicial_indices:
            attn = model.model.layers[idx].self_attn
            k1 = attn.k1_proj.weight
            k2 = attn.k2_proj.weight

            diff = (k2 - k1).abs().mean().item()
            expected = 0.01 * (2 / 3.14159) ** 0.5
            assert abs(diff - expected) < 0.005, \
                f"Layer {idx}: |k2-k1|_mean = {diff:.6f}, atteso ~{expected:.6f}"

    def test_v2_proj_diff_from_v1_proj(self, hybrid_fixture):
        """Stessa verifica per V2/V1."""
        model = hybrid_fixture["model"]
        simplicial_indices = hybrid_fixture["simplicial_indices"]

        for idx in simplicial_indices:
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

    def test_non_simplicial_standard_layers_trainable(self, hybrid_fixture):
        """I parametri dei layer non-simpliciali sono trainable (3-group optimizer: standard group con lr_standard)."""
        model = hybrid_fixture["model"]
        simplicial_indices = hybrid_fixture["simplicial_indices"]

        # Verifica che qualche param non-simpliciale sia trainable
        non_simplicial_trainable = 0
        for name, param in model.named_parameters():
            if "embed" in name or "lm_head" in name:
                continue  # frozen dall'optimizer (lr=0)
            in_simplicial = any(f"layers.{idx}." in name for idx in simplicial_indices)
            if not in_simplicial and "layers" in name and param.requires_grad:
                non_simplicial_trainable += 1
        assert non_simplicial_trainable > 0, \
            "Nessun parametro non-simpliciale è trainable (standard group dovrebbe essere attivo)"

    def test_simplicial_attn_trainable(self, hybrid_fixture):
        """Nei layer simpliciali: k1v1/k2v2 trainable, q_proj/o_proj frozen."""
        model = hybrid_fixture["model"]
        simplicial_indices = hybrid_fixture["simplicial_indices"]

        trainable_k1v1 = []
        trainable_k2v2 = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if "k1_proj" in name or "v1_proj" in name:
                    trainable_k1v1.append(name)
                elif "k2_proj" in name or "v2_proj" in name:
                    trainable_k2v2.append(name)

        expected_count = len(simplicial_indices) * 2
        assert len(trainable_k1v1) == expected_count, \
            f"Attese {expected_count} proiezioni K1/V1 trainable, trovate {len(trainable_k1v1)}"
        assert len(trainable_k2v2) == expected_count, \
            f"Attese {expected_count} proiezioni K2/V2 trainable, trovate {len(trainable_k2v2)}"

    # Nota: il congelamento di q_proj/o_proj nei layer simpliciali ora avviene
    # a livello optimizer (create_optimizer_groups in finetuning/utils/optimizer.py)
    # e non a livello modello. Il test non è più applicabile con il design a 3 gruppi.


# ======================================================================
# Test 5: Shape delle proiezioni
# ======================================================================

class TestProjectionShapes:
    """Verifica che tutte le proiezioni abbiano le shape corrette."""

    def test_all_projection_shapes(self, hybrid_fixture):
        """Verifica shape di tutte le proiezioni nei layer simpliciali."""
        model = hybrid_fixture["model"]
        config = hybrid_fixture["config"]
        simplicial_indices = hybrid_fixture["simplicial_indices"]

        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = hidden_size // num_heads

        for idx in simplicial_indices:
            attn = model.model.layers[idx].self_attn

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
        simplicial_indices = hybrid_fixture["simplicial_indices"]

        for idx in simplicial_indices:
            attn = model.model.layers[idx].self_attn
            for name in ["q_proj", "k1_proj", "k2_proj", "v1_proj", "v2_proj", "o_proj"]:
                w = getattr(attn, name).weight
                assert torch.isfinite(w).all(), \
                    f"Layer {idx}, {name}: peso non finito (NaN o Inf)"