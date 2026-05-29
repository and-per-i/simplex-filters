"""
Level 2 — Forward + Backward pass.

Richiede GPU con Triton per eseguire il kernel 2-simplicial.
Se non c'è GPU, i test vengono saltati con pytest.mark.skipif.

Verifica:
1. Forward pass senza crash
2. Shape output
3. Nessun NaN/Inf
4. K2/V2 gradienti non-nulli (kernel differenziabile)
5. K1/V1 gradienti non-nulli ma contenuti
6. Layer frozen hanno gradienti nulli
7. q_proj/o_proj frozen anche nei layer simpliciali
8. Gradient non esplodono
"""

import pytest
import torch
import torch.nn.functional as F

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from modeling.simplicial_attention import SimplicialAttention


# Salta tutti i test se non c'è CUDA o Triton
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


# Input di test comune per tutti i test
BATCH_SIZE = 2
SEQ_LENGTH = 64
VOCAB_SIZE = 128256


@pytest.fixture
def small_input(hybrid_fixture):
    """Crea un batch piccolo per il forward pass."""
    model = hybrid_fixture["model"]
    device = next(model.parameters()).device
    input_ids = torch.randint(0, 1000, (BATCH_SIZE, SEQ_LENGTH), device=device)
    return input_ids


# ======================================================================
# Forward pass
# ======================================================================

class TestForwardPass:
    """Verifiche sul forward pass."""

    @requires_gpu
    def test_forward_pass_no_crash(self, hybrid_fixture, small_input):
        """Il forward passa senza eccezioni."""
        model = hybrid_fixture["model"]
        outputs = model(small_input)
        assert outputs is not None, "Output è None"

    @requires_gpu
    def test_output_shape(self, hybrid_fixture, small_input):
        """Logits hanno shape [B, S, vocab_size]."""
        model = hybrid_fixture["model"]
        outputs = model(small_input)
        expected = (BATCH_SIZE, SEQ_LENGTH, VOCAB_SIZE)
        assert outputs.logits.shape == expected, \
            f"Attesi {expected}, ottenuti {outputs.logits.shape}"

    @requires_gpu
    def test_hidden_state_shape(self, hybrid_fixture, small_input):
        """Hidden state (prima del lm_head) ha shape [B, S, 4096]."""
        model = hybrid_fixture["model"]
        # Attiviamo il registratore di hidden states
        with torch.no_grad():
            # Prendiamo l'output dell'ultimo decoder layer
            output = model(
                small_input,
                output_hidden_states=True,
            )
            hidden = output.hidden_states[-1]
            assert hidden.shape == (BATCH_SIZE, SEQ_LENGTH, 4096), \
                f"Hidden state shape {hidden.shape}"

    @requires_gpu
    def test_no_nan_inf_output(self, hybrid_fixture, small_input):
        """Logits non hanno NaN o Inf."""
        model = hybrid_fixture["model"]
        with torch.no_grad():
            outputs = model(small_input)
        assert torch.isfinite(outputs.logits).all(), \
            "Logits contengono NaN o Inf!"

    @requires_gpu
    def test_causal_lm_output(self, hybrid_fixture, small_input):
        """Loss di cross-entropy è calcolabile (questo verifica l'intero grafo)."""
        model = hybrid_fixture["model"]
        outputs = model(small_input, labels=small_input)
        assert outputs.loss is not None, "Loss è None"
        assert torch.isfinite(outputs.loss), f"Loss non finita: {outputs.loss}"


# ======================================================================
# Backward pass — gradienti
# ======================================================================

class TestBackwardGradients:
    """Verifiche sul backward pass — il test più importante del progetto."""

    @requires_gpu
    def test_backward_no_crash(self, hybrid_fixture, small_input):
        """Backward passa senza eccezioni."""
        model = hybrid_fixture["model"]
        outputs = model(small_input, labels=small_input)
        outputs.loss.backward()
        # Successo se arriva qui senza eccezioni

    @requires_gpu
    def test_k2v2_gradients_nonzero(self, hybrid_fixture, small_input):
        """
        [TEST CRITICO] K2/V2 hanno gradienti non-nulli.
        Se questo test FALLISCE, il kernel 2-simplicial non è differenziabile
        e il finetuning non può funzionare.
        """
        model = hybrid_fixture["model"]
        simplicial_indices = hybrid_fixture["simplicial_indices"]

        # Forward + backward
        outputs = model(small_input, labels=small_input)
        outputs.loss.backward()

        for idx in simplicial_indices:
            attn = model.model.layers[idx].self_attn

            # K2
            assert attn.k2_proj.weight.grad is not None, \
                f"Layer {idx}: k2_proj.grad è None!"
            grad_k2 = attn.k2_proj.weight.grad.abs().sum().item()
            assert grad_k2 > 0, \
                f"Layer {idx}: k2_proj.grad è zero ({grad_k2})!"

            # V2
            assert attn.v2_proj.weight.grad is not None, \
                f"Layer {idx}: v2_proj.grad è None!"
            grad_v2 = attn.v2_proj.weight.grad.abs().sum().item()
            assert grad_v2 > 0, \
                f"Layer {idx}: v2_proj.grad è zero ({grad_v2})!"

            print(f"  Layer {idx}: |grad_k2|={grad_k2:.6f}, |grad_v2|={grad_v2:.6f}")

    @requires_gpu
    def test_k1v1_gradients_nonzero(self, hybrid_fixture, small_input):
        """
        K1/V1 hanno gradienti non-nulli (sono trainable, non frozen).
        Devono esistere anche se saranno controllati da lr piccolo.
        """
        model = hybrid_fixture["model"]
        simplicial_indices = hybrid_fixture["simplicial_indices"]

        outputs = model(small_input, labels=small_input)
        outputs.loss.backward()

        for idx in simplicial_indices:
            attn = model.model.layers[idx].self_attn

            assert attn.k1_proj.weight.grad is not None, \
                f"Layer {idx}: k1_proj.grad è None!"
            assert attn.v1_proj.weight.grad is not None, \
                f"Layer {idx}: v1_proj.grad è None!"

            grad_k1 = attn.k1_proj.weight.grad.abs().sum().item()
            grad_v1 = attn.v1_proj.weight.grad.abs().sum().item()
            assert grad_k1 > 0, f"Layer {idx}: k1_proj.grad è zero ({grad_k1})!"
            assert grad_v1 > 0, f"Layer {idx}: v1_proj.grad è zero ({grad_v1})!"

    @requires_gpu
    def test_frozen_params_gradients_zero(self, hybrid_fixture, small_input):
        """
        I parametri frozen (tutto fuori da K1/V1/K2/V2) hanno grad = None o 0.
        Se hanno grad non-nullo, il freeze non funziona.
        """
        model = hybrid_fixture["model"]
        simplicial_indices = hybrid_fixture["simplicial_indices"]

        outputs = model(small_input, labels=small_input)
        outputs.loss.backward()

        has_error = False
        errors = []

        for name, param in model.named_parameters():
            # Determina se è frozen
            in_simplicial = any(f"layers.{idx}." in name for idx in simplicial_indices)
            is_trainable_kv = in_simplicial and (
                "k1_proj" in name or "v1_proj" in name or
                "k2_proj" in name or "v2_proj" in name
            )

            if not is_trainable_kv:
                g = param.grad
                if g is not None and g.abs().sum().item() > 1e-10:
                    has_error = True
                    errors.append(f"{name}: grad={g.abs().sum().item():.8f}")

        if has_error:
            error_msg = "\n".join(errors[:10])
            if len(errors) > 10:
                error_msg += f"\n... e altri {len(errors) - 10} parametri"
            pytest.fail(f"Parametri frozen con grad non-nullo:\n{error_msg}")

    @requires_gpu
    def test_qproj_oproj_frozen_in_simplicial(self, hybrid_fixture, small_input):
        """q_proj e o_proj nei layer simpliciali non hanno gradienti."""
        model = hybrid_fixture["model"]
        simplicial_indices = hybrid_fixture["simplicial_indices"]

        outputs = model(small_input, labels=small_input)
        outputs.loss.backward()

        for idx in simplicial_indices:
            attn = model.model.layers[idx].self_attn

            # q_proj
            if attn.q_proj.weight.grad is not None:
                grad_q = attn.q_proj.weight.grad.abs().sum().item()
                assert grad_q < 1e-10, \
                    f"Layer {idx}: q_proj.grad = {grad_q} (dovrebbe essere ~0)"

            # o_proj
            if attn.o_proj.weight.grad is not None:
                grad_o = attn.o_proj.weight.grad.abs().sum().item()
                assert grad_o < 1e-10, \
                    f"Layer {idx}: o_proj.grad = {grad_o} (dovrebbe essere ~0)"

    @requires_gpu
    def test_gradient_not_exploding(self, hybrid_fixture, small_input):
        """Nessun gradiente è eccessivamente grande (>100)."""
        model = hybrid_fixture["model"]
        simplicial_indices = hybrid_fixture["simplicial_indices"]

        outputs = model(small_input, labels=small_input)
        outputs.loss.backward()

        max_grad = 0.0

        for idx in simplicial_indices:
            attn = model.model.layers[idx].self_attn
            for name in ["k1_proj", "k2_proj", "v1_proj", "v2_proj"]:
                w = getattr(attn, name).weight
                if w.grad is not None:
                    max_grad = max(max_grad, w.grad.abs().max().item())

        assert max_grad < 100, \
            f"Gradienti troppo grandi: max_grad = {max_grad:.2f} (soglia: 100)"
        print(f"  Max grad (tutti i layer trainable): {max_grad:.6f}")


# ======================================================================
# Forward con RoPE
# ======================================================================

class TestForwardWithRoPE:
    """Verifica che il forward funzioni con position_embeddings."""

    @requires_gpu
    def test_forward_with_rope_no_crash(self, hybrid_fixture, small_input):
        """Forward con label passa (j_ internally LLaMA usa position_embeddings)."""
        model = hybrid_fixture["model"]
        # Usando labels, il modello gestisce internamente tutto
        outputs = model(small_input, labels=small_input)
        assert torch.isfinite(outputs.loss), "Loss non finita con RoPE"

    @requires_gpu
    def test_forward_attention_mask_no_error(self, hybrid_fixture, small_input):
        """Mask di attenzione non causa errori (viene ignorata dai layer simpliciali)."""
        model = hybrid_fixture["model"]
        attention_mask = torch.ones(BATCH_SIZE, SEQ_LENGTH, dtype=torch.long, device=small_input.device)
        outputs = model(small_input, attention_mask=attention_mask, labels=small_input)
        assert outputs.loss is not None
        assert torch.isfinite(outputs.loss)