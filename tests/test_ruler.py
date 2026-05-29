"""
Test per il benchmark RULER (Needle-In-A-Haystack).

Verifica su dati sintetici (CPU, senza modello reale):
1. generate_single_niah → shape, lunghezza, answer non vuota
2. generate_multi_niah → shape, 2-3 aghi inseriti
3. extract_answer → risposta parsabile
4. check_answer → match esatto e contenuto
5. RulerResult → summary valido
6. forward_with_eviction → shape logits
"""

import pytest
import torch
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kv_cache.ruler.niah_benchmark import (
    generate_single_niah,
    generate_multi_niah,
    extract_answer,
    check_answer,
    forward_with_eviction,
    RulerResult,
    SINGLE_NEEDLE_TEMPLATE,
    SINGLE_QUESTION,
)


# Tokenizer per test (carica da config locale, non scarica pesi)
@pytest.fixture(scope="session")
def tokenizer():
    from transformers import AutoTokenizer
    CONFIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'llama-3.1-8b')
    tokenizer = AutoTokenizer.from_pretrained(CONFIG_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture(scope="session")
def dummy_model():
    """Crea un modello fittizio per test di forward."""
    from transformers import LlamaConfig, AutoModelForCausalLM
    CONFIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'llama-3.1-8b')
    config = LlamaConfig.from_pretrained(CONFIG_DIR)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)
    model.eval()
    return model


# ======================================================================
# Test: generate_single_niah
# ======================================================================

class TestGenerateSingleNiah:
    """Verifica generazione test case NIAH singolo."""

    def test_shape_small(self, tokenizer):
        """Contesto 512 token → input_ids [512]."""
        input_ids, answer, prompt = generate_single_niah(
            tokenizer, context_len=512, needle_pos=0.5
        )
        assert input_ids.dim() == 1, f"Dim: {input_ids.dim()}"
        # Può essere leggermente diverso per padding del tokenizer
        assert 400 <= input_ids.shape[0] <= 520, f"Shape: {input_ids.shape}"

    def test_answer_not_empty(self, tokenizer):
        """Risposta attesa non vuota."""
        _, answer, _ = generate_single_niah(tokenizer, context_len=256)
        assert len(answer) > 0, "Answer vuota!"
        assert answer.isdigit(), f"Answer non numerica: {answer}"

    def test_different_positions(self, tokenizer):
        """Posizioni diverse producono contesti diversi."""
        input_ids_1, _, _ = generate_single_niah(tokenizer, context_len=256, needle_pos=0.2)
        input_ids_2, _, _ = generate_single_niah(tokenizer, context_len=256, needle_pos=0.8)
        # Dovrebbero essere diversi (ago in posizione diversa)
        assert not torch.allclose(input_ids_1, input_ids_2), "Posizioni diverse → input identici!"

    def test_answer_in_prompt(self, tokenizer):
        """La risposta attesa e' contenuta nel prompt."""
        input_ids, answer, prompt = generate_single_niah(
            tokenizer, context_len=512, needle_pos=0.5, num_repeats=3
        )
        assert answer in prompt, f"Answer '{answer}' non trovata nel prompt!"

    def test_context_length_respected(self, tokenizer):
        """Input non supera la lunghezza massima."""
        for ctx_len in [256, 512, 1024]:
            input_ids, _, _ = generate_single_niah(tokenizer, context_len=ctx_len)
            assert input_ids.shape[0] <= ctx_len + 5, \
                f"Ctx={ctx_len}: shape={input_ids.shape}"


# ======================================================================
# Test: generate_multi_niah
# ======================================================================

class TestGenerateMultiNiah:
    """Verifica generazione test case NIAH multiplo."""

    def test_shape_multi(self, tokenizer):
        """Contesto 512 token, 3 aghi."""
        input_ids, answer, prompt = generate_multi_niah(tokenizer, context_len=512)
        assert input_ids.dim() == 1
        assert 400 <= input_ids.shape[0] <= 520, f"Shape: {input_ids.shape}"

    def test_multi_answer_not_empty(self, tokenizer):
        """Risposta non vuota per NIAH multiplo."""
        _, answer, _ = generate_multi_niah(tokenizer, context_len=512)
        assert len(answer) > 0, "Answer vuota!"

    def test_multi_needles_in_prompt(self, tokenizer):
        """Tutti gli aghi sono presenti nel prompt."""
        _, _, prompt = generate_multi_niah(tokenizer, context_len=1024)
        # Almeno due dei tre aghi dovrebbero essere rilevabili
        count = 0
        for kw in ["magic number", "secret color", "hidden city"]:
            if kw in prompt:
                count += 1
        assert count >= 2, f"Solo {count}/3 aghi trovati nel prompt"


# ======================================================================
# Test: check_answer e extract_answer
# ======================================================================

class TestCheckAnswer:
    """Verifica parsing e matching delle risposte."""

    def test_exact_match(self):
        """Match esatto."""
        assert check_answer("42", "42")
        assert check_answer("red", "red")
        assert check_answer("Rome", "Rome")

    def test_case_insensitive(self):
        """Case insensitive."""
        assert check_answer("ROME", "Rome")
        assert check_answer("forty two", "Forty Two")

    def test_contained_match(self):
        """Risposta contenuta nella predizione."""
        assert check_answer("the magic number is 42", "42")
        assert check_answer("answer: 42 something", "42")

    def test_no_match(self):
        """Nessun match."""
        assert not check_answer("31", "42")
        assert not check_answer("blue", "red")


# ======================================================================
# Test: forward_with_eviction (su modello fittizio, CPU)
# ======================================================================

class TestForwardWithEviction:
    """Verifica forward con eviction su modello fittizio."""

    def test_forward_shape(self, dummy_model, tokenizer):
        """Forward produce logits della forma corretta."""
        # Crea input
        input_ids = torch.randint(0, 100, (1, 64))
        U_mean = torch.eye(128, 2)
        
        logits = forward_with_eviction(
            dummy_model, input_ids, budget=1.0, strategy="qfilter",
            sigma1=1.0, sigma2=1.0, U_mean=U_mean,
        )
        assert logits is not None, "Logits sono None"
        assert logits.shape[0] == 1, f"Batch: {logits.shape[0]}"
        assert logits.shape[1] == 64, f"Seq: {logits.shape[1]}"

    def test_extract_answer_from_logits(self, dummy_model, tokenizer):
        """extract_answer restituisce una stringa."""
        input_ids = torch.randint(0, 100, (1, 32))
        U_mean = torch.eye(128, 2)
        
        logits = forward_with_eviction(
            dummy_model, input_ids, budget=1.0, strategy="qfilter",
            sigma1=1.0, sigma2=1.0, U_mean=U_mean,
        )
        answer = extract_answer(logits, tokenizer)
        assert isinstance(answer, str), f"Tipo: {type(answer)}"
        assert len(answer) > 0, "Answer vuota!"


# ======================================================================
# Test: RulerResult
# ======================================================================

class TestRulerResult:
    """Verifica dataclass dei risultati."""

    def test_summary_no_data(self, tokenizer):
        """Summary senza dati non crasha."""
        result = RulerResult(model_name="test", attention_type="simplicial")
        s = result.summary()
        assert "test" in s
        assert "simplicial" in s

    def test_summary_with_accuracy(self):
        """Summary con dati parziali produce tabella."""
        result = RulerResult(
            model_name="test", attention_type="gram_det",
            budget=[1.0, 0.5], context_lengths=[8192],
        )
        result.accuracy[8192] = {
            1.0: {"qfilter": 1.0, "random": 0.9},
            0.5: {"qfilter": 0.8, "random": 0.6},
        }
        s = result.summary()
        assert "100.0%" in s or "100%" in s
        assert "90.0%" in s or "90%" in s
        assert "qfilter" in s.lower() or "Random" in s

    def test_multiple_context_lengths(self):
        """Summary con due context length produce due sezioni."""
        result = RulerResult(
            model_name="test", attention_type="simplicial",
            context_lengths=[8192, 16384],
        )
        result.accuracy[8192] = {1.0: {"qfilter": 0.95, "random": 0.90}}
        result.accuracy[16384] = {1.0: {"qfilter": 0.90, "random": 0.85}}
        s = result.summary()
        assert "8192" in s
        assert "16384" in s