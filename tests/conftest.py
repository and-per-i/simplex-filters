"""
conftest.py — Fixture condivise per tutti i test del modello ibrido.

La fixture principale:
1. Crea un modello LLaMA 3.1 8B con pesi random (nessun download di 30 GB)
2. Salva i pesi originali di k_proj, v_proj PRIMA della conversione
3. Converte in modello ibrido con convert_llama_to_hybrid()
4. Restituisce tutto (modello ibrido + pesi originali + metadati)
"""

import os
import sys
import pytest
import torch
from transformers import (
    LlamaConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

# Aggiungi src al PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from modeling.convert_to_hybrid import convert_llama_to_hybrid, freeze_parameters


# Percorso della config locale di LLaMA 3.1 8B
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'llama-3.1-8b')
MODEL_NAME = "meta-llama/Llama-3.1-8B"
SIMPLICIAL_INDICES = [16, 20, 24, 28]


@pytest.fixture(scope="session")
def config():
    """Carica la configurazione di LLaMA 3.1 8B."""
    return LlamaConfig.from_pretrained(CONFIG_PATH)


@pytest.fixture(scope="session")
def random_model(config):
    """Crea un modello LLaMA 3.1 8B con pesi random (nessun download dei pesi reali)."""
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)
    model.eval()
    return model


@pytest.fixture(scope="session")
def tokenizer():
    """Carica il tokenizer di LLaMA 3.1 8B."""
    if os.path.exists(CONFIG_PATH):
        return AutoTokenizer.from_pretrained(CONFIG_PATH)
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="session")
def original_weights(random_model):
    """
    Salva i pesi originali di k_proj e v_proj PRIMA della conversione.
    
    Serve per test_k1v1_match_expanded_original: verifica che k1_proj
    sia esattamente k_proj originale espanso 4×.
    """
    weights = {}
    for idx in SIMPLICIAL_INDICES:
        attn = random_model.model.layers[idx].self_attn
        weights[idx] = {
            "k_proj": attn.k_proj.weight.data.clone(),  # [4096, 1024]
            "v_proj": attn.v_proj.weight.data.clone(),  # [4096, 1024]
        }
    return weights


@pytest.fixture(scope="session")
def hybrid_model(random_model, original_weights):
    """
    Converte il modello random in ibrido.
    
    Dopo la conversione:
    - 4 layer con SimplicialAttention (indici 16, 20, 24, 28)
    - Il resto invariato (LlamaAttention)
    """
    model, converted = convert_llama_to_hybrid(
        random_model,
        simplicial_indices=SIMPLICIAL_INDICES,
        alpha=0.01,
        w1=32,
        w2=256,
    )
    model.eval()
    return model


@pytest.fixture(scope="session")
def hybrid_fixture(hybrid_model, tokenizer, original_weights):
    """
    Fixture principale: restituisce tutto in un unico dict.
    
    Returns:
        dict con chiavi:
        - model: modello ibrido
        - tokenizer: tokenizer
        - original_weights: pesi originali pros-pr-conversione
        - simplicial_indices: [16, 20, 24, 28]
        - config: configurazione del modello
    """
    return {
        "model": hybrid_model,
        "tokenizer": tokenizer,
        "original_weights": original_weights,
        "simplicial_indices": SIMPLICIAL_INDICES,
        "config": hybrid_model.config,
    }


def pytest_configure(config):
    """Aggiunge marker per i livelli di test."""
    config.addinivalue_line(
        "markers",
        "level_1: Sanity check strutturale (non serve GPU)"
    )
    config.addinivalue_line(
        "markers",
        "level_2: Forward + Backward pass (richiede GPU per kernel Triton)"
    )
    config.addinivalue_line(
        "markers",
        "level_3: Sanity check numerico (richiede GPU per kernel Triton)"
    )