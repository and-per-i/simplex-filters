#!/usr/bin/env python3
"""
benchmark_gramdet_step0.py — Benchmark eviction su GramDet step 0 (nessun training).
Carica LLaMA 3.2 1B, converte in GramDet, calcola U_mean inline, testa eviction.

Usage (su Mac, nessun proxy):
    python3 scripts/benchmark_gramdet_step0.py
"""

import os, sys, math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.modeling.convert_to_hybrid import convert_llama_to_hybrid
from src.geometry.hooks import ActivationSaver, batch_to_planes_gram_det
from src.geometry.grassmann import frechet_mean_planes
from src.geometry.plane import plane_projector_and_basis
from src.kv_cache.qfilter_score import qfilter_score_orthogonal

MODEL = "meta-llama/Llama-3.2-1B"
INDICES = [8, 10, 12, 14]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 256
NUM_BATCHES = 5
BUDGETS = [1.0, 0.5, 0.3, 0.1]


@torch.no_grad()
def compute_U_mean_inline(model, device="cuda"):
    """
    Calcola U_mean attivando i layer GramDet su Wikitext locale.
    Senza salvare/caricare checkpoint.
    """
    from datasets import load_from_disk
    
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    
    # Carica Wikitext locale
    wikitext_local = "./data/wikitext_test"
    if os.path.exists(wikitext_local):
        print(f"  Dataset da disco: {wikitext_local}")
        ds = load_from_disk(wikitext_local)
    else:
        print(f"  Dataset da HF: Salesforce/wikitext")
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test", streaming=True)
    
    # Prepara lista testi
    all_texts = []
    for example in ds:
        t = example.get("text", "")
        if t.strip():
            all_texts.append(t)
    print(f"  Testi: {len(all_texts)}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    
    all_U_means = []
    
    for layer_idx in INDICES:
        all_U = []
        text_cursor = 0
        
        for _ in range(5):  # 5 batch
            texts = []
            while len(texts) < 2 and text_cursor < len(all_texts):
                texts.append(all_texts[text_cursor][:SEQ_LEN*4])
                text_cursor += 1
            if not texts:
                break
            
            enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=SEQ_LEN)
            input_ids = enc["input_ids"].to(device)
            
            saver = ActivationSaver(model, INDICES, "gram_det")
            saver.register_hooks()
            model(input_ids)
            activations = saver.get_data()
            saver.remove_hooks()
            
            U_list, _ = batch_to_planes_gram_det(
                activations, layer_idx, num_heads, head_dim, device, num_pairs=500,
            )
            all_U.append(U_list)
        
        if all_U:
            U_layer = torch.cat(all_U, dim=0)
            U_mean, _ = frechet_mean_planes(U_layer, n_iter=10, verbose=False)
            all_U_means.append(U_mean)
    
    if not all_U_means:
        raise RuntimeError("Nessun U_mean calcolato!")
    
    U_mean = torch.stack(all_U_means, dim=0).mean(dim=0)
    print(f"  U_mean: {U_mean.shape}")
    return U_mean.to(device), tokenizer


@torch.no_grad()
def eval_ppl(model, tokenizer, U_mean, budget, strategy, dataset):
    """Calcola PPL con eviction su Wikitext."""
    total_loss = 0.0
    total_tokens = 0
    count = 0
    
    for idx in INDICES:
        attn = model.model.layers[idx].self_attn
        if hasattr(attn, "eviction_params"):
            attn.eviction_params = {
                "U_mean": U_mean,
                "budget": budget,
                "strategy": strategy,
            }
    
    for example in dataset:
        if count >= NUM_BATCHES:
            break
        text = example.get("text", "")
        if not text.strip():
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=SEQ_LEN)
        if len(tokens) < SEQ_LEN + 1:
            continue
        input_ids = torch.tensor(tokens[:SEQ_LEN], dtype=torch.long, device=DEVICE).unsqueeze(0)
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()
        total_loss += loss * (SEQ_LEN - 1)
        total_tokens += (SEQ_LEN - 1)
        count += 1
    
    for idx in INDICES:
        attn = model.model.layers[idx].self_attn
        if hasattr(attn, "eviction_params"):
            attn.eviction_params = None
    
    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss)


def main():
    print(f"Benchmark GramDet step 0 su {MODEL}, device={DEVICE}")
    
    # 1. Carica e converti
    print("Caricamento modello base...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map=DEVICE,
        attn_implementation="eager",
    )
    print("Conversione in GramDet...")
    model, _ = convert_llama_to_hybrid(
        model, simplicial_indices=INDICES,
        attention_type="gram_det", gram_window=8,
    )
    model.eval()
    
    # 2. Calcola U_mean
    print("Calcolo U_mean inline...")
    U_mean, tokenizer = compute_U_mean_inline(model, DEVICE)
    
    # 3. Dataset di validazione
    wikitext_local = "./data/wikitext_test"
    if os.path.exists(wikitext_local):
        from datasets import load_from_disk
        dataset = load_from_disk(wikitext_local)
    else:
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test", streaming=True)
    
    # 4. Benchmark
    print(f"\n{'Budget':>8} {'Q-filter':>12} {'Random':>12} {'Delta':>12}")
    print("-" * 48)
    
    for budget in BUDGETS:
        ppl_qf = eval_ppl(model, tokenizer, U_mean, budget, "qfilter", dataset)
        ppl_rand = eval_ppl(model, tokenizer, U_mean, budget, "random", dataset)
        delta = ppl_qf - ppl_rand
        print(f"{budget*100:>6.0f}% {ppl_qf:>12.2f} {ppl_rand:>12.2f} {delta:>+12.2f}")


if __name__ == "__main__":
    main()