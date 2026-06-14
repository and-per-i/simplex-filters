#!/usr/bin/env python3
"""
benchmark_kvpress.py — Benchmark eviction KV cache con kvpress.

Confronta GrassmannianPress (score ortogonale al piano medio) vs
QFilterPress (NVIDIA) vs RandomPress su LLaMA 3.2 1B.

Procedura:
1. Carica LLaMA 3.2 1B standard
2. Calcola U_mean (piano medio) via analyze_llama_pure
3. Per ogni budget [0%, 50%, 70%, 90%]:
   - Instanzia GrassmannianPress, QFilterPress, RandomPress
   - Forward prefisso (256 token) con press attiva
   - Forward suffisso (256 token) con KV ridotto → calcola PPL
4. Tabella comparativa

Usage:
    /venv/main/bin/python scripts/benchmark_kvpress.py

Dipendenze: pip install kvpress transformers datasets
"""

import os, sys, math, gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.geometry.analyzer import analyze_llama_pure
from src.kv_cache.grassmann_press import GrassmannianPress

# Cerca di importare kvpress — se non presente, usa fallback
try:
    from kvpress import QFilterPress, RandomPress
    HAVE_KVPRESS = True
except ImportError:
    HAVE_KVPRESS = False
    print("[WARN] kvpress non installata, solo GrassmannianPress disponibile")

MODEL = "meta-llama/Llama-3.2-1B"
LAYERS_ANALYZE = [8, 10, 12, 14]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
PREFIX_LEN = 256
SUFFIX_LEN = 256
SEQ_LEN = PREFIX_LEN + SUFFIX_LEN
NUM_SEQUENCES = 10
# compression_ratio per kvpress: frazione di chiavi da EVINCERE
# budget = 1 - compression_ratio (budget tradizionale: frazione da TENERE)
COMPRESSION_RATIOS = [0.0, 0.5, 0.7, 0.9]  # corrisponde a budget 100%, 50%, 30%, 10%


def compute_U_mean_llama():
    """Calcola U_mean su LLaMA puro usando analyze_llama_pure."""
    print("Calcolo U_mean su LLaMA puro...")
    results = analyze_llama_pure(
        model_name=MODEL,
        simplicial_indices=LAYERS_ANALYZE,
        num_analysis_batches=5,
        seq_length=256,
        device=DEVICE,
        verbose=False,
        dataset_name="wikitext",
        do_shuffle_test=False,
    )
    U_means = []
    for lidx in LAYERS_ANALYZE:
        if lidx in results:
            U_means.append(results[lidx]["U_mean"])
    if not U_means:
        raise RuntimeError("Nessun U_mean disponibile!")
    U_mean = torch.stack(U_means, dim=0).mean(dim=0)  # [64, 2] float32
    print(f"  U_mean: {U_mean.shape} su {U_mean.device}")
    gc.collect()
    torch.cuda.empty_cache()
    return U_mean


def get_model_and_tokenizer():
    """Carica LLaMA standard."""
    print(f"Caricamento {MODEL}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=DTYPE, device_map=DEVICE,
        attn_implementation="eager",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def run_benchmark():
    # Prepara dati
    _, tokenizer = get_model_and_tokenizer()
    
    wikitext_local = "./data/wikitext_test"
    if os.path.exists(wikitext_local):
        from datasets import load_from_disk
        ds = load_from_disk(wikitext_local)
    else:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test", streaming=True)
    
    all_tokens = []
    for example in ds:
        t = example.get("text", "")
        if not t.strip():
            continue
        tokens = tokenizer.encode(t, add_special_tokens=False)
        if len(tokens) < 2:
            continue
        all_tokens.extend(tokens)
        if len(all_tokens) >= (NUM_SEQUENCES + 1) * SEQ_LEN:
            break
    
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    # Carica U_mean e modello
    U_mean = compute_U_mean_llama()
    model, tokenizer = get_model_and_tokenizer()
    
    sequences = []
    for i in range(NUM_SEQUENCES):
        chunk = all_tokens[i * SEQ_LEN : (i + 1) * SEQ_LEN]
        if len(chunk) == SEQ_LEN:
            sequences.append(torch.tensor(chunk, dtype=torch.long))
        else:
            chunk = chunk + [0] * (SEQ_LEN - len(chunk))
            sequences.append(torch.tensor(chunk[:SEQ_LEN], dtype=torch.long))
    
    print(f"Sequenze da {SEQ_LEN} token: {len(sequences)}")
    
    # Benchmark
    headers = f"{'Budget':>8} {'Grassmann':>12} {'QFilter':>12} {'Random':>12}"
    print(f"\nBenchmark KV eviction su {MODEL} (kvpress)")
    print(f"Prefisso: {PREFIX_LEN} token | Suffisso: {SUFFIX_LEN} token")
    print(headers)
    print("-" * 48)
    
    # Per ogni compression_ratio
    for cr in COMPRESSION_RATIOS:
        budget = 1.0 - cr  # budget tradizionale per output
        print(f"\nBudget {budget*100:.0f}% (compression={cr:.0%})...")
        
        results = {"grassmann": [], "qfilter": [], "random": []}
        strategies = ["grassmann"]
        if HAVE_KVPRESS:
            strategies = ["grassmann", "qfilter", "random"]
        
        with torch.no_grad():
            for seq_idx, input_ids in enumerate(sequences):
                input_ids = input_ids.to(DEVICE).unsqueeze(0)
                prefix = input_ids[:, :PREFIX_LEN]
                suffix = input_ids[:, PREFIX_LEN:]
                
                for strategy in strategies:
                    # Crea press
                    if strategy == "grassmann":
                        press = GrassmannianPress(
                            U_mean=U_mean,
                            compression_ratio=cr,
                        )
                    elif strategy == "qfilter" and HAVE_KVPRESS:
                        press = QFilterPress(compression_ratio=cr)
                    elif strategy == "random" and HAVE_KVPRESS:
                        press = RandomPress(compression_ratio=cr)
                    else:
                        continue
                    
                    # Forward con press (eviction durante prefisso)
                    with press(model):
                        out = model(prefix, use_cache=True)
                    
                    # Forward suffisso con KV ridotto
                    out_suffix = model(suffix, past_key_values=out.past_key_values)
                    logits = out_suffix.logits
                    loss_fct = torch.nn.CrossEntropyLoss()
                    shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
                    shift_labels = suffix[:, 1:].reshape(-1)
                    loss = loss_fct(shift_logits, shift_labels)
                    
                    results[strategy].append(loss.item())
        
        # Stampa riga
        grass_avg = sum(results["grassmann"]) / len(results["grassmann"])
        grass_ppl = math.exp(grass_avg)
        
        if HAVE_KVPRESS and len(results["qfilter"]) > 0:
            qf_avg = sum(results["qfilter"]) / len(results["qfilter"])
            qf_ppl = math.exp(qf_avg)
        else:
            qf_ppl = 0.0
        
        if HAVE_KVPRESS and len(results["random"]) > 0:
            ra_avg = sum(results["random"]) / len(results["random"])
            ra_ppl = math.exp(ra_avg)
        else:
            ra_ppl = 0.0
        
        print(f"{budget*100:>6.0f}% {grass_ppl:>12.2f} {qf_ppl:>12.2f} {ra_ppl:>12.2f}")


if __name__ == "__main__":
    run_benchmark()