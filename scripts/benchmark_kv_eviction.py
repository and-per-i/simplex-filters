#!/usr/bin/env python3
"""
benchmark_kv_eviction.py — Benchmark eviction su KV cache REALE di LLaMA puro.
Nessuna conversione GramDet. Nessuna modifica architetturale.

Procedura:
1. Carica LLaMA 3.2 1B standard
2. Calcola U_mean (piano medio) via --analyze-llama
3. Per ogni sequenza: prefisso → past_key_values → eviction → suffisso → PPL
4. Strategie: qfilter (ortogonale), random, fifo

Usage:
    /venv/main/bin/python scripts/benchmark_kv_eviction.py
"""

import os, sys, math, gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.geometry.analyzer import analyze_llama_pure
from src.kv_cache.qfilter_score import qfilter_score_orthogonal, top_k_indices, random_indices

MODEL = "meta-llama/Llama-3.2-1B"
LAYERS_ANALYZE = [8, 10, 12, 14]  # layer dove calcolare U_mean
LAYERS_EVICT = [8, 10, 12, 14]    # layer dove applicare eviction
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
PREFIX_LEN = 256
SUFFIX_LEN = 256
SEQ_LEN = PREFIX_LEN + SUFFIX_LEN  # 512
NUM_SEQUENCES = 10
BUDGETS = [1.0, 0.5, 0.3, 0.1]


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
    # Media U_mean su tutti i layer
    U_means = []
    for lidx in LAYERS_ANALYZE:
        if lidx in results:
            U_means.append(results[lidx]["U_mean"])
    if not U_means:
        raise RuntimeError("Nessun U_mean disponibile!")
    U_mean = torch.stack(U_means, dim=0).mean(dim=0).to(DEVICE)
    print(f"  U_mean: {U_mean.shape} su {DEVICE}")
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


def apply_eviction_to_past(past_key_values, U_mean, budget, strategy):
    """
    Applica eviction al KV cache sui layer selezionati.
    
    past_key_values: tuple of tuple (K, V) per ogni layer
      K: [1, num_heads, seq_len, head_dim]
    
    Restituisce past_key_values modificato.
    """
    import math as _math
    
    new_past = []
    for layer_idx in range(len(past_key_values)):
        k, v = past_key_values[layer_idx]
        
        if layer_idx not in LAYERS_EVICT or budget == 1.0:
            # Layer non selezionato: KV cache intatto
            new_past.append((k, v))
            continue
        
        num_heads = k.shape[1]
        head_dim = k.shape[-1]
        seq_len = k.shape[2]
        B = max(1, int(_math.ceil(seq_len * budget)))  # arrotonda per eccesso
        
        # Appiattisci teste per scoring: [num_heads * seq_len, head_dim]
        k_flat = k.squeeze(0).transpose(0, 1).reshape(-1, head_dim)  # [H*S, d]
        
        if strategy == "qfilter":
            scores = qfilter_score_orthogonal(k_flat, U_mean)  # [H*S]
            keep_ids = top_k_indices(scores, budget)
        elif strategy == "random":
            keep_ids = random_indices(k_flat.shape[0], budget)
        elif strategy == "fifo":
            # FIFO: tieni le ultime B su H*S (le più recenti per ogni testa)
            k_flat = k.squeeze(0).transpose(0, 1)  # [H, S, d]
            max_survive = max(1, int(seq_len * budget))
            # Per ogni testa, prendi le ultime 'max_survive' posizioni
            k_new = k_flat[:, -max_survive:, :]  # [H, B', d]
            v_new = v.squeeze(0).transpose(0, 1)[:, -max_survive:, :]
            new_past.append((
                k_new.transpose(0, 1).unsqueeze(0),  # [1, H, B', d]
                v_new.transpose(0, 1).unsqueeze(0),
            ))
            continue
        
        # Ricostruisci K, V dagli keep_ids
        keep_ids = keep_ids.to(k.device)
        k_new = k_flat[keep_ids]  # [B, d]
        v_new = v.squeeze(0).transpose(0, 1).reshape(-1, head_dim)[keep_ids]
        
        # Ricostruisci: [1, H, B', d] dove B' = numero sopravvissuti
        k_new = k_new.T.unsqueeze(0)  # [1, d, B] → serve [1, H, B_per_head, d]
        # Raggruppa per testa
        k_new = k_new.view(1, num_heads, -1, head_dim)
        v_new = v_new.T.unsqueeze(0).view(1, num_heads, -1, head_dim)
        
        new_past.append((k_new, v_new))
    
    return tuple(new_past)


def run_benchmark():
    # Prima carica tokenizer (serve per concatenare i testi)
    _, tokenizer = get_model_and_tokenizer()
    
    # Carica Wikitext e concatena in token stream
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
    
    # Pulisci memoria prima di caricare modello
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    # Ora carica modello e U_mean
    U_mean = compute_U_mean_llama()
    model, tokenizer = get_model_and_tokenizer()
    
    sequences = []
    for i in range(NUM_SEQUENCES):
        chunk = all_tokens[i * SEQ_LEN : (i + 1) * SEQ_LEN]
        if len(chunk) == SEQ_LEN:
            sequences.append(torch.tensor(chunk, dtype=torch.long))
        else:
            # Padding con zeros
            chunk = chunk + [0] * (SEQ_LEN - len(chunk))
            sequences.append(torch.tensor(chunk[:SEQ_LEN], dtype=torch.long))
    
    print(f"Sequenze da {SEQ_LEN} token: {len(sequences)}")
    
    # Benchmark
    headers = f"{'Budget':>8} {'Q-filter':>12} {'Random':>12} {'FIFO':>12}"
    print(f"\nBenchmark KV cache eviction su {MODEL}")
    print(f"Prefisso: {PREFIX_LEN} token | Suffisso: {SUFFIX_LEN} token")
    print(f"Layer evict: {LAYERS_EVICT}")
    print(headers)
    print("-" * 48)
    
    total_tokens = 0
    results = {s: {"qfilter": [], "random": [], "fifo": []} for s in BUDGETS}
    
    for budget in BUDGETS:
        print(f"\nBudget {budget*100:.0f}%...")
        for seq_idx, input_ids in enumerate(sequences):
            input_ids = input_ids.to(DEVICE).unsqueeze(0)
            prefix = input_ids[:, :PREFIX_LEN]
            suffix = input_ids[:, PREFIX_LEN:]
            
            with torch.no_grad():
                # Forward prefisso con past_key_values
                outputs = model(prefix, use_cache=True)
                past = outputs.past_key_values
                
                # DynamicCache → tuple of tuple per compatibilità
                past_type_name = type(past).__name__
                if past_type_name == 'DynamicCache':
                    # DynamicCache: prova tutti i pattern di accesso noti
                    converted = False
                    
                    # Pattern 1: accesso tramite past.layers (transformers 4.37+ / 5.x)
                    if hasattr(past, 'layers') and past.layers:
                        try:
                            layer0 = past.layers[0]
                            if hasattr(layer0, 'key_cache') and hasattr(layer0, 'value_cache'):
                                # transformers 4.x (DynamicCacheLayer)
                                past = tuple((layer.key_cache, layer.value_cache) for layer in past.layers)
                                converted = True
                            elif hasattr(layer0, 'keys') and hasattr(layer0, 'values'):
                                # transformers 5.x (DynamicLayer)
                                past = tuple((layer.keys, layer.values) for layer in past.layers)
                                converted = True
                        except Exception as e:
                            print(f"  Pattern layers fallito: {e}")
                    
                    if not converted:
                        try:
                            # Pattern 2: key_cache/ value_cache come proprietà del DynamicCache
                            past = tuple(zip(past.key_cache, past.value_cache))
                            converted = True
                        except (AttributeError, TypeError):
                            pass
                    
                    if not converted:
                        try:
                            # Pattern 3: to_legacy_cache()
                            past = past.to_legacy_cache()
                            converted = True
                        except (AttributeError, TypeError):
                            pass
                    
                    if not converted:
                        try:
                            # Pattern 4: attributi privati _key_cache / _value_cache
                            past = tuple(zip(past._key_cache, past._value_cache))
                            converted = True
                        except (AttributeError, TypeError):
                            pass
                    
                    if not converted:
                        # Pattern 5: stampa diagnostica dettagliata
                        print(f"ERROR: impossibile convertire {past_type_name}", flush=True)
                        print(f"  Attributi DynamicCache: {[a for a in dir(past) if not a.startswith('__')]}", flush=True)
                        if hasattr(past, 'layers') and past.layers:
                            layer0 = past.layers[0]
                            print(f"  Tipo layer0: {type(layer0).__name__}", flush=True)
                            print(f"  Attributi layer0: {[a for a in dir(layer0) if not a.startswith('__')]}", flush=True)
                        raise RuntimeError(f"Impossibile convertire DynamicCache ({past_type_name})")
                elif hasattr(past, 'to_legacy_cache'):
                    past = past.to_legacy_cache()
                
                for strategy in ["qfilter", "random", "fifo"]:
                    if budget == 1.0 and strategy != "qfilter":
                        continue  # idem, salta
                    past_filtered = apply_eviction_to_past(past, U_mean, budget, strategy)
                    
                    # Forward suffisso con KV ridotto
                    out_suffix = model(suffix, past_key_values=past_filtered)
                    logits = out_suffix.logits
                    loss_fct = torch.nn.CrossEntropyLoss()
                    shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
                    shift_labels = suffix[:, 1:].reshape(-1)
                    loss = loss_fct(shift_logits, shift_labels)
                    
                    if strategy == "qfilter":
                        results[budget]["qfilter"].append(loss.item())
                    elif strategy == "random":
                        results[budget]["random"].append(loss.item())
                    elif strategy == "fifo":
                        results[budget]["fifo"].append(loss.item())
        
        # Stampa riga
        qf_avg = sum(results[budget]["qfilter"]) / len(results[budget]["qfilter"])
        qf_ppl = math.exp(qf_avg)
        if len(results[budget]["random"]) > 0:
            ra_avg = sum(results[budget]["random"]) / len(results[budget]["random"])
            ra_ppl = math.exp(ra_avg)
        else:
            ra_ppl = 0.0
        if len(results[budget]["fifo"]) > 0:
            fi_avg = sum(results[budget]["fifo"]) / len(results[budget]["fifo"])
            fi_ppl = math.exp(fi_avg)
        else:
            fi_ppl = 0.0
        print(f"{budget*100:>6.0f}% {qf_ppl:>12.2f} {ra_ppl:>12.2f} {fi_ppl:>12.2f}")


if __name__ == "__main__":
    run_benchmark()