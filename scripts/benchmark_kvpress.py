#!/usr/bin/env python3
"""
benchmark_kvpress.py — Benchmark eviction KV cache con kvpress.

Supporta due modalità:
- Normale: LLaMA puro, tutte le strategie via kvpress
- --gramdet: LLaMA convertito in ibrido GramDet.
  - GrassmannianPress su layer GramDet via eviction_params (fallback A)
  - QFilterPress / RandomPress su layer standard via kvpress

Usage:
    /venv/main/bin/python scripts/benchmark_kvpress.py
    /venv/main/bin/python scripts/benchmark_kvpress.py --gramdet

Dipendenze: pip install kvpress transformers datasets
"""

import os, sys, math, gc, argparse
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
INDICES = [8, 10, 12, 14]  # GramDet layer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
PREFIX_LEN = 256
SUFFIX_LEN = 256
SEQ_LEN = PREFIX_LEN + SUFFIX_LEN
NUM_SEQUENCES = 10
BUDGETS = [1.0, 0.5, 0.3, 0.1]


def compute_U_mean_llama(gramdet_mode=False):
    """Calcola U_mean su LLaMA puro o su GramDet (se gramdet_mode)."""
    if gramdet_mode:
        print("Calcolo U_mean su modello GramDet step 0...")
        # Carica e converti
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=DTYPE, device_map=DEVICE,
            attn_implementation="eager",
        )
        from src.modeling.convert_to_hybrid import convert_llama_to_hybrid
        model, _ = convert_llama_to_hybrid(
            model, simplicial_indices=INDICES,
            attention_type="gram_det", gram_window=8,
        )
        model.eval()

        from src.geometry.hooks import ActivationSaver, batch_to_planes_gram_det
        from src.geometry.grassmann import frechet_mean_planes

        num_heads = model.config.num_attention_heads
        head_dim = model.config.hidden_size // model.config.num_attention_heads

        # Carica Wikitext
        wikitext_local = "./data/wikitext_test"
        if os.path.exists(wikitext_local):
            ds = load_from_disk(wikitext_local)
        else:
            ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test", streaming=True)

        all_texts = [ex["text"] for ex in ds if ex.get("text", "").strip()]
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        tokenizer.pad_token = tokenizer.eos_token

        all_U_means = []
        for layer_idx in INDICES:
            all_U = []
            text_cursor = 0
            for _ in range(5):
                texts = []
                while len(texts) < 2 and text_cursor < len(all_texts):
                    texts.append(all_texts[text_cursor][:SEQ_LEN*4])
                    text_cursor += 1
                if not texts:
                    break
                enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=SEQ_LEN)
                input_ids = enc["input_ids"].to(DEVICE)
                saver = ActivationSaver(model, INDICES, "gram_det")
                saver.register_hooks()
                model(input_ids)
                activations = saver.get_data()
                saver.remove_hooks()
                U_list, _ = batch_to_planes_gram_det(
                    activations, layer_idx, num_heads, head_dim, DEVICE, num_pairs=500,
                )
                all_U.append(U_list)
            if all_U:
                U_layer = torch.cat(all_U, dim=0)
                U_mean, _ = frechet_mean_planes(U_layer, n_iter=10, verbose=False)
                all_U_means.append(U_mean)

        U_mean = torch.stack(all_U_means, dim=0).mean(dim=0)
        print(f"  U_mean: {U_mean.shape} su {U_mean.device}")
        gc.collect()
        torch.cuda.empty_cache()
        del model
        return U_mean

    # Modo normale: analyze_llama_pure
    print("Calcolo U_mean su LLaMA puro...")
    results = analyze_llama_pure(
        model_name=MODEL,
        simplicial_indices=INDICES,
        num_analysis_batches=5,
        seq_length=256,
        device=DEVICE,
        verbose=False,
        dataset_name="wikitext",
        do_shuffle_test=False,
    )
    U_means = []
    for lidx in INDICES:
        if lidx in results:
            U_means.append(results[lidx]["U_mean"])
    if not U_means:
        raise RuntimeError("Nessun U_mean disponibile!")
    U_mean = torch.stack(U_means, dim=0).mean(dim=0)
    print(f"  U_mean: {U_mean.shape} su {U_mean.device}")
    gc.collect()
    torch.cuda.empty_cache()
    return U_mean


def get_model_and_tokenizer(gramdet_mode=False):
    """Carica LLaMA standard, opzionalmente convertito in GramDet."""
    print(f"Caricamento {MODEL}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=DTYPE, device_map=DEVICE,
        attn_implementation="eager",
    )
    if gramdet_mode:
        from src.modeling.convert_to_hybrid import convert_llama_to_hybrid
        print("  Conversione in GramDet ibrido...")
        model, _ = convert_llama_to_hybrid(
            model, simplicial_indices=INDICES,
            attention_type="gram_det", gram_window=8,
        )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def set_eviction_params(model, U_mean, budget, strategy):
    """Imposta eviction_params sui layer GramDet."""
    for idx in INDICES:
        attn = model.model.layers[idx].self_attn
        if hasattr(attn, "eviction_params"):
            attn.eviction_params = {
                "U_mean": U_mean,
                "budget": budget,
                "strategy": strategy,
            }


def clear_eviction_params(model):
    """Rimuove eviction_params dai layer GramDet."""
    for idx in INDICES:
        attn = model.model.layers[idx].self_attn
        if hasattr(attn, "eviction_params"):
            attn.eviction_params = None


def run_benchmark(gramdet_mode=False):
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
    U_mean = compute_U_mean_llama(gramdet_mode)
    model, tokenizer = get_model_and_tokenizer(gramdet_mode)
    
    sequences = []
    for i in range(NUM_SEQUENCES):
        chunk = all_tokens[i * SEQ_LEN : (i + 1) * SEQ_LEN]
        if len(chunk) == SEQ_LEN:
            sequences.append(torch.tensor(chunk, dtype=torch.long))
        else:
            chunk = chunk + [0] * (SEQ_LEN - len(chunk))
            sequences.append(torch.tensor(chunk[:SEQ_LEN], dtype=torch.long))
    
    print(f"Sequenze da {SEQ_LEN} token: {len(sequences)}")
    
    # Header
    mode_str = "GramDet step 0" if gramdet_mode else "LLaMA puro"
    headers = f"{'Budget':>8} {'Grassmann':>12} {'QFilter':>12} {'Random':>12}"
    print(f"\nBenchmark KV eviction su {MODEL} ({mode_str}, kvpress)")
    print(f"Prefisso: {PREFIX_LEN} token | Suffisso: {SUFFIX_LEN} token")
    print(f"Layer GramDet: {INDICES}")
    print(headers)
    print("-" * 48)
    
    results = {s: {"grassmann": [], "qfilter": [], "random": []} for s in BUDGETS}
    
    for budget in BUDGETS:
        cr = 1.0 - budget  # compression_ratio per kvpress
        print(f"\nBudget {budget*100:.0f}%...")
        
        for seq_idx, input_ids in enumerate(sequences):
            input_ids = input_ids.to(DEVICE).unsqueeze(0)
            prefix = input_ids[:, :PREFIX_LEN]
            suffix = input_ids[:, PREFIX_LEN:]
            
            with torch.no_grad():
                if gramdet_mode:
                    # GramDet: confronto Grassmann vs Random su STESSI layer GramDet
                    # via eviction_params (nessun kvpress coinvolto su questi layer).
                    
                    # Strategia: Grassmann orthogonal score
                    set_eviction_params(model, U_mean, budget, "qfilter")
                    out = model(prefix, use_cache=True)
                    clear_eviction_params(model)
                    out_suffix = model(suffix, past_key_values=out.past_key_values)
                    logits = out_suffix.logits
                    loss_fct = torch.nn.CrossEntropyLoss()
                    shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
                    shift_labels = suffix[:, 1:].reshape(-1)
                    loss = loss_fct(shift_logits, shift_labels)
                    results[budget]["grassmann"].append(loss.item())
                    
                    # Strategia: Random eviction (stessi layer)
                    set_eviction_params(model, U_mean, budget, "random")
                    out = model(prefix, use_cache=True)
                    clear_eviction_params(model)
                    out_suffix = model(suffix, past_key_values=out.past_key_values)
                    logits = out_suffix.logits
                    shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
                    shift_labels = suffix[:, 1:].reshape(-1)
                    loss = loss_fct(shift_logits, shift_labels)
                    results[budget]["random"].append(loss.item())
                    
                    # QFilter non applicabile (crasha su GramDetAttention)
                    if budget == 1.0 and seq_idx == 0:
                        print(f"  [INFO] QFilter non disponibile per GramDet (attenzione non standard)")
                else:
                    # LLaMA puro: tutte le strategie via kvpress
                    press = GrassmannianPress(
                        U_mean=U_mean, compression_ratio=cr,
                    )
                    with press(model):
                        out = model(prefix, use_cache=True)
                    out_suffix = model(suffix, past_key_values=out.past_key_values)
                    logits = out_suffix.logits
                    loss_fct = torch.nn.CrossEntropyLoss()
                    shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
                    shift_labels = suffix[:, 1:].reshape(-1)
                    loss = loss_fct(shift_logits, shift_labels)
                    results[budget]["grassmann"].append(loss.item())
                    
                    # === QFilterPress (kvpress) ===
                    if HAVE_KVPRESS:
                        press = QFilterPress(compression_ratio=cr)
                        with press(model):
                            out = model(prefix, use_cache=True)
                        out_suffix = model(suffix, past_key_values=out.past_key_values)
                        logits = out_suffix.logits
                        loss_fct = torch.nn.CrossEntropyLoss()
                        shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
                        shift_labels = suffix[:, 1:].reshape(-1)
                        loss = loss_fct(shift_logits, shift_labels)
                        results[budget]["qfilter"].append(loss.item())
                    
                    # === RandomPress (kvpress) ===
                    if HAVE_KVPRESS:
                        press = RandomPress(compression_ratio=cr)
                        with press(model):
                            out = model(prefix, use_cache=True)
                        out_suffix = model(suffix, past_key_values=out.past_key_values)
                        logits = out_suffix.logits
                        loss_fct = torch.nn.CrossEntropyLoss()
                        shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
                        shift_labels = suffix[:, 1:].reshape(-1)
                        loss = loss_fct(shift_logits, shift_labels)
                        results[budget]["random"].append(loss.item())
        
        # Stampa riga
        grass_avg = sum(results[budget]["grassmann"]) / len(results[budget]["grassmann"])
        grass_ppl = math.exp(grass_avg)
        
        if HAVE_KVPRESS and len(results[budget]["qfilter"]) > 0:
            qf_avg = sum(results[budget]["qfilter"]) / len(results[budget]["qfilter"])
            qf_ppl = math.exp(qf_avg)
        else:
            qf_ppl = 0.0
        
        if HAVE_KVPRESS and len(results[budget]["random"]) > 0:
            ra_avg = sum(results[budget]["random"]) / len(results[budget]["random"])
            ra_ppl = math.exp(ra_avg)
        else:
            ra_ppl = 0.0
        
        print(f"{budget*100:>6.0f}% {grass_ppl:>12.2f} {qf_ppl:>12.2f} {ra_ppl:>12.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark KV eviction con kvpress")
    parser.add_argument("--gramdet", action="store_true", help="Usa modello GramDet ibrido (step 0)")
    args = parser.parse_args()
    run_benchmark(gramdet_mode=args.gramdet)