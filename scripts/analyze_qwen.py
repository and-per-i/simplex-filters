#!/usr/bin/env python3
"""
analyze_qwen.py — Analisi geometrica di Qwen2.5-0.5B (QKV bias).
Hardcoded: 24 layer, [9, 11, 13, 15], Gr(2,64).
Il tokenizer Qwen non ha eos_token/pad_token di default → fix esplicito.
"""

import os, sys
import torch
from transformers import AutoConfig, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.geometry.analyzer import analyze_llama_pure, summarize_results

MODEL = "Qwen/Qwen2.5-0.5B"
INDICES = [9, 11, 13, 15]
HEAD_DIM = 64  # 896 hidden / 14 heads = 64
DEVICE = "cpu"
SEQ_LEN = 256

print(f"Analisi geometrica di {MODEL}")
print(f"Layer: {INDICES}")
print(f"Device: {DEVICE}")

torch.set_num_threads(8)

# Fix esplicito tokenizer Qwen: non ha eos_token/pad_token
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
print(f"  Tokenizer pad_token: {tokenizer.pad_token}, eos_token: {tokenizer.eos_token}")

# Qwen2.5 non ha pad_token né eos_token → settiamo manualmente
if tokenizer.pad_token is None:
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.eos_token = "<|endoftext|>"

# Debug: testa un testo di Wikitext
test_text = "Beginners BBQ Class Taking Place in Missoula! Do you want to get better"
tokens = tokenizer.encode(test_text)
print(f"  Test tokenizzazione: '{test_text[:40]}...' → {len(tokens)} token")
if len(tokens) > 0:
    print(f"  Primi 3 token: {tokens[:3]} → {tokenizer.decode(tokens[:3])}")

try:
    results = analyze_llama_pure(
        model_name=MODEL,
        simplicial_indices=INDICES,
        num_analysis_batches=5,
        seq_length=SEQ_LEN,
        device=DEVICE,
        verbose=True,
    )
    summarize_results(results)
    
    print("\nRIEPILOGO COMPATTO:")
    print(f"{'Layer':>6} | {'Varianza':>10} | {'Riduzione':>10} | {'Angolo':>7}")
    print("-" * 42)
    for lidx, m in sorted(results.items()):
        v = m["geodesic_variance"]
        r = m.get("reduction_pct", 0)
        a = m.get("query_angle_from_plane_deg", 0)
        print(f"{lidx:>6} | {v:>10.4f} | {r:>9.1f}% | {a:>6.1f}°")
    print(f"\nBaseline random Gr(2,{HEAD_DIM}): 3.8507")
    
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"[ERR] Analisi Qwen fallita: {e}")