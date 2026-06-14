#!/usr/bin/env python3
"""
analyze_olmo.py — Analisi geometrica di OLMo 1B (QK-norm).
Usa la versione HF-native OLMo-1B-0724-hf (non richiede hf_olmo).
Hardcoded: 16 layer, [5, 7, 9, 11], Gr(2,64).
"""

import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.geometry.analyzer import analyze_llama_pure, summarize_results

MODEL = "allenai/OLMo-1B-0724-hf"
INDICES = [5, 7, 9, 11]  # ~1/4, 2/4, 3/4 dei 16 layer totali
HEAD_DIM = 64  # 2048 hidden / 32 heads = 64
DEVICE = "cpu"
SEQ_LEN = 256

print(f"Analisi geometrica di {MODEL}")
print(f"Layer: {INDICES}")
print(f"Device: {DEVICE}")

torch.set_num_threads(8)

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
    
    # Print summary line
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
    print(f"[ERR] Analisi OLMo fallita: {e}")