"""
Valutazione perplexity su Wikitext-2.

Confronta il modello ibrido con la baseline di LLaMA 3.1 8B.
Gate condition: Δ < 0.5 punti di perplexity.
"""

import os
import sys
import math
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Baseline note per LLaMA 3.1 8B su Wikitext-2
# (ottenute da literature, modello base senza finetuning)
LLAMA_31_8B_BASELINE_PPL = {
    "wikitext-2": 8.2,  # PPL su Wikitext-2, seq_len=512
}


@torch.no_grad()
def compute_perplexity(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    dataset,
    seq_length: int = 512,
    stride: int = 256,
    max_samples: Optional[int] = None,
    device: str = "cuda",
) -> float:
    """
    Calcola la perplexity su un dataset di test usando sliding window.
    
    La sliding window evita il problema del contesto limitato
    per sequenze più lunghe della lunghezza massima del modello.
    
    Args:
        model: modello da valutare
        tokenizer: tokenizer
        dataset: dataset HuggingFace (split di test)
        seq_length: lunghezza della finestra
        stride: stride della sliding window
        max_samples: numero massimo di campioni (None = tutti)
        device: device per il modello
    
    Returns:
        perplexity: float
    """
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    sample_count = 0
    
    for example in dataset:
        if max_samples and sample_count >= max_samples:
            break
        
        text = example["text"][:10000]  # limita a 10K caratteri per sample
        encodings = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=seq_length * 2,
        )
        input_ids = encodings.input_ids.to(device)
        
        if input_ids.size(1) < seq_length:
            continue  # skip troppo corti
        
        # Sliding window
        nll = 0.0
        tokens = 0
        
        for start in range(0, input_ids.size(1) - 1, stride):
            end = min(start + seq_length, input_ids.size(1) - 1)
            
            chunk = input_ids[:, start:end + 1]
            
            with torch.no_grad():
                outputs = model(chunk, labels=chunk)
                loss = outputs.loss
            
            # Loss media * numero di token nella finestra
            window_tokens = end - start
            nll += loss.item() * window_tokens
            tokens += window_tokens
        
        total_loss += nll
        total_tokens += tokens
        sample_count += 1
    
    # Perplexity = exp(NLL medio)
    avg_nll = total_loss / total_tokens
    perplexity = math.exp(avg_nll)
    
    return perplexity


def evaluate_perplexity(
    model_name_or_path: str,
    tokenizer_name: Optional[str] = None,
    seq_length: int = 512,
    stride: int = 256,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    max_test_samples: int = 10,
    compare_baseline: bool = True,
    baseline_key: str = "wikitext-2",
    gate_threshold: float = 0.5,
):
    """
    Valuta la perplexity su Wikitext-2 e confronta con LLaMA baseline.
    
    Args:
        model_name_or_path: modello da valutare (path o HuggingFace ID)
        tokenizer_name: tokenizer (default = model_name_or_path)
        seq_length: lunghezza della sequenza per valutazione
        stride: stride per sliding window
        dataset_name: nome dataset HuggingFace
        dataset_config: configurazione dataset
        max_test_samples: numero massimo campioni di test
        compare_baseline: se confrontare con baseline nota
        baseline_key: chiave per baseline (es. "wikitext-2")
        gate_threshold: differenza massima accettabile di PPL
    
    Returns:
        dict con risultati
    """
    if tokenizer_name is None:
        tokenizer_name = model_name_or_path
    
    # Carica tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Carica modello
    print(f"\n[eval] Caricamento modello: {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    
    # Carica dataset
    print(f"[eval] Caricamento dataset: {dataset_name}/{dataset_config}")
    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split="test",
        streaming=True,
    )
    
    # Calcola perplexity
    print(f"[eval] Calcolo perplexity (max {max_test_samples} samples)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ppl = compute_perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        seq_length=seq_length,
        stride=stride,
        max_samples=max_test_samples,
        device=device,
    )
    
    # Risultati
    results = {
        "model": model_name_or_path,
        "perplexity": round(ppl, 2),
        "seq_length": seq_length,
        "stride": stride,
        "dataset": f"{dataset_name}/{dataset_config}",
        "test_samples": max_test_samples,
    }
    
    print(f"\n[eval] Risultati:")
    print(f"  Modello: {model_name_or_path}")
    print(f"  Dataset: {dataset_name}/{dataset_config}")
    print(f"  Perplexity: {ppl:.2f}")
    
    # Confronto con baseline
    if compare_baseline and baseline_key in LLAMA_31_8B_BASELINE_PPL:
        baseline = LLAMA_31_8B_BASELINE_PPL[baseline_key]
        delta = ppl - baseline
        
        results["baseline_ppl"] = baseline
        results["delta_ppl"] = round(delta, 2)
        results["gate_passed"] = delta < gate_threshold
        
        print(f"  Baseline LLaMA 3.1 8B: {baseline:.2f}")
        print(f"  Δ PPL (ibrido - baseline): {delta:+.2f}")
        
        if delta < gate_threshold:
            print(f"  ✅ Gate PASSATO: Δ < {gate_threshold}")
        else:
            print(f"  ❌ Gate FALLITO: Δ ≥ {gate_threshold}")
            print(f"     → Tornare a Step 2 e riprovare con α o LR diversi")
    else:
        results["gate_passed"] = None
    
    return results


# Script eseguibile
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Valutazione perplexity modello ibrido")
    parser.add_argument("--model", type=str, required=True,
                        help="Path del modello o HuggingFace ID")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Tokenizer (default = model)")
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--gate", type=float, default=0.5)
    
    args = parser.parse_args()
    
    results = evaluate_perplexity(
        model_name_or_path=args.model,
        tokenizer_name=args.tokenizer,
        seq_length=args.seq_length,
        stride=args.stride,
        max_test_samples=args.max_samples,
        compare_baseline=not args.no_baseline,
        gate_threshold=args.gate,
    )
    
    # Exit code per CI/CD
    if results.get("gate_passed") is False:
        sys.exit(1)