#!/usr/bin/env python3
"""
eval_wikitext.py — Valutazione PPL su Wikitext-2 per checkpoint GramDet.

Valuta ogni checkpoint, calcola PPL su Wikitext-2 (dominio out-of-training),
stampa tabella comparativa con LLaMA base.

Usage:
    python scripts/eval_wikitext.py --checkpoints ./checkpoints/checkpoint-6000 ./checkpoints/checkpoint-8000 ./checkpoints/final
    python scripts/eval_wikitext.py --checkpoints ./checkpoints/checkpoint-6000 --max-samples 50
    python scripts/eval_wikitext.py --checkpoints ./checkpoints/*/final --llama-baseline-only
"""

import argparse
import os
import sys
import math
import json
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import glob
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from src.eval.perplexity import compute_perplexity, LLAMA_31_8B_BASELINE_PPL

# Baseline LLaMA 3.1 8B su Wikitext-2 (letteratura)
WIKITEXT2_BASELINE = LLAMA_31_8B_BASELINE_PPL.get("wikitext-2", 8.2)

RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
BOLD = "\033[1m"
NC = "\033[0m"


def load_model(ckpt_path: str, gram_window: int = 8, device: str = "cuda") -> tuple:
    """
    Carica modello dal checkpoint GramDet.
    Stessa logica di train_hybrid.py resume: carica checkpoint ignorando mismatch,
    poi converti in GramDet, poi ricarica pesi GramDet dal ckpt_state.
    """
    from safetensors.torch import load_file as safetensors_load
    from src.modeling.convert_to_hybrid import convert_llama_to_hybrid
    from src.modeling.gram_det_attention import GramDetAttention

    # 1. Carica state_dict dal checkpoint
    ckpt_state = {}
    safetensor_files = glob.glob(os.path.join(ckpt_path, "*.safetensors"))
    if not safetensor_files:
        idx_path = os.path.join(ckpt_path, "model.safetensors.index.json")
        if os.path.exists(idx_path):
            import json
            with open(idx_path) as f:
                ix = json.load(f)
            safetensor_files = list(set(ix["weight_map"].values()))
            safetensor_files = [os.path.join(ckpt_path, f) for f in safetensor_files]
    for sf in safetensor_files:
        ckpt_state.update(safetensors_load(sf))

    # 2. Carica checkpoint direttamente (lascia mismatch — verranno ricopiati)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        local_files_only=True,
        ignore_mismatched_sizes=True,
    )

    # 3. Converti in GramDet (cambia architettura layer 16,20,24,28)
    model, converted = convert_llama_to_hybrid(
        model,
        simplicial_indices=[16, 20, 24, 28],
        alpha=0.01,
        w1=32,
        w2=256,
        attention_type="gram_det",
        gram_window=gram_window,
    )

    # 4. Sovrascrivi TUTTI i pesi GramDet dal checkpoint
    #    Dopo la conversione, le shape coincidono ([4096,4096] come salvato)
    loaded = 0
    for name, param in model.named_parameters():
        if name in ckpt_state and ckpt_state[name].shape == param.shape:
            param.data.copy_(ckpt_state[name].to(param.device))
            loaded += 1
    print(f"  Caricati {loaded} pesi totali dal checkpoint (match per shape).")

    model.eval()
    return model


def get_tokenizer():
    """Carica tokenizer LLaMA."""
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_wikitext2_test(max_samples: int = 50):
    """Carica Wikitext-2 test set."""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", streaming=True)
    if max_samples:
        ds = ds.take(max_samples)
    return ds


def eval_checkpoint(
    ckpt_path: str,
    tokenizer,
    dataset,
    seq_length: int = 512,
    stride: int = 256,
    device: str = "cuda",
) -> float:
    """
    Valuta un checkpoint su Wikitext-2.
    Restituisce PPL.
    """
    print(f"  Caricamento checkpoint: {ckpt_path}...", end=" ", flush=True)
    model = load_model(ckpt_path, device=device)
    print("OK")

    ppl = compute_perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        seq_length=seq_length,
        stride=stride,
        max_samples=None,  # already limited
        device=device,
    )

    del model
    torch.cuda.empty_cache()
    return ppl


def eval_llama_base(tokenizer, dataset, seq_length: int = 512, stride: int = 256, device: str = "cuda") -> float:
    """Valuta LLaMA 3.1 8B base su Wikitext-2."""
    print(f"  Caricamento LLaMA 3.1 8B da HuggingFace...", end=" ", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    print("OK")

    ppl = compute_perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        seq_length=seq_length,
        stride=stride,
        max_samples=None,
        device=device,
    )

    del model
    torch.cuda.empty_cache()
    return ppl


def print_table(results: list):
    """Stampa tabella comparativa."""
    print(f"\n{BOLD}{'='*70}{NC}")
    print(f"{BOLD}  VALUTAZIONE WIKITEXT-2{NC}")
    print(f"{BOLD}{'='*70}{NC}")
    print(f"{'Checkpoint':<25} {'PPL':<10} {'Δ vs LLaMA':<15} {'Miglioramento':<15}")
    print(f"{'-'*25} {'-'*10} {'-'*15} {'-'*15}")

    for r in results:
        name = os.path.basename(r["checkpoint"]) if r["checkpoint"] != "LLaMA base" else "LLaMA base"
        ppl = r["ppl"]
        delta = r.get("delta", 0.0)
        improvement = r.get("improvement", "-")

        ppl_str = f"{GREEN}{ppl:.2f}{NC}" if ppl < WIKITEXT2_BASELINE else f"{YELLOW}{ppl:.2f}{NC}"
        delta_str = f"{GREEN}{delta:+.2f}{NC}" if delta < 0 else f"{RED}{delta:+.2f}{NC}"

        print(f"{name:<25} {ppl_str:<10} {delta_str:<15} {improvement:<15}")

    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Valutazione PPL su Wikitext-2 per checkpoint GramDet")
    parser.add_argument("--checkpoints", type=str, nargs="+", default=None,
                        help="Lista checkpoint da valutare (default: auto in ./checkpoints/)")
    parser.add_argument("--max-samples", type=int, default=50,
                        help="Campioni Wikitext-2 (default: 50)")
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--llama-baseline-only", action="store_true",
                        help="Valuta solo LLaMA base e esci")
    parser.add_argument("--output", type=str, default=None,
                        help="File JSON per risultati")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Baseline LLaMA su Wikitext-2: {WIKITEXT2_BASELINE:.2f}")

    # Trova checkpoint se non specificati
    ckpt_paths = args.checkpoints
    if ckpt_paths is None and not args.llama_baseline_only:
        import glob
        ckpt_paths = sorted(
            glob.glob("./checkpoints/checkpoint-*"),
            key=lambda p: int(os.path.basename(p).split("-")[-1]),
        )
        final_path = "./checkpoints/final"
        if os.path.exists(final_path):
            ckpt_paths.append(final_path)
        if not ckpt_paths:
            print(f"  {RED}[ERR]{NC} Nessun checkpoint trovato in ./checkpoints/")
            print(f"  Specifica con --checkpoints <path1> <path2> ...")
            return 1

    # Carica tokenizer e dataset una volta
    print(f"\nCaricamento tokenizer...", end=" ", flush=True)
    tokenizer = get_tokenizer()
    print("OK")

    print(f"Caricamento Wikitext-2 test fino a {args.max_samples} samples validi...", end=" ", flush=True)
    # Wikitext-2 raw ha molti campioni corti (< seq_length) che vengono saltati.
    # Filtriamo usando tokenizzatore reale per lunghezza esatta.
    min_length = args.seq_length
    raw_ds = get_wikitext2_test(max_samples=None)  # streaming
    dataset = []
    for sample in raw_ds:
        text = sample.get("text", "")
        enc = tokenizer(text, truncation=False)
        if len(enc["input_ids"]) < min_length:
            continue
        dataset.append(sample)
        if len(dataset) >= args.max_samples:
            break
    print(f"OK ({len(dataset)} campioni validi >= {min_length} token)")

    results = []

    # Baseline LLaMA
    print(f"\n{BOLD}Baseline LLaMA 3.1 8B{NC}")
    ppl_llama = eval_llama_base(tokenizer, dataset, args.seq_length, args.stride, device)
    results.append({
        "checkpoint": "LLaMA base",
        "ppl": round(ppl_llama, 2),
        "delta": 0.0,
        "improvement": "-",
    })
    print(f"  PPL: {ppl_llama:.2f}\n")

    if args.llama_baseline_only:
        print(f"  Baseline: {ppl_llama:.2f}")
        return 0

    # Valuta ogni checkpoint
    print(f"{BOLD}Valutazione checkpoint:{NC}\n")
    first_ppl = ppl_llama

    for ckpt in ckpt_paths:
        if not os.path.exists(ckpt):
            print(f"  {YELLOW}[WARN]{NC} Checkpoint {ckpt} non trovato. Salto.")
            continue

        ppl = eval_checkpoint(ckpt, tokenizer, dataset, args.seq_length, args.stride, device)
        delta = ppl - first_ppl
        improvement_pct = ((first_ppl - ppl) / first_ppl) * 100 if first_ppl > 0 else 0.0
        improvement_str = f"{GREEN}{improvement_pct:.1f}%{NC}" if improvement_pct > 0 else f"{RED}{improvement_pct:.1f}%{NC}"

        results.append({
            "checkpoint": ckpt,
            "ppl": round(ppl, 2),
            "delta": round(delta, 2),
            "improvement_pct": round(improvement_pct, 1),
        })

        print(f"  PPL: {ppl:.2f} (Δ vs LLaMA: {delta:+.2f}, miglioramento: {improvement_pct:.1f}%)")

    # Tabella
    print_table(results)

    # Trova miglior checkpoint
    best = min(results[1:], key=lambda r: r["ppl"]) if len(results) > 1 else None
    if best:
        name = os.path.basename(best["checkpoint"])
        print(f"  {BOLD}🏆 Miglior checkpoint per la tesi:{NC} {name} (PPL: {best['ppl']:.2f})")
        print(f"  Miglioramento vs LLaMA base: {best['improvement_pct']:.1f}%\n")

    # Salva risultati
    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "baseline_ppl": round(first_ppl, 2),
                "checkpoints": results,
            }, f, indent=2)
        print(f"  Risultati salvati in: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())