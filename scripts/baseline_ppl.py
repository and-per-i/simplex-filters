#!/usr/bin/env python3
"""
baseline_ppl.py — Calcola PPL baseline per qualsiasi modello su C4 o Wikitext.

Forward pass only, senza training. Utile per smoke test e per ottenere
il valore di baseline_ppl da inserire in config.yaml.

Usage:
    python scripts/baseline_ppl.py --model meta-llama/Llama-3.2-1B --dataset wikitext
    python scripts/baseline_ppl.py --model meta-llama/Llama-3.1-8B --dataset c4 --max-samples 50
    python scripts/baseline_ppl.py --model meta-llama/Llama-3.2-1B --dataset c4 --seq-length 512 --output baseline.json
"""

import argparse
import json
import math
import os
import sys
from typing import Optional

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


@torch.no_grad()
def compute_ppl(
    model_name: str,
    dataset: str = "c4",
    max_samples: int = 50,
    seq_length: int = 512,
    stride: int = 256,
    device: str = "cuda",
    output: Optional[str] = None,
) -> float:
    """
    Calcola la PPL baseline per un modello su un dataset.

    Forward pass only: carica modello, scorre max_samples batch,
    calcola NLL media, restituisce 2^NLL.

    Args:
        model_name: nome del modello su HuggingFace (es. "meta-llama/Llama-3.2-1B")
        dataset: "c4" o "wikitext"
        max_samples: numero massimo di campioni da valutare
        seq_length: lunghezza della sequenza
        device: device per il calcolo
        output: path opzionale per salvare risultati JSON

    Returns:
        PPL media
    """
    print(f"\n{'='*60}")
    print(f"  BASELINE PPL")
    print(f"  Modello: {model_name}")
    print(f"  Dataset: {dataset}")
    print(f"  Campioni: {max_samples}")
    print(f"  Seq length: {seq_length}")
    print(f"{'='*60}\n")

    # Carica modello
    print(f"Caricamento modello...", end=" ", flush=True)
    hf_token = os.environ.get("HF_TOKEN")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        token=hf_token,
    )
    model.eval()
    print("OK")

    # Carica tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    # Carica dataset
    if dataset == "c4":
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        dataset_label = "C4 (validation)"
        print(f"Dataset: {dataset_label}")
        print(f"Calcolo PPL...")

        total_nll = 0.0
        total_tokens = 0
        count = 0

        for example in ds:
            if count >= max_samples:
                break
            text = example.get("text", "")[:10000]
            if not text.strip():
                continue
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_length * 2)
            input_ids = enc["input_ids"]
            if input_ids.shape[1] < 50:
                continue
            input_ids = input_ids.to(device)

            for start in range(0, input_ids.size(1) - 1, stride):
                end = min(start + seq_length, input_ids.size(1) - 1)
                chunk = input_ids[:, start:end + 1]
                outputs = model(chunk, labels=chunk)
                window_tokens = end - start
                nll = outputs.loss.item() * window_tokens
                total_nll += nll
                total_tokens += window_tokens
            count += 1

            if count % 10 == 0:
                current_ppl = math.exp(total_nll / max(total_tokens, 1))
                print(f"  [{count}/{max_samples}] PPL corrente: {current_ppl:.2f}")

    elif dataset == "wikitext":
        # Metodo standard letteratura: concatena tutti i record in un unico stream di token
        # poi applica sliding window sullo stream intero
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test", streaming=True)
        dataset_label = "Wikitext-2 (test)"
        print(f"Dataset: {dataset_label}")
        print(f"Concatenamento token in unico stream...")

        all_tokens = []
        for example in ds:
            text = example.get("text", "")
            if not text.strip():
                continue
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) < 2:
                continue
            all_tokens.extend(tokens)
            # Limita a ~2M token per non esaurire RAM
            if len(all_tokens) > 2_000_000:
                break

        total_tokens_in_stream = len(all_tokens)
        print(f"  Token totali nello stream: {total_tokens_in_stream:,}")
        print(f"  Calcolo PPL con sliding window (seq={seq_length}, stride={stride})...")

        all_tokens = torch.tensor(all_tokens, dtype=torch.long, device=device).unsqueeze(0)

        total_nll = 0.0
        total_tokens = 0

        for start in range(0, all_tokens.size(1) - 1, stride):
            end = min(start + seq_length, all_tokens.size(1) - 1)
            chunk = all_tokens[:, start:end + 1]
            outputs = model(chunk, labels=chunk)
            window_tokens = end - start
            nll = outputs.loss.item() * window_tokens
            total_nll += nll
            total_tokens += window_tokens

            if (start // stride + 1) % 100 == 0:
                current_ppl = math.exp(total_nll / max(total_tokens, 1))
                print(f"  [{(start // stride) + 1} windows] PPL corrente: {current_ppl:.2f}")

        count = total_tokens_in_stream
    else:
        raise ValueError(f"Dataset sconosciuto: {dataset}, usa 'c4' o 'wikitext'")

    avg_nll = total_nll / max(total_tokens, 1)
    ppl = math.exp(avg_nll)

    print(f"\n  PPL finale: {ppl:.2f} (su {count} token, {total_tokens} finestre)")

    results = {
        "model": model_name,
        "dataset": dataset,
        "perplexity": round(ppl, 2),
        "samples": count,
        "tokens": total_tokens,
        "seq_length": seq_length,
    }

    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Risultati salvati in: {output}")

    return ppl


def main():
    parser = argparse.ArgumentParser(
        description="Calcola PPL baseline per qualsiasi modello su C4 o Wikitext"
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B",
                        help="Nome del modello HuggingFace (default: meta-llama/Llama-3.1-8B)")
    parser.add_argument("--dataset", type=str, default="c4",
                        choices=["c4", "wikitext"],
                        help="Dataset per la valutazione (default: c4)")
    parser.add_argument("--max-samples", type=int, default=50,
                        help="Numero massimo di campioni (default: 50)")
    parser.add_argument("--seq-length", type=int, default=512,
                        help="Lunghezza sequenza per finestra (default: 512)")
    parser.add_argument("--stride", type=int, default=256,
                        help="Stride per sliding window (default: 256)")
    parser.add_argument("--output", type=str, default=None,
                        help="Salva risultati in JSON (opzionale)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device per il calcolo (default: cuda)")

    args = parser.parse_args()

    ppl = compute_ppl(
        model_name=args.model,
        dataset=args.dataset,
        max_samples=args.max_samples,
        seq_length=args.seq_length,
        stride=args.stride,
        device=args.device,
        output=args.output,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())