#!/usr/bin/env python3
"""
download_c4.py — Scarica C4 in locale per training su Vast.ai (offline).
Converte da streaming IterableDataset a Dataset materializzato.
"""

import os
import argparse
import time
from datasets import load_dataset, Dataset


def main():
    parser = argparse.ArgumentParser(description="Scarica C4 per training offline")
    parser.add_argument("--num_samples", type=int, default=1_500_000,
                        help="Numero di campioni da scaricare (default: 1.5M)")
    parser.add_argument("--output", type=str, default="./data/c4_train/",
                        help="Directory di output")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Download C4: {args.num_samples} campioni in {args.output}")
    start = time.time()

    # 1. Carica in streaming
    print("  Caricamento streaming C4 train...")
    ds_stream = load_dataset("allenai/c4", "en", split="train", streaming=True)
    
    # 2. Prendi i primi N campioni
    print(f"  Prelievo {args.num_samples} campioni...")
    samples = []
    for i, example in enumerate(ds_stream):
        samples.append({"text": example["text"]})
        if (i + 1) % 100_000 == 0:
            print(f"    ...{i+1} campioni")
        if i + 1 >= args.num_samples:
            break

    # 3. Converti in Dataset materializzato
    print(f"  Conversione in Dataset ({len(samples)} campioni)...")
    ds = Dataset.from_list(samples)

    # 4. Salva su disco
    print(f"  Salvataggio in {args.output}...")
    ds.save_to_disk(args.output)

    elapsed = time.time() - start
    print(f"Fatto in {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"Dataset salvato: {len(ds)} campioni in {args.output}")
    
    # Stima token (assumendo ~300 token per campione)
    est_tokens = len(ds) * 300
    est_batches = est_tokens // (32 * 512)  # batch=32, seq=512
    print(f"Token stimati: ~{est_tokens:,} ({est_batches:,} batch da 32)")


if __name__ == "__main__":
    main()