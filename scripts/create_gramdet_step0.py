#!/usr/bin/env python3
"""
create_gramdet_step0.py — Crea checkpoint step 0 per GramDet (architettura pura, nessun training).
Utile per analisi geometrica baseline pre-training.
"""

import os, sys, argparse
import torch
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.modeling.convert_to_hybrid import convert_llama_to_hybrid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--indices", type=str, default="8,10,12,14")
    parser.add_argument("--output", default="./gramdet_1b/step0")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    indices = [int(x) for x in args.indices.split(",")]
    os.makedirs(args.output, exist_ok=True)

    print(f"Caricamento {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device if args.device != "cpu" else None,
        attn_implementation="eager",
    )

    print(f"Conversione in GramDet ({indices})...")
    model, converted = convert_llama_to_hybrid(
        model, simplicial_indices=indices,
        attention_type="gram_det", gram_window=8,
    )
    print(f"  Layer convertiti: {converted}")

    state_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            state_dict[name] = param.detach().cpu()

    save_file(state_dict, os.path.join(args.output, "model.safetensors"))
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"Salvato: {args.output}/model.safetensors ({total_params:,} params)")
    print(f"Step 0 pronto per analisi: --analyze {args.output}")


if __name__ == "__main__":
    main()