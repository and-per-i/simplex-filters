#!/usr/bin/env python3
"""
diagnose_checkpoint.py — Diagnostica rapida su checkpoint GramDet.

Tre sezioni indipendenti:
  1. Distribuzione attenzione (mean pre-softmax, entropia, max weight, Gini)
  2. Test di generazione (3 prompt italiani)
  3. Grad norm (forward + backward su micro-batch)

Usage:
    python scripts/diagnose_checkpoint.py --ckpt ./checkpoints/gram_det/checkpoint-2000
    python scripts/diagnose_checkpoint.py --ckpt ./checkpoints/gram_det/checkpoint-2000 --model meta-llama/Llama-3.2-1B --simplicial-indices 8,10,12,14
"""

import argparse
import gc
import math
import os
import sys
import json
import glob
from typing import Optional

import torch
import torch.nn.functional as F

# Aggiungi root del progetto al path (serve per importare src/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ==========================================================================
# ANSI colors
# ==========================================================================
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
BLUE = "\033[0;34m"
BOLD = "\033[1m"
NC = "\033[0m"


def _flag(text: str, s: float, green: float, yellow: float) -> str:
    """Colora il flag in base al valore soglia."""
    if s <= green:
        return f"{GREEN}{BOLD}✅{NC} {text}"
    elif s <= yellow:
        return f"{YELLOW}🟡{NC} {text}"
    else:
        return f"{RED}🔴{NC} {text}"


# ==========================================================================
# Helper: trova ultimo checkpoint in una directory
# ==========================================================================
def _find_latest_checkpoint(base_dir: str) -> Optional[str]:
    """Trova l'ultimo checkpoint nella directory es: ./checkpoints/gram_det/."""
    ckpt_dirs = sorted(glob.glob(os.path.join(base_dir, "checkpoint-*")))
    if not ckpt_dirs:
        return None
    return ckpt_dirs[-1]


# ==========================================================================
# Sezione 1: Distribuzione attenzione
# ==========================================================================
def _install_hooks(model, indices: list) -> list[dict]:
    """
    Installa hook forward su ogni layer GramDet per catturare scores e attn_weights.
    Restituisce una lista di dict dove conservare i dati di ogni layer.
    """
    hook_data = []

    for layer_idx in indices:
        attn_module = model.model.layers[layer_idx].self_attn
        data = {"layer": layer_idx, "scores": [], "attn_weights": []}
        hook_data.append(data)

        original_forward = attn_module._forward_gram_det

        def make_hook(data_ref):
            def hooked_forward(self, x, return_weights=False):
                """Replica _forward_gram_det e cattura scores + attn_weights."""
                B, N, D = x.shape
                H = self.n_heads
                d = self.head_dim
                W = self.W

                q = self.q_proj(x).view(B, N, H, d).transpose(1, 2)
                k = self.k_proj(x).view(B, N, H, d).transpose(1, 2)
                v = self.v_proj(x).view(B, N, H, d).transpose(1, 2)
                q = F.normalize(q, p=2, dim=-1, eps=1e-8)
                k = F.normalize(k, p=2, dim=-1, eps=1e-8)

                k_pad = F.pad(k, (0, 0, W, W))
                v_pad = F.pad(v, (0, 0, W, W))
                win_idx = (
                    torch.arange(N, device=x.device)[:, None]
                    + torch.arange(2 * W + 1, device=x.device)[None, :]
                )
                k_windows = k_pad[:, :, win_idx, :]
                v_windows = v_pad[:, :, win_idx, :]
                pi = self.pair_indices
                k1 = k_windows[:, :, :, pi[:, 0], :]
                k2 = k_windows[:, :, :, pi[:, 1], :]
                v1 = v_windows[:, :, :, pi[:, 0], :]
                v2 = v_windows[:, :, :, pi[:, 1], :]

                q_exp = q.unsqueeze(-2)
                qq = (q * q).sum(dim=-1)
                k1k1 = (k1 * k1).sum(dim=-1)
                k2k2 = (k2 * k2).sum(dim=-1)
                qk1 = (q_exp * k1).sum(dim=-1)
                qk2 = (q_exp * k2).sum(dim=-1)
                k1k2 = (k1 * k2).sum(dim=-1)

                term1 = qq.unsqueeze(-1) * (k1k1 * k2k2 - k1k2.pow(2))
                term2 = qk1 * (qk1 * k2k2 - k1k2 * qk2)
                term3 = qk2 * (qk1 * k1k2 - k1k1 * qk2)
                scores = (term1 - term2 + term3) * self.scaling

                data_ref["scores"].append(scores.detach().float().cpu())

                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = self.dropout(attn_weights)

                data_ref["attn_weights"].append(attn_weights.detach().float().cpu())

                v_hadamard = v1 * v2
                output = (attn_weights.unsqueeze(-1) * v_hadamard).sum(dim=-2)

                output = self.o_proj(output.transpose(1, 2).reshape(B, N, H * d))
                return output

            return hooked_forward

        attn_module._forward_gram_det = make_hook(data).__get__(attn_module, type(attn_module))

    return hook_data


def _gini(probs: torch.Tensor) -> torch.Tensor:
    """
    Gini coefficient normalizzato per distribuzioni discrete.
    0 = uniforme, 1 = one-hot.
    """
    n = probs.shape[-1]
    g = (probs.pow(2).sum(dim=-1) * n - 1.0) / (n - 1.0)
    return g.clamp(0, 1).mean()


def section_attention(
    model: torch.nn.Module,
    tokenizer,
    model_name: str,
    simplicial_indices: list,
    seq_length: int = 512,
    num_batches: int = 2,
    device: str = "cuda",
) -> int:
    """
    Sezione 1: analisi distribuzione attenzione GramDet.
    """
    print(f"\n{BOLD}{'='*60}{NC}")
    print(f"{BOLD}  SEZIONE 1: DISTRIBUZIONE ATTENZIONE GRAMDET{NC}")
    print(f"{BOLD}{'='*60}{NC}\n")

    hook_data = _install_hooks(model, simplicial_indices)

    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        total_tokens = 0
        for batch_idx in range(num_batches):
            batch_ids = []
            for _ in range(1):
                sample = next(iter(ds))
                tokens = tokenizer(
                    sample["text"],
                    truncation=True,
                    max_length=seq_length,
                    return_tensors="pt",
                )["input_ids"]
                if tokens.shape[-1] < 10:
                    continue
                batch_ids.append(tokens[0, :seq_length])
            if not batch_ids:
                continue
            x = torch.stack(batch_ids, dim=0).to(device)
            with torch.no_grad():
                model(input_ids=x)
            total_tokens += x.numel()
    except Exception as e:
        print(f"  {RED}[ERR]{NC} Dataset C4 non disponibile: {e}")
        print(f"  {YELLOW}[INFO]{NC} Genero dati sintetici per il test...")
        x = torch.randint(0, 32000, (2, seq_length), device=device)
        with torch.no_grad():
            model(input_ids=x)

    print(f"  Token processati: ~{total_tokens:,}\n")

    all_entropies = []
    all_max_weights = []
    all_ginis = []
    all_pre_softmax_means = []

    for data in hook_data:
        layer_idx = data["layer"]
        scores = torch.cat(data["scores"], dim=0)
        attn = torch.cat(data["attn_weights"], dim=0)
        b, h, n, p = scores.shape

        pre_softmax_mean = scores.abs().mean().item()
        entropy = (-attn * torch.log(attn.clamp(min=1e-8))).sum(dim=-1)
        entropy_mean = entropy.mean().item()
        entropy_max = math.log(p)
        max_weight = attn.max(dim=-1).values.mean().item()
        gini = _gini(attn).item()

        all_entropies.append(entropy_mean)
        all_max_weights.append(max_weight)
        all_ginis.append(gini)
        all_pre_softmax_means.append(pre_softmax_mean)

        print(f"  {BOLD}Layer {layer_idx}{NC} (P={p})")
        print(f"    Pre-softmax mean:  {pre_softmax_mean:.6f}  "
              f"{_flag('', pre_softmax_mean, 10.0, 100.0)}")
        print(f"    Entropia:          {entropy_mean:.4f} / {entropy_max:.2f}  "
              f"{_flag('', entropy_max - entropy_mean, entropy_max - 1.0, entropy_max - 0.5)}")
        print(f"    Max weight medio:   {max_weight:.6f}  "
              f"{'🔴 one-hot!' if max_weight > 0.9 else '🟡 concentrato' if max_weight > 0.5 else '✅ sano'}")
        print(f"    Gini:              {gini:.4f}  "
              f"{'🔴 collassato' if gini > 0.8 else '🟡 moderato' if gini > 0.5 else '✅ sano'}")
        print()

    avg_maxw = sum(all_max_weights) / len(all_max_weights)
    avg_gini = sum(all_ginis) / len(all_ginis)

    if avg_maxw > 0.9 or avg_gini > 0.8:
        print(f"  {RED}{BOLD}🔴 COLLASSO ATTENZIONE DETECTATO{NC}")
        return 1
    elif avg_maxw > 0.5:
        print(f"  {YELLOW}🟡 ATTENZIONE CONCENTRATA (ma non collassata){NC}")
        return 1
    else:
        print(f"  {GREEN}{BOLD}✅ DISTRIBUZIONE ATTENZIONE SANA{NC}")
        return 0


# ==========================================================================
# Sezione 2: Test di generazione
# ==========================================================================
PROMPTS_ITALIAN = [
    ("capitale", "La capitale d'Italia è"),
    ("pitagora", "Il teorema di Pitagora afferma che"),
    ("gatto", "Una volta un gatto"),
    ("sole", "Il sole sorge a est e tramonta a"),
    ("acqua", "L'acqua bolle a"),
]


def section_generation(
    model: torch.nn.Module,
    tokenizer,
    device: str = "cuda",
    temperature: float = 0.8,
    max_new_tokens: int = 30,
) -> int:
    """
    Sezione 2: test di generazione su 5 prompt italiani.
    """
    print(f"\n{BOLD}{'='*60}{NC}")
    print(f"{BOLD}  SEZIONE 2: GENERAZIONE{NC}")
    print(f"{BOLD}{'='*60}{NC}\n")

    model.eval()
    n_good = 0

    for name, prompt in PROMPTS_ITALIAN:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = input_ids
            tokens_generated = []
            for _ in range(max_new_tokens):
                outputs = model(output_ids)
                logits = outputs.logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                output_ids = torch.cat([output_ids, next_id], dim=-1)
                token = tokenizer.decode(next_id[0], skip_special_tokens=True)
                tokens_generated.append(token)
                if next_id.item() == tokenizer.eos_token_id:
                    break

        generated = ''.join(tokens_generated)
        tokens_set = set(tokenizer.encode(generated) if generated.strip() else [])
        if len(tokens_set) <= 1 and len(generated) > 10:
            status = f"{RED}🔴 COLLASSO{NC}"
        elif len(tokens_set) <= 3 and len(generated) > 10:
            status = f"{YELLOW}🟡 RIDONDANTE{NC}"
        else:
            status = f"{GREEN}✅ OK{NC}"
            n_good += 1

        print(f"  {BOLD}{name.upper():>10}{NC}: \"{prompt}{generated}\"  {status}")

    if n_good >= 4:
        print(f"  {GREEN}{BOLD}✅ MODELLO SANO{NC}")
        return 0
    elif n_good >= 2:
        print(f"  {YELLOW}🟡 MODELLO PARZIALMENTE COLLASSATO{NC}")
        return 1
    else:
        print(f"  {RED}{BOLD}🔴 MODELLO COLLASSATO{NC}")
        return 2


# ==========================================================================
# Sezione 3: Grad norm
# ==========================================================================
def section_grad(
    model: torch.nn.Module,
    tokenizer,
    simplicial_indices: list,
    seq_length: int = 128,
    device: str = "cuda",
) -> int:
    """
    Sezione 3: forward + backward su micro-batch, stampa grad norm L2.
    """
    print(f"\n{BOLD}{'='*60}{NC}")
    print(f"{BOLD}  SEZIONE 3: GRAD NORM (FORWARD + BACKWARD){NC}")
    print(f"{BOLD}{'='*60}{NC}\n")

    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-4,
    )

    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        sample = next(iter(ds))
        tokens = tokenizer(sample["text"], truncation=True, max_length=seq_length, return_tensors="pt")
        input_ids = tokens["input_ids"].to(device)
        labels = input_ids.clone()
    except Exception as e:
        print(f"  {YELLOW}[WARN]{NC} Dataset non disponibile: {e}")
        print(f"  {YELLOW}[INFO]{NC} Genero dati sintetici...")
        input_ids = torch.randint(0, 32000, (1, seq_length), device=device)
        labels = input_ids.clone()

    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()

    print(f"  Loss: {loss.item():.4f}\n")

    n_zeros = 0
    n_exploded = 0
    n_ok = 0

    for layer_idx in simplicial_indices:
        attn = model.model.layers[layer_idx].self_attn
        print(f"  {BOLD}Layer {layer_idx}{NC}")
        for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            param = getattr(attn, name).weight
            if param.grad is None:
                grad_norm = 0.0
            else:
                grad_norm = param.grad.norm().item()

            if grad_norm == 0.0:
                flag = f"{RED}🔴 CONGELATO{NC}"
                n_zeros += 1
            elif grad_norm > 100.0:
                flag = f"{RED}🔴 ESPLOSO ({grad_norm:.2f}){NC}"
                n_exploded += 1
            else:
                flag = f"{GREEN}{grad_norm:.6f}{NC}"
                n_ok += 1
            print(f"    {name}: {flag}")
        print()

    optimizer.zero_grad()
    model.eval()

    if n_exploded > 0:
        print(f"  {RED}{BOLD}🔴 INSTABILITA' GRADIENTI{NC}")
        return 1
    elif n_zeros > 0:
        print(f"  {YELLOW}🟡 ALCUNI GRADIENTI ZERO{NC}")
        return 1
    else:
        print(f"  {GREEN}{BOLD}✅ GRADIENTI SANI{NC}")
        return 0


# ==========================================================================
# Carica modello + tokenizer
# ==========================================================================
def load_model_from_ckpt(
    ckpt_path: Optional[str],
    gram_window: int,
    model_name: str,
    simplicial_indices: list,
    device: str,
) -> tuple:
    """
    Carica modello dal checkpoint (con pesi GramDet sovrascritti),
    OPPURE carica modello fresco da HuggingFace se ckpt_path e' None.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.modeling.convert_to_hybrid import convert_llama_to_hybrid

    if ckpt_path is None:
        print(f"\n  {BOLD}Nessun checkpoint specificato.{NC}")
        print(f"  Carico {model_name} fresco da HuggingFace e converto al volo con gram_window={gram_window}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="eager",
        )
        model.train()

        model, converted = convert_llama_to_hybrid(
            model,
            simplicial_indices=simplicial_indices,
            alpha=0.01,
            w1=32,
            w2=256,
            attention_type="gram_det",
            gram_window=gram_window,
        )
        print(f"  Layer convertiti (da zero): {converted}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    print(f"\n  {BOLD}Caricamento modello da checkpoint:{NC} {ckpt_path}")

    from safetensors.torch import load_file as safetensors_load
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

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
    )
    model.train()

    loaded_match = 0
    for name, param in model.named_parameters():
        if name in ckpt_state and ckpt_state[name].shape == param.shape:
            param.data.copy_(ckpt_state[name].to(param.device))
            loaded_match += 1
    print(f"  Caricati {loaded_match} pesi dal checkpoint (match per shape).")

    model, converted = convert_llama_to_hybrid(
        model,
        simplicial_indices=simplicial_indices,
        alpha=0.01,
        w1=32,
        w2=256,
        attention_type="gram_det",
        gram_window=gram_window,
    )
    print(f"  Layer convertiti: {converted}")

    loaded_gram = 0
    for name, param in model.named_parameters():
        if name in ckpt_state and any(f"layers.{i}." in name for i in simplicial_indices):
            if ckpt_state[name].shape == param.shape:
                param.data.copy_(ckpt_state[name].to(param.device))
                loaded_gram += 1
    print(f"  Caricati {loaded_gram} pesi GramDet dal checkpoint.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ==========================================================================
# Main
# ==========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Diagnostica su checkpoint GramDet"
    )
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path al checkpoint (default: ultimo in ./checkpoints/gram_det/)")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B",
                        help="Nome del modello HuggingFace (default: meta-llama/Llama-3.1-8B)")
    parser.add_argument("--simplicial-indices", type=str, default="16,20,24,28",
                        help="Indici layer simpliciali, separati da virgola (default: 16,20,24,28)")
    parser.add_argument("--gram-window", type=int, default=8,
                        help="Half-window per test (default: 8)")
    parser.add_argument("--section", type=str, default="all",
                        choices=["all", "attention", "generation", "grad"],
                        help="Sezione da eseguire (default: all)")
    parser.add_argument("--seq-length", type=int, default=512,
                        help="Sequenza per forward pass (default: 512)")
    args = parser.parse_args()

    simplicial_indices = [int(x) for x in args.simplicial_indices.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    print(f"  Modello: {args.model}")
    print(f"  Indici simpliciali: {simplicial_indices}")
    print(f"  Gram window: {args.gram_window}")
    print(f"  Sezione: {args.section}")

    ckpt_path = args.ckpt
    if ckpt_path is not None:
        if not os.path.exists(ckpt_path):
            print(f"  {YELLOW}[WARN]{NC} Checkpoint {ckpt_path} non trovato. Cerco l'ultimo...")
            ckpt_path = _find_latest_checkpoint("./checkpoints/gram_det/")
            if ckpt_path is None:
                ckpt_path = _find_latest_checkpoint("./gram_det/")
            if ckpt_path is not None:
                print(f"  Checkpoint auto-rilevato: {ckpt_path}")
        else:
            print(f"  Checkpoint: {ckpt_path}")
    else:
        print(f"  {YELLOW}[INFO]{NC} Nessun checkpoint specificato. Carico {args.model} fresco da HuggingFace...")
        print(f"  La conversione al volo creerà layer GramDet con gram_window={args.gram_window}.")

    model, tokenizer = load_model_from_ckpt(ckpt_path, args.gram_window, args.model, simplicial_indices, device)

    exit_code = 0

    if args.section in ("all", "attention"):
        ec = section_attention(model, tokenizer, args.model, simplicial_indices,
                                seq_length=256, num_batches=2, device=device)
        exit_code = max(exit_code, ec)

    if args.section in ("all", "generation"):
        ec = section_generation(model, tokenizer, device=device)
        exit_code = max(exit_code, ec)

    if args.section in ("all", "grad"):
        ec = section_grad(model, tokenizer, simplicial_indices, seq_length=128, device=device)
        exit_code = max(exit_code, ec)

    print(f"\n{BOLD}{'='*60}{NC}")
    if exit_code == 0:
        print(f"  {GREEN}{BOLD}✅ DIAGNOSTICA COMPLETATA — TUTTI I TEST OK{NC}")
    elif exit_code == 1:
        print(f"  {YELLOW}{BOLD}🟡 DIAGNOSTICA COMPLETATA — TEST CON WARNING{NC}")
    else:
        print(f"  {RED}{BOLD}🔴 DIAGNOSTICA COMPLETATA — TEST FALLITI{NC}")
    print(f"{BOLD}{'='*60}{NC}\n")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())