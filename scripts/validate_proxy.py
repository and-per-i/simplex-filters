#!/usr/bin/env python3
"""
validate_proxy.py — Validazione empirica del proxy del piano medio.

Misura la correlazione tra:
1. Score "vero" di ogni chiave: contributo medio della chiave ai pesi di attenzione (ground truth)
2. Score "proxy": Q-filter score calcolato dal piano medio della Grassmanniana

Se Pearson r > 0.7, il proxy preserva l'ordinamento delle chiavi e l'evizione è giustificata.

Usage:
    python scripts/validate_proxy.py --ckpt ./checkpoints/checkpoint-6000
    python scripts/validate_proxy.py --ckpt ./checkpoints/checkpoint-6000 --attention-type gram_det
    python scripts/validate_proxy.py --ckpt ./checkpoints/checkpoint-6000 --attention-type simplicial
"""

import argparse
import os
import sys
import math
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from scipy.stats import pearsonr, spearmanr

# Colori ANSI
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
BOLD = "\033[1m"
NC = "\033[0m"


# ==========================================================================
# True score: contributo medio di ogni chiave all'attenzione
# ==========================================================================
def compute_true_score(
    model,
    tokenizer,
    layer_idx: int,
    attention_type: str,
    head_dim: int = 128,
    num_q_heads: int = 32,
    device: str = "cuda",
    seq_length: int = 256,
    num_batches: int = 5,
):
    """
    Calcola lo score "vero" per ogni chiave = quanto contribuisce
    ai pesi di attenzione nelle coppie in cui partecipa.

    head_dim e num_q_heads sono derivati dal model.config a runtime
    (o passati come argomenti).
    """
    from src.geometry.hooks import ActivationSaver, extract_key_vectors
    from src.modeling.gram_det_attention import GramDetAttention

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test", streaming=True)
    scale = 1.0 / math.sqrt(head_dim)

    if attention_type == "gram_det":
        all_scores_list = []
        all_k_list = []
    else:
        all_k1_scores = []
        all_k2_scores = []
        all_k1_list = []
        all_k2_list = []

    for batch_idx in range(num_batches):
        texts = []
        for _ in range(2):
            try:
                texts.append(next(iter(dataset))["text"][:seq_length * 4])
            except StopIteration:
                break
        if not texts:
            break

        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=seq_length)
        input_ids = enc["input_ids"].to(device)

        saver = ActivationSaver(model, [layer_idx], attention_type)
        saver.register_hooks()

        with torch.no_grad():
            model(input_ids)

        activations = saver.get_data()
        saver.remove_hooks()

        max_pairs = 500

        if attention_type == "gram_det":
            K = extract_key_vectors(activations, layer_idx, 'k1', num_q_heads, head_dim).to(device)
            Q = extract_key_vectors(activations, layer_idx, 'q', num_q_heads, head_dim).to(device)
            N = K.shape[0]

            key_scores = torch.zeros(N, device=device)
            num_queries = Q.shape[0]
            pairs_per_query = min(max_pairs, N * (N - 1) // 2)

            for qi in range(num_queries):
                q = Q[qi]
                q = F.normalize(q, p=2, dim=-1, eps=1e-8)
                K_norm = F.normalize(K, p=2, dim=-1, eps=1e-8)

                all_indices = torch.randperm(N, device=device)
                actual_pairs = min(pairs_per_query, N // 2)

                scores_pairs = torch.zeros(actual_pairs, device=device)
                pair_indices = []

                for p in range(actual_pairs):
                    j1 = all_indices[p]
                    j2 = all_indices[(p + N // 2) % N]
                    if j1 == j2:
                        j2 = (j2 + 1) % N

                    k1 = K_norm[j1]
                    k2 = K_norm[j2]

                    qq = (q * q).sum()
                    k1k1 = (k1 * k1).sum()
                    k2k2 = (k2 * k2).sum()
                    qk1 = (q * k1).sum()
                    qk2 = (q * k2).sum()
                    k1k2 = (k1 * k2).sum()

                    term1 = qq * (k1k1 * k2k2 - k1k2 ** 2)
                    term2 = qk1 * (qk1 * k2k2 - k1k2 * qk2)
                    term3 = qk2 * (qk1 * k1k2 - k1k1 * qk2)
                    score = (term1 - term2 + term3) * 10.0
                    scores_pairs[p] = score
                    pair_indices.append((j1, j2))

                softmax_weights = F.softmax(scores_pairs, dim=-1)

                for p in range(actual_pairs):
                    j1, j2 = pair_indices[p]
                    key_scores[j1] += softmax_weights[p]
                    key_scores[j2] += softmax_weights[p]

            all_scores_list.append(key_scores.cpu())
            all_k_list.append(K.cpu())

        else:
            K1 = extract_key_vectors(activations, layer_idx, 'k1', num_q_heads, head_dim).to(device)
            K2 = extract_key_vectors(activations, layer_idx, 'k2', num_q_heads, head_dim).to(device)
            Q = extract_key_vectors(activations, layer_idx, 'q', num_q_heads, head_dim).to(device)
            N1 = K1.shape[0]
            N2 = K2.shape[0]

            k1_scores = torch.zeros(N1, device=device)
            k2_scores = torch.zeros(N2, device=device)
            actual_pairs = min(max_pairs, min(N1, N2))
            num_queries = Q.shape[0]

            for qi in range(num_queries):
                q = Q[qi]
                idx1 = torch.randperm(N1, device=device)[:actual_pairs]
                idx2 = torch.randperm(N2, device=device)[:actual_pairs]

                scores_pairs = torch.zeros(actual_pairs, device=device)

                for p in range(actual_pairs):
                    j1 = idx1[p]
                    j2 = idx2[p]
                    score = (q * K1[j1] * K2[j2]).sum() * scale
                    scores_pairs[p] = score

                softmax_weights = F.softmax(scores_pairs, dim=-1)

                for p in range(actual_pairs):
                    j1 = idx1[p]
                    j2 = idx2[p]
                    k1_scores[j1] += softmax_weights[p]
                    k2_scores[j2] += softmax_weights[p]

            all_k1_scores.append(k1_scores.cpu())
            all_k2_scores.append(k2_scores.cpu())
            all_k1_list.append(K1.cpu())
            all_k2_list.append(K2.cpu())

    if attention_type == "gram_det":
        if not all_scores_list:
            raise RuntimeError("Nessun dato raccolto")
        true_scores = torch.cat(all_scores_list, dim=0)
        k_vectors = torch.cat(all_k_list, dim=0)
        return true_scores, k_vectors
    else:
        if not all_k1_scores:
            raise RuntimeError("Nessun dato raccolto")
        true_scores_k1 = torch.cat(all_k1_scores, dim=0)
        true_scores_k2 = torch.cat(all_k2_scores, dim=0)
        k1_vectors = torch.cat(all_k1_list, dim=0)
        k2_vectors = torch.cat(all_k2_list, dim=0)
        return (true_scores_k1, k1_vectors), (true_scores_k2, k2_vectors)


# ==========================================================================
# Proxy score: Q-filter dal piano medio
# ==========================================================================
def compute_proxy_score_gramdet(
    k_vectors: torch.Tensor,
    proxy_type: str = "orthogonal",
    n_planes: int = 5000,
):
    """Calcola proxy score per GramDet. Vedi docstring originale."""
    from src.geometry.plane import plane_projector_and_basis
    from src.geometry.grassmann import frechet_mean_planes, q_filters_query_mean
    from src.kv_cache.qfilter_score import qfilter_score, qfilter_score_orthogonal

    N, d = k_vectors.shape
    device = k_vectors.device
    k_vectors = k_vectors.float()

    n_planes_actual = min(N, n_planes)
    U_list = torch.zeros(n_planes_actual, d, 2, device=device)

    all_indices = torch.randperm(N, device=device)
    for i in range(n_planes_actual):
        j1 = all_indices[i % N].item()
        j2 = all_indices[(i + N // 2) % N].item()
        if j1 == j2:
            j2 = (j2 + 1) % N
        _, U, _ = plane_projector_and_basis(k_vectors[j1], k_vectors[j2])
        U_list[i] = U

    U_mean, P_mean = frechet_mean_planes(U_list, n_iter=10)

    if proxy_type == "orthogonal":
        proxy_scores = qfilter_score_orthogonal(k_vectors, U_mean)
        sigma1, sigma2 = 0.0, 0.0
    else:
        q_mean = q_filters_query_mean(k_vectors)
        q_proj = U_mean.T @ k_vectors.T
        U_svd, sigma, Vh_svd = torch.linalg.svd(q_proj.float(), full_matrices=False)
        sigma1, sigma2 = sigma[0].item(), sigma[1].item()
        proxy_scores = qfilter_score(k_vectors.float(), sigma1, sigma2, U_mean.float())

    return proxy_scores, U_mean, sigma1, sigma2


def compute_proxy_score_trilinear(
    k1_vectors: torch.Tensor,
    k2_vectors: torch.Tensor,
    n_planes: int = 5000,
) -> tuple:
    """Calcola proxy score per trilineare. Vedi docstring originale."""
    from src.geometry.plane import plane_projector_and_basis
    from src.geometry.grassmann import frechet_mean_planes

    N1, d = k1_vectors.shape
    N2 = k2_vectors.shape[0]
    device = k1_vectors.device
    k1_vectors = k1_vectors.float()
    k2_vectors = k2_vectors.float()

    n_planes_actual = min(min(N1, N2), n_planes)
    U_list = torch.zeros(n_planes_actual, d, 2, device=device)

    indices1 = torch.randperm(N1, device=device)[:n_planes_actual]
    indices2 = torch.randperm(N2, device=device)[:n_planes_actual]

    for i in range(n_planes_actual):
        j1 = indices1[i].item()
        j2 = indices2[i].item()
        _, U, _ = plane_projector_and_basis(k1_vectors[j1], k2_vectors[j2])
        U_list[i] = U

    U_mean, P_mean = frechet_mean_planes(U_list, n_iter=10)

    P = U_mean @ U_mean.T
    k1_proj = k1_vectors @ P
    k1_orth = k1_vectors - k1_proj
    proxy_scores_k1 = torch.norm(k1_orth, dim=-1)

    k2_proj = k2_vectors @ P
    k2_orth = k2_vectors - k2_proj
    proxy_scores_k2 = torch.norm(k2_orth, dim=-1)

    return proxy_scores_k1, proxy_scores_k2, U_mean


# ==========================================================================
# Carica modello
# ==========================================================================
def load_model(
    ckpt_path: str,
    attention_type: str,
    model_name: str,
    simplicial_indices: list,
    device: str = "cuda",
    gram_window: int = 8,
):
    """Carica modello dal checkpoint con model_name e indici parametrizzati."""
    from safetensors.torch import load_file as safetensors_load
    from src.modeling.convert_to_hybrid import convert_llama_to_hybrid
    import glob

    # 1. Carica state_dict (se non è step 0)
    ckpt_state = {}
    if ckpt_path.lower() != "none":
        safetensor_files = glob.glob(os.path.join(ckpt_path, "*.safetensors"))
        if not safetensor_files:
            idx_path = os.path.join(ckpt_path, "model.safetensors.index.json")
            if os.path.exists(idx_path):
                with open(idx_path) as f:
                    import json
                    ix = json.load(f)
                safetensor_files = list(set(ix["weight_map"].values()))
                safetensor_files = [os.path.join(ckpt_path, f) for f in safetensor_files]
        for sf in safetensor_files:
            ckpt_state.update(safetensors_load(sf))

    # 2. Carica LLaMA fresco
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    # 3. Copia pesi backbone
    for name, param in model.named_parameters():
        if name in ckpt_state and ckpt_state[name].shape == param.shape:
            param.data.copy_(ckpt_state[name].to(param.device))

    # 4. Converti in ibrido
    model, converted = convert_llama_to_hybrid(
        model,
        simplicial_indices=simplicial_indices,
        alpha=0.01, w1=32, w2=256,
        attention_type=attention_type,
        gram_window=gram_window,
    )

    # 5. Copia pesi GramDet/Simplicial
    for name, param in model.named_parameters():
        if name in ckpt_state and any(f"layers.{i}." in name for i in simplicial_indices):
            if ckpt_state[name].shape == param.shape:
                param.data.copy_(ckpt_state[name].to(param.device))

    model.eval()
    return model


# ==========================================================================
# Main
# ==========================================================================
def main():
    parser = argparse.ArgumentParser(description="Validazione proxy del piano medio")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path al checkpoint")
    parser.add_argument("--attention-type", type=str, default="gram_det",
                        choices=["simplicial", "gram_det"])
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B",
                        help="Nome del modello HuggingFace (default: meta-llama/Llama-3.2-1B)")
    parser.add_argument("--simplicial-indices", type=str, default="8,10,12,14",
                        help="Indici layer simpliciali, separati da virgola (default: 8,10,12,14)")
    parser.add_argument("--layer", type=int, default=14,
                        help="Layer da analizzare (default: 14, ultimo layer GramDet)")
    parser.add_argument("--all-layers", action="store_true",
                        help="Valida proxy su TUTTI i layer in simplicial-indices invece di uno solo")
    parser.add_argument("--seq-length", type=int, default=256)
    parser.add_argument("--num-batches", type=int, default=3)
    parser.add_argument("--gram-window", type=int, default=8,
                        help="Half-window per GramDet (default: 8)")
    parser.add_argument("--proxy-type", type=str, default="orthogonal",
                        choices=["projection", "orthogonal"],
                        help="Tipo di proxy (default: orthogonal)")
    args = parser.parse_args()

    # Parametri derivati
    simplicial_indices = [int(x) for x in args.simplicial_indices.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{BOLD}{'='*65}{NC}")
    print(f"{BOLD}  VALIDAZIONE PROXY — PIANO MEDIO VS TRUTH{NC}")
    print(f"{BOLD}  Checkpoint: {args.ckpt}{NC}")
    print(f"{BOLD}  Modello: {args.model}{NC}")
    print(f"{BOLD}  Attenzione: {args.attention_type}{NC}")
    print(f"{BOLD}  Layer: {args.layer}{NC}")
    print(f"{BOLD}  Indici simpliciali: {simplicial_indices}{NC}")
    print(f"{BOLD}{'='*65}{NC}\n")

    # Carica modello
    print("Caricamento modello...", end=" ", flush=True)
    model = load_model(args.ckpt, args.attention_type, args.model, simplicial_indices, device, args.gram_window)
    print(f"OK (W={args.gram_window})")

    # Deriva head_dim e num_q_heads dal model.config
    head_dim = getattr(model.config, "head_dim", None) or (
        model.config.hidden_size // model.config.num_attention_heads
    )
    num_q_heads = model.config.num_attention_heads

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    layers_to_test = simplicial_indices if args.all_layers else [args.layer]
    all_rho = []
    
    for layer_idx in layers_to_test:
        print(f"\n{'='*65}")
        print(f"{BOLD}  LAYER {layer_idx}{NC}")
        print(f"{'='*65}")
        
        # TRUE SCORE per questo layer
        print(f"\n  FASE 1/2: Calcolo TRUE score...")
        if args.attention_type == "gram_det":
            true_scores, k_vectors = compute_true_score(
                model, tokenizer, layer_idx,
                args.attention_type, head_dim, num_q_heads, device,
                args.seq_length, args.num_batches,
            )
            N = k_vectors.shape[0]
        else:
            (true_scores_k1, k1_vectors), (true_scores_k2, k2_vectors) = compute_true_score(
                model, tokenizer, layer_idx,
                args.attention_type, head_dim, num_q_heads, device,
                args.seq_length, args.num_batches,
            )
            N1, N2 = k1_vectors.shape[0], k2_vectors.shape[0]
        
        # PROXY SCORE per questo layer
        print(f"  FASE 2/2: Calcolo PROXY score...")
        if args.attention_type == "gram_det":
            proxy_scores, _, sigma1, sigma2 = compute_proxy_score_gramdet(
                k_vectors.to(device), proxy_type=args.proxy_type,
            )
            true_all = true_scores
            proxy_all = proxy_scores.cpu()
            N_total = N
        else:
            proxy_k1, proxy_k2, _ = compute_proxy_score_trilinear(
                k1_vectors.to(device), k2_vectors.to(device),
            )
            true_all = torch.cat([true_scores_k1.cpu(), true_scores_k2.cpu()], dim=0)
            proxy_all = torch.cat([proxy_k1.cpu(), proxy_k2.cpu()], dim=0)
            N_total = len(true_all)
        
        # Correlazione per questo layer
        r_pearson, p_pearson = pearsonr(true_all.numpy(), proxy_all.numpy())
        r_spearman, p_spearman = spearmanr(true_all.numpy(), proxy_all.numpy())
        all_rho.append(r_spearman)
        
        print(f"\n  Pearson r:   {r_pearson:+.4f}  (p={p_pearson:.2e})")
        print(f"  Spearman ρ:  {r_spearman:+.4f}  (p={p_spearman:.2e})")
    
    # Riepilogo multi-layer
    if len(all_rho) > 1:
        rho_mean = np.mean(all_rho)
        rho_std = np.std(all_rho)
        print(f"\n{BOLD}{'='*65}{NC}")
        print(f"{BOLD}  RIEPILOGO — TUTTI I LAYER{NC}")
        print(f"{BOLD}{'='*65}{NC}")
        for i, (lidx, rho) in enumerate(zip(layers_to_test, all_rho)):
            print(f"  Layer {lidx}: Spearman ρ = {rho:+.4f}")
        print(f"  {'─'*45}")
        print(f"  Media ρ: {rho_mean:.4f}  ±  {rho_std:.4f}")
        print(f"  {GREEN if rho_mean > 0 else RED}  Segnale dominante: {'POSITIVO ✓' if rho_mean > 0 else 'NEGATIVO ✗'}{NC}")
    elif len(all_rho) == 1:
        print(f"\n{BOLD}{'='*65}{NC}")
        print(f"{BOLD}  RISULTATO — LAYER {args.layer}{NC}")
        print(f"{BOLD}{'='*65}{NC}")
        print(f"  Spearman ρ = {all_rho[0]:+.4f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())