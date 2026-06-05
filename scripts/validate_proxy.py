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

# ==========================================================================
# Costanti
# ==========================================================================
MODEL_NAME = "meta-llama/Llama-3.1-8B"
SIMPLICIAL_INDICES = [16, 20, 24, 28]
HEAD_DIM = 128

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
    device: str = "cuda",
    seq_length: int = 256,
    num_batches: int = 5,
) -> torch.Tensor:
    """
    Calcola lo score "vero" per ogni chiave = quanto contribuisce
    ai pesi di attenzione nelle coppie in cui partecipa.

    Per GramDet:
        - Estrae K e Q dal layer
        - Per ogni query, calcola softmax su tutte le coppie (j1,j2) nella finestra
        - Score di un token = somma dei pesi softmax di tutte le coppie in cui partecipa
        - Distribuzione attesa: K keys, ciascuna con un peso medio

    Per trilineare:
        - Estrae K1, K2, Q
        - Score di k1_j = somma softmax su tutte le coppie dove j e' il primo elemento
        - Score di k2_j = somma softmax su tutte le coppie dove j e' il secondo elemento

    Restituisce:
        true_scores: [N] — ground truth per ogni chiave nel batch
        k_vectors: [N, head_dim] — vettori K corrispondenti (allineati)
    """
    from src.geometry.hooks import ActivationSaver, extract_key_vectors
    from src.modeling.gram_det_attention import GramDetAttention

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", streaming=True)

    all_scores_list = []
    all_k_list = []

    for batch_idx in range(num_batches):
        # Prepara batch: 2 testi
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

        # Attiva hook
        saver = ActivationSaver(model, [layer_idx], attention_type)
        saver.register_hooks()

        with torch.no_grad():
            model(input_ids)

        activations = saver.get_data()
        saver.remove_hooks()

        max_pairs = 500  # limite per query (usato sia da GramDet che trilineare)

        if attention_type == "gram_det":
            # GramDet: una proiezione K, coppie da posizioni diverse
            K = extract_key_vectors(activations, layer_idx, 'k1', 32, HEAD_DIM).to(device)  # [N_k, d]
            Q = extract_key_vectors(activations, layer_idx, 'q', 32, HEAD_DIM).to(device)  # [N_q, d]
            N = K.shape[0]

            # Per ogni query, calcola score su TUTTE le coppie (j1, j2) nella finestra
            # Limitiamo a max_pairs coppie casuali per query per fattibilità
            key_scores = torch.zeros(N, device=device)

            # Campiona coppie per ogni query
            num_queries = Q.shape[0]
            pairs_per_query = min(max_pairs, N * (N - 1) // 2)

            for qi in range(num_queries):
                q = Q[qi]  # [d]
                q = F.normalize(q, p=2, dim=-1, eps=1e-8)
                K_norm = F.normalize(K, p=2, dim=-1, eps=1e-8)

                # Calcola prodotto trilineare su tutte le coppie
                # Usiamo un batch di coppie casuali
                all_indices = torch.randperm(N, device=device)
                # Limitiamo a pairs_per_query coppie
                actual_pairs = min(pairs_per_query, N // 2)

                scores_pairs = torch.zeros(actual_pairs, device=device)
                pair_indices = []

                # Genera coppie
                for p in range(actual_pairs):
                    j1 = all_indices[p]
                    j2 = all_indices[(p + N // 2) % N]
                    if j1 == j2:
                        j2 = (j2 + 1) % N

                    # Score = det(Gram(q, k1, k2)) * scaling
                    k1 = K_norm[j1]  # [d]
                    k2 = K_norm[j2]  # [d]

                    qq = (q * q).sum()
                    k1k1 = (k1 * k1).sum()
                    k2k2 = (k2 * k2).sum()
                    qk1 = (q * k1).sum()
                    qk2 = (q * k2).sum()
                    k1k2 = (k1 * k2).sum()

                    term1 = qq * (k1k1 * k2k2 - k1k2 ** 2)
                    term2 = qk1 * (qk1 * k2k2 - k1k2 * qk2)
                    term3 = qk2 * (qk1 * k1k2 - k1k1 * qk2)
                    score = (term1 - term2 + term3) * 10.0  # scaling=10.0
                    scores_pairs[p] = score
                    pair_indices.append((j1, j2))

                # Softmax sulle coppie (distribuzione attenzione)
                softmax_weights = F.softmax(scores_pairs, dim=-1)

                # Accumula contributi per ogni chiave
                for p in range(actual_pairs):
                    j1, j2 = pair_indices[p]
                    key_scores[j1] += softmax_weights[p]
                    key_scores[j2] += softmax_weights[p]

            all_scores_list.append(key_scores.cpu())
            all_k_list.append(K.cpu())

        else:
            # Trilineare: due proiezioni separate (K1, K2)
            K1 = extract_key_vectors(activations, layer_idx, 'k1', 32, HEAD_DIM).to(device)
            K2 = extract_key_vectors(activations, layer_idx, 'k2', 32, HEAD_DIM).to(device)
            Q = extract_key_vectors(activations, layer_idx, 'q', 32, HEAD_DIM).to(device)
            N1 = K1.shape[0]
            N2 = K2.shape[0]

            key_scores = torch.zeros(N1, device=device)
            actual_pairs = min(max_pairs, N1 // 2)
            num_queries = Q.shape[0]

            for qi in range(num_queries):
                q = Q[qi]
                # Campiona coppie
                idx1 = torch.randperm(N1, device=device)[:actual_pairs]
                idx2 = torch.randperm(N2, device=device)[:actual_pairs]

                scores_pairs = torch.zeros(actual_pairs, device=device)

                for p in range(actual_pairs):
                    j1 = idx1[p]
                    j2 = idx2[p]
                    score = (q * K1[j1] * K2[j2]).sum().abs() * 0.088  # scaling trilineare
                    scores_pairs[p] = score

                softmax_weights = F.softmax(scores_pairs, dim=-1)

                for p in range(actual_pairs):
                    j1 = idx1[p]
                    key_scores[j1] += softmax_weights[p]
                    # per K2 usiamo un array separato, semplifichiamo usando K1 solo
                    # Nota: per trilineare il vero score K2 sarebbe analogo

            all_scores_list.append(key_scores.cpu())
            # Per trilineare usiamo K1 come base
            all_k_list.append(K1.cpu())

    if not all_scores_list:
        raise RuntimeError("Nessun dato raccolto")

    # Concatena
    true_scores = torch.cat(all_scores_list, dim=0)
    k_vectors = torch.cat(all_k_list, dim=0)

    return true_scores, k_vectors


# ==========================================================================
# Proxy score: Q-filter dal piano medio
# ==========================================================================
def compute_proxy_score(
    k_vectors: torch.Tensor,
    q_vectors: torch.Tensor,
    proxy_type: str = "orthogonal",
) -> torch.Tensor:
    """
    Calcola lo score proxy usando il piano medio della Grassmanniana.
    Due modalita':
      - "projection": Q-filter classico proiettato sul piano medio
      - "orthogonal": componente ortogonale al piano medio (per GramDet)

    1. Estrae piani dalle coppie di K (stessa logica di batch_to_planes_gram_det)
    2. Calcola Frechet mean → piano medio Ū
    3. Calcola proxy score

    Restituisce:
        proxy_scores: [N] — score proxy per ogni chiave
        U_mean: piano medio [d, 2]
        sigma1, sigma2: valori singolari (solo per projection, 0 per orthogonal)
    """
    from src.geometry.plane import plane_projector_and_basis
    from src.geometry.grassmann import frechet_mean_planes, geodesic_variance, q_filters_query_mean
    from src.kv_cache.qfilter_score import qfilter_score

    N, d = k_vectors.shape
    device = k_vectors.device
    # Cast a float32 per compatibilita' SVD (il modello usa bfloat16)
    k_vectors = k_vectors.float()
    q_vectors = q_vectors.float()

    # 1. Costruisci piani da coppie di K
    # Se N è grande, campiona fino a 5000 piani
    n_planes = min(N, 5000)
    U_list = torch.zeros(n_planes, d, 2, device=device)

    all_indices = torch.randperm(N, device=device)
    for i in range(n_planes):
        j1 = all_indices[i % N].item()
        j2 = all_indices[(i + N // 2) % N].item()
        if j1 == j2:
            j2 = (j2 + 1) % N
        _, U, _ = plane_projector_and_basis(k_vectors[j1], k_vectors[j2])
        U_list[i] = U

    # 2. Media di Frechet
    U_mean, P_mean = frechet_mean_planes(U_list, n_iter=10)

    if proxy_type == "orthogonal":
        # Score ortogonale: norma della componente di k_j ortogonale al piano medio
        # P = U U^T  →  k_proj = k @ P  →  k_orth = k - k_proj
        P = U_mean @ U_mean.T  # [d, d] proiettore sul piano
        k_proj = k_vectors @ P  # [N, d]
        k_orth = k_vectors - k_proj  # [N, d]
        proxy_scores = torch.norm(k_orth, dim=-1)  # [N]
        sigma1, sigma2 = 0.0, 0.0
    else:
        # Q-filter classico: proiezione sul piano medio
        q_mean = q_filters_query_mean(q_vectors)

        q_proj = U_mean.T @ q_vectors.T  # [2, N]
        U_svd, sigma, Vh_svd = torch.linalg.svd(q_proj.float(), full_matrices=False)
        sigma1, sigma2 = sigma[0].item(), sigma[1].item()

        proxy_scores = qfilter_score(k_vectors.float(), sigma1, sigma2, U_mean.float())

    return proxy_scores, U_mean, sigma1, sigma2


# ==========================================================================
# Carica modello
# ==========================================================================
def load_model(ckpt_path: str, attention_type: str, device: str = "cuda"):
    """Carica modello dal checkpoint."""
    from safetensors.torch import load_file as safetensors_load
    from src.modeling.convert_to_hybrid import convert_llama_to_hybrid
    import glob

    # 1. Carica state_dict
    ckpt_state = {}
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
        MODEL_NAME,
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
        simplicial_indices=SIMPLICIAL_INDICES,
        alpha=0.01, w1=32, w2=256,
        attention_type=attention_type,
        gram_window=8,
    )

    # 5. Copia pesi GramDet
    for name, param in model.named_parameters():
        if name in ckpt_state and any(f"layers.{i}." in name for i in SIMPLICIAL_INDICES):
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
    parser.add_argument("--layer", type=int, default=28,
                        help="Layer da analizzare (default: 28)")
    parser.add_argument("--seq-length", type=int, default=256)
    parser.add_argument("--num-batches", type=int, default=3)
    parser.add_argument("--proxy-type", type=str, default="orthogonal",
                        choices=["projection", "orthogonal"],
                        help="Tipo di proxy: 'projection' (Q-filter) o 'orthogonal' (norma ortogonale)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{BOLD}{'='*65}{NC}")
    print(f"{BOLD}  VALIDAZIONE PROXY — PIANO MEDIO VS TRUTH{NC}")
    print(f"{BOLD}  Checkpoint: {args.ckpt}{NC}")
    print(f"{BOLD}  Attenzione: {args.attention_type}{NC}")
    print(f"{BOLD}  Layer: {args.layer}{NC}")
    print(f"{BOLD}{'='*65}{NC}\n")

    # Carica modello
    print("Caricamento modello...", end=" ", flush=True)
    model = load_model(args.ckpt, args.attention_type, device)
    print(f"OK")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # TRUE SCORE — ground truth dai pesi di attenzione
    print(f"\n{BOLD}FASE 1/2: Calcolo TRUE score (attenzione){NC}")
    print(f"  {args.num_batches} batch x {args.seq_length} token...")
    true_scores, k_vectors = compute_true_score(
        model, tokenizer, args.layer,
        args.attention_type, device,
        args.seq_length, args.num_batches,
    )
    N = k_vectors.shape[0]
    print(f"  TRUE score calcolato: {N} chiavi")

    # PROXY SCORE — Q-filter dal piano medio
    print(f"\n{BOLD}FASE 2/2: Calcolo PROXY score (Q-filter){NC}")
    proxy_scores, U_mean, sigma1, sigma2 = compute_proxy_score(
        k_vectors.to(device), k_vectors.to(device), proxy_type=args.proxy_type,
    )
    proxy_scores = proxy_scores.cpu()
    proxy_label = "ortogonale (∥k − P̄k∥)" if args.proxy_type == "orthogonal" else "proiezione (Q-filter)"
    print(f"  PROXY score ({proxy_label}): σ₁={sigma1:.4f}, σ₂={sigma2:.4f}")

    # Metriche di correlazione
    print(f"\n{BOLD}{'='*65}{NC}")
    print(f"{BOLD}  CORRELAZIONE TRUE vs PROXY{NC}")
    print(f"{BOLD}{'='*65}{NC}")

    # Pearson r (correlazione lineare)
    r_pearson, p_pearson = pearsonr(true_scores.numpy(), proxy_scores.numpy())

    # Spearman ρ (correlazione per ranghi — la più importante per eviction)
    r_spearman, p_spearman = spearmanr(true_scores.numpy(), proxy_scores.numpy())

    # Top-k overlap: se tengo le top 20%, quanto si sovrappongono?
    top_k_frac = 0.2
    k = max(1, int(N * top_k_frac))

    top_true = torch.topk(true_scores, k).indices.numpy()
    top_proxy = torch.topk(proxy_scores, k).indices.numpy()
    top_overlap = len(set(top_true) & set(top_proxy)) / k * 100

    # Bottom-k overlap (peggiori)
    bottom_true = torch.topk(true_scores, k, largest=False).indices.numpy()
    bottom_proxy = torch.topk(proxy_scores, k, largest=False).indices.numpy()
    bottom_overlap = len(set(bottom_true) & set(bottom_proxy)) / k * 100

    # Stampa risultati
    print(f"\n  Pearson r:        {r_pearson:.4f} (p={p_pearson:.2e})  "
          f"{'✅ r>0.7' if r_pearson > 0.7 else '🟡 0.4<r<0.7' if r_pearson > 0.4 else '🔴 r<0.4'}")
    print(f"  Spearman ρ:       {r_spearman:.4f} (p={p_spearman:.2e})  "
          f"{'✅ ρ>0.7' if r_spearman > 0.7 else '🟡 0.4<ρ<0.7' if r_spearman > 0.4 else '🔴 ρ<0.4'}")
    print(f"  Top-{k} overlap:    {top_overlap:.1f}%  "
          f"{'✅ alta' if top_overlap > 60 else '🟡 media' if top_overlap > 30 else '🔴 bassa'}")
    print(f"  Bottom-{k} overlap: {bottom_overlap:.1f}%  "
          f"{'✅ alta' if bottom_overlap > 60 else '🟡 media' if bottom_overlap > 30 else '🔴 bassa'}")

    print(f"\n  {BOLD}Diagnosi:{NC}")
    if r_spearman > 0.7:
        print(f"  {GREEN}✅ PROXY VALIDO{NC} — il piano medio preserva l'ordinamento delle chiavi.")
        print(f"    L'evizione Q-filter è giustificata empiricamente.")
    elif r_spearman > 0.4:
        print(f"  {YELLOW}🟡 PROXY PARZIALE{NC} — correlazione moderata.")
        print(f"    Il piano medio cattura tendenze ma non è una approssimazione fedele.")
        print(f"    L'evizione può funzionare ma con precisione limitata.")
    else:
        print(f"  {RED}🔴 PROXY INVALIDO{NC} — il piano medio NON preserva l'ordinamento.")
        print(f"    L'ipotesi del proxy è falsificata per questo layer/configurazione.")
        print(f"    L'evizione Q-filter non è giustificata in queste condizioni.")

    print(f"\n  Min true score:   {true_scores.min().item():.6f}")
    print(f"  Max true score:   {true_scores.max().item():.6f}")
    print(f"  Min proxy score:  {proxy_scores.min().item():.6f}")
    print(f"  Max proxy score:  {proxy_scores.max().item():.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())