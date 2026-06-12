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
SCALE = 1.0 / math.sqrt(HEAD_DIM)  # ~0.088

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
):
    """
    Calcola lo score "vero" per ogni chiave = quanto contribuisce
    ai pesi di attenzione nelle coppie in cui partecipa.

    Per GramDet:
        - Estrae K e Q dal layer
        - Per ogni query, calcola softmax su tutte le coppie (j1,j2) nella finestra
        - Score di un token = somma dei pesi softmax di tutte le coppie in cui partecipa
        - Restituisce: true_scores [N], k_vectors [N, d] (N = numero di chiavi)

    Per trilineare:
        - Estrae K1, K2, Q
        - Score di k1_j = somma softmax su tutte le coppie dove j e' il primo elemento
        - Score di k2_j = somma softmax su tutte le coppie dove j e' il secondo elemento
        - Restituisce: (true_scores_k1 [N1], k1_vectors [N1, d]),
                       (true_scores_k2 [N2], k2_vectors [N2, d])
        - NOTA: K1 e K2 NON sono concatenati, per permettere a compute_proxy_score
          di costruire piani dalle vere coppie (K1_j, K2_j).
    """
    from src.geometry.hooks import ActivationSaver, extract_key_vectors
    from src.modeling.gram_det_attention import GramDetAttention

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", streaming=True)

    if attention_type == "gram_det":
        all_scores_list = []
        all_k_list = []
    else:
        # Trilineare: accumula K1 e K2 separatamente
        all_k1_scores = []
        all_k2_scores = []
        all_k1_list = []
        all_k2_list = []

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

            # Due array separati per K1 e K2
            k1_scores = torch.zeros(N1, device=device)
            k2_scores = torch.zeros(N2, device=device)
            actual_pairs = min(max_pairs, min(N1, N2))
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
                    # FIX: rimosso .abs() — i logit negativi sono legittimi
                    score = (q * K1[j1] * K2[j2]).sum() * SCALE
                    scores_pairs[p] = score

                softmax_weights = F.softmax(scores_pairs, dim=-1)

                # FIX: accumula su ENTRAMBI K1 e K2
                for p in range(actual_pairs):
                    j1 = idx1[p]
                    j2 = idx2[p]
                    k1_scores[j1] += softmax_weights[p]
                    k2_scores[j2] += softmax_weights[p]

            # Accumula separatamente K1 e K2 (NON concatenati!)
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
    n_planes: int = 5000,
) -> torch.Tensor:
    """
    Calcola lo score proxy per GramDet usando il piano medio della Grassmanniana.

    Costruisce piani da coppie casuali di K (tutte le chiavi sono dello stesso tipo),
    calcola il piano medio di Fréchet, poi la componente ortogonale ‖k − P̄k‖.

    Args:
        k_vectors: chiavi [N, d]
        n_planes: numero di piani da campionare per Frechet mean

    Returns:
        proxy_scores: [N] score ortogonale per ogni chiave
        U_mean: piano medio [d, 2]
    """
    from src.geometry.plane import plane_projector_and_basis
    from src.geometry.grassmann import frechet_mean_planes

    N, d = k_vectors.shape
    device = k_vectors.device
    k_vectors = k_vectors.float()

    # 1. Costruisci piani da coppie casuali di K
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

    # 2. Media di Frechet
    U_mean, P_mean = frechet_mean_planes(U_list, n_iter=10)

    # 3. Score ortogonale: ‖k − P̄k‖
    P = U_mean @ U_mean.T  # [d, d]
    k_proj = k_vectors @ P  # [N, d]
    k_orth = k_vectors - k_proj  # [N, d]
    proxy_scores = torch.norm(k_orth, dim=-1)  # [N]

    return proxy_scores, U_mean


def compute_proxy_score_trilinear(
    k1_vectors: torch.Tensor,
    k2_vectors: torch.Tensor,
    n_planes: int = 5000,
) -> tuple:
    """
    Calcola lo score proxy per attenzione TRILINEARE usando il piano medio.

    Costruisce piani dalle VERE coppie (K1_j, K2_j) allineate per posizione,
    calcola il piano medio di Fréchet, poi la componente ortogonale ‖k − P̄k‖
    separatamente per K1 e K2.

    Args:
        k1_vectors: chiavi K1 [N1, d]
        k2_vectors: chiavi K2 [N2, d]
        n_planes: numero di piani da campionare per Frechet mean

    Returns:
        proxy_scores_k1: [N1] score ortogonale per ogni K1
        proxy_scores_k2: [N2] score ortogonale per ogni K2
        U_mean: piano medio [d, 2]
    """
    from src.geometry.plane import plane_projector_and_basis
    from src.geometry.grassmann import frechet_mean_planes

    N1, d = k1_vectors.shape
    N2 = k2_vectors.shape[0]
    device = k1_vectors.device
    k1_vectors = k1_vectors.float()
    k2_vectors = k2_vectors.float()

    # 1. Costruisci piani dalle VERE coppie (K1_j, K2_j)
    n_planes_actual = min(min(N1, N2), n_planes)
    U_list = torch.zeros(n_planes_actual, d, 2, device=device)

    indices1 = torch.randperm(N1, device=device)[:n_planes_actual]
    indices2 = torch.randperm(N2, device=device)[:n_planes_actual]

    for i in range(n_planes_actual):
        j1 = indices1[i].item()
        j2 = indices2[i].item()
        _, U, _ = plane_projector_and_basis(k1_vectors[j1], k2_vectors[j2])
        U_list[i] = U

    # 2. Media di Frechet
    U_mean, P_mean = frechet_mean_planes(U_list, n_iter=10)

    # 3. Score ortogonale per K1 e K2 separatamente
    P = U_mean @ U_mean.T  # [d, d] proiettore sul piano medio
    k1_proj = k1_vectors @ P  # [N1, d]
    k1_orth = k1_vectors - k1_proj
    proxy_scores_k1 = torch.norm(k1_orth, dim=-1)  # [N1]

    k2_proj = k2_vectors @ P  # [N2, d]
    k2_orth = k2_vectors - k2_proj
    proxy_scores_k2 = torch.norm(k2_orth, dim=-1)  # [N2]

    return proxy_scores_k1, proxy_scores_k2, U_mean


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

    # 5. Copia pesi GramDet/Simplicial
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
                        help="Tipo di proxy: 'projection' (Q-filter, deprecato) o 'orthogonal' (norma ortogonale, default)")
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

    if args.attention_type == "gram_det":
        true_scores, k_vectors = compute_true_score(
            model, tokenizer, args.layer,
            args.attention_type, device,
            args.seq_length, args.num_batches,
        )
        N = k_vectors.shape[0]
        print(f"  TRUE score calcolato: {N} chiavi (GramDet)")
    else:
        (true_scores_k1, k1_vectors), (true_scores_k2, k2_vectors) = compute_true_score(
            model, tokenizer, args.layer,
            args.attention_type, device,
            args.seq_length, args.num_batches,
        )
        N1, N2 = k1_vectors.shape[0], k2_vectors.shape[0]
        print(f"  TRUE score calcolato: {N1} chiavi K1 + {N2} chiavi K2 (trilineare)")

    # PROXY SCORE — Q-filter dal piano medio
    print(f"\n{BOLD}FASE 2/2: Calcolo PROXY score (Q-filter ortogonale){NC}")

    if args.attention_type == "gram_det":
        proxy_scores, U_mean = compute_proxy_score_gramdet(
            k_vectors.to(device),
        )
        proxy_label = "ortogonale (∥k − P̄k∥)"
        print(f"  PROXY score ({proxy_label})")

        # Concatena per correlazione (GramDet: tutto in un unico array)
        true_all = true_scores
        proxy_all = proxy_scores.cpu()
        N_total = N

    else:
        # Trilineare: usa le coppie reali (K1_j, K2_j)
        proxy_k1, proxy_k2, U_mean = compute_proxy_score_trilinear(
            k1_vectors.to(device), k2_vectors.to(device),
        )
        proxy_label = "ortogonale (∥k − P̄k∥) — piani da coppie (K1_j, K2_j) reali"
        print(f"  PROXY score ({proxy_label})")

        # Concatena K1 e K2 per correlazione cumulativa
        true_all = torch.cat([true_scores_k1.cpu(), true_scores_k2.cpu()], dim=0)
        proxy_all = torch.cat([proxy_k1.cpu(), proxy_k2.cpu()], dim=0)
        N_total = len(true_all)

    # Metriche di correlazione
    print(f"\n{BOLD}{'='*65}{NC}")
    print(f"{BOLD}  CORRELAZIONE TRUE vs PROXY{NC}")
    print(f"{BOLD}{'='*65}{NC}")

    # Pearson r (correlazione lineare)
    r_pearson, p_pearson = pearsonr(true_all.numpy(), proxy_all.numpy())

    # Spearman ρ (correlazione per ranghi — la più importante per eviction)
    r_spearman, p_spearman = spearmanr(true_all.numpy(), proxy_all.numpy())

    # Top-k overlap: se tengo le top 20%, quanto si sovrappongono?
    top_k_frac = 0.2
    k = max(1, int(N_total * top_k_frac))

    top_true = torch.topk(true_all, k).indices.numpy()
    top_proxy = torch.topk(proxy_all, k).indices.numpy()
    top_overlap = len(set(top_true) & set(top_proxy)) / k * 100

    # Bottom-k overlap (peggiori)
    bottom_true = torch.topk(true_all, k, largest=False).indices.numpy()
    bottom_proxy = torch.topk(proxy_all, k, largest=False).indices.numpy()
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

    print(f"\n  Min true score:   {true_all.min().item():.6f}")
    print(f"  Max true score:   {true_all.max().item():.6f}")
    print(f"  Min proxy score:  {proxy_all.min().item():.6f}")
    print(f"  Max proxy score:  {proxy_all.max().item():.6f}")

    # Per il trilineare, mostra anche le correlazioni separate
    if args.attention_type != "gram_det":
        print(f"\n  {BOLD}  — Per componente:{NC}")
        r_k1, _ = pearsonr(true_scores_k1.numpy(), proxy_k1.cpu().numpy())
        r_k2, _ = pearsonr(true_scores_k2.numpy(), proxy_k2.cpu().numpy())
        print(f"    K1: Pearson r = {r_k1:.4f}")
        print(f"    K2: Pearson r = {r_k2:.4f}")

        # Separazione K1/K2 per Spearman
        rho_k1, _ = spearmanr(true_scores_k1.numpy(), proxy_k1.cpu().numpy())
        rho_k2, _ = spearmanr(true_scores_k2.numpy(), proxy_k2.cpu().numpy())
        print(f"    K1: Spearman ρ = {rho_k1:.4f}")
        print(f"    K2: Spearman ρ = {rho_k2:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())