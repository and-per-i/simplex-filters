"""
analyzer.py — Pipeline completa di analisi geometrica su modello addestrato.

Carica un checkpoint, estrae K1/K2/Q via hook, calcola metriche geometriche.
"""

import torch
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.geometry.hooks import ActivationSaver, batch_to_planes
from src.geometry.grassmann import (
    frechet_mean_planes,
    geodesic_variance,
    frechet_mean_queries,
    q_filters_query_mean,
    query_plane_relation,
)


def analyze_query_distribution(
    q_all: torch.Tensor,
    U_mean: torch.Tensor,
) -> Dict[str, float]:
    """
    Analizza la distribuzione delle query proiettate sul piano medio.
    
    Proietta ogni query q_i sul piano medio (via U_mean):
        q_proj_i = U_mean^T @ q_i ∈ R^2
    
    Poi calcola SVD sulla matrice [N, 2] delle proiezioni.
    Il rapporto σ₁/σ₂ misura l'anisotropia:
        - σ₁ ≈ σ₂  → distribuzione isotropica nel piano
        - σ₁ >> σ₂ → distribuzione concentrata lungo un asse (anisotropica)
    
    Args:
        q_all: vettori query [N, d] (raw, non normalizzati)
        U_mean: base ortonormale del piano medio [d, 2]
    
    Returns:
        dict con sigma1, sigma2, anisotropy_ratio
    """
    # Proietta tutte le query sul piano medio
    # q_proj: [2, N] = U_mean^T @ q_all^T
    q_proj = U_mean.T @ q_all.T  # [2, N]
    
    # SVD sulla matrice delle proiezioni
    U, sigma, Vh = torch.linalg.svd(q_proj, full_matrices=False)
    sigma1, sigma2 = sigma[0].item(), sigma[1].item()
    ratio = sigma1 / sigma2 if sigma2 > 1e-10 else float('inf')
    
    return {
        "query_sigma1": sigma1,
        "query_sigma2": sigma2,
        "query_anisotropy_ratio": ratio,
    }


def analyze_checkpoint(
    checkpoint_path: str,
    attention_type: str = "simplicial",
    simplicial_indices: List[int] = [16, 20, 24, 28],
    num_heads: int = 32,
    head_dim: int = 128,
    num_analysis_batches: int = 10,
    seq_length: int = 512,
    device: str = "cuda",
    verbose: bool = True,
) -> Dict:
    """
    Analisi geometrica completa di un checkpoint.
    
    Args:
        checkpoint_path: path al checkpoint
        attention_type: "simplicial" o "gram_det"
        simplicial_indices: indici dei layer simpliciali
        num_heads, head_dim: architettura modello
        num_analysis_batches: numero di batch da analizzare
        seq_length: lunghezza sequenza per analisi
        device: device
        verbose: stampa progresso
        
    Returns:
        dict con risultati per ogni layer
    """
    from datasets import load_dataset
    
    # Carica modello
    if verbose:
        print(f"\nCaricamento checkpoint: {checkpoint_path}")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
    )
    model.eval()
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Dataset di analisi
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", streaming=True)
    
    results = {}
    
    for layer_idx in simplicial_indices:
        if verbose:
            print(f"\n  Layer {layer_idx}:")
        
        all_U = []
        all_q = []
        
        for batch_idx in range(num_analysis_batches):
            # Prepara batch
            texts = []
            for _ in range(2):
                try:
                    texts.append(next(iter(dataset))["text"][:seq_length*4])
                except StopIteration:
                    break
            if not texts:
                break
            
            enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=seq_length)
            input_ids = enc["input_ids"].to(device)
            
            # Hook + forward
            saver = ActivationSaver(model, simplicial_indices, attention_type)
            saver.register_hooks()
            
            with torch.no_grad():
                model(input_ids)
            
            activations = saver.get_data()
            saver.remove_hooks()
            
            # Estrai piani e query
            U_list, q_vectors = batch_to_planes(
                activations, layer_idx, num_heads, head_dim, device=device
            )
            all_U.append(U_list)
            all_q.append(q_vectors)
        
        if not all_U:
            continue
        
        # Concatena tutti i batch
        U_all = torch.cat(all_U, dim=0)
        q_all = torch.cat(all_q, dim=0)
        
        N = U_all.shape[0]
        if verbose:
            print(f"    Totale vettori: {N}")
        
        # 1. Media di Frechet dei piani (iterativa, 10 iterazioni)
        U_mean, P_mean = frechet_mean_planes(U_all, n_iter=10)
        
        # 2. Varianza geodesica
        var_g, distances = geodesic_variance(U_all, U_mean)
        
        # 3. Media delle query — metodo Q-filters (SVD su matrici raw, non normalizzate)
        #    Ref: github.com/NathanGodey/qfilters -> make_filters.py (righe 94-101)
        q_mean = q_filters_query_mean(q_all)
        
        # 4. Relazione query-piano medio
        proj_norm, angle = query_plane_relation(q_mean, U_mean)
        
        # 5. Analisi distribuzione query nel piano medio
        query_dist = analyze_query_distribution(q_all, U_mean)
        
        results[layer_idx] = {
            "num_vectors": N,
            "U_mean": U_mean.cpu(),
            "P_mean": P_mean.cpu(),
            "geodesic_variance": var_g.item(),
            "geodesic_distances": distances.cpu(),
            "q_mean": q_mean.cpu(),
            "query_plane_proj_norm": proj_norm.item(),
            "query_plane_angle": angle.item(),
            "query_sigma1": query_dist["query_sigma1"],
            "query_sigma2": query_dist["query_sigma2"],
            "query_anisotropy_ratio": query_dist["query_anisotropy_ratio"],
        }
        
        if verbose:
            print(f"    Varianza geodesica:  {var_g.item():.6f}")
            print(f"    Proiezione q su P:   {proj_norm.item():.6f}")
            print(f"    Angolo q-P (rad):    {angle.item():.4f}")
            print(f"    σ1 query nel piano:  {query_dist['query_sigma1']:.4f}")
            print(f"    σ2 query nel piano:  {query_dist['query_sigma2']:.4f}")
            print(f"    Anisotropia σ1/σ2:   {query_dist['query_anisotropy_ratio']:.2f}")
    
    return results


def summarize_results(results: Dict):
    """Stampa un riepilogo dei risultati di analisi."""
    print("\n" + "="*60)
    print("  RIEPILOGO ANALISI GEOMETRICA")
    print("="*60)
    
    for layer_idx, metrics in sorted(results.items()):
        print(f"\n  Layer {layer_idx}:")
        print(f"    Vettori:              {metrics['num_vectors']}")
        print(f"    Varianza geodesica:   {metrics['geodesic_variance']:.6f}")
        print(f"    ||P q̄||:              {metrics['query_plane_proj_norm']:.6f}")
        print(f"    Angolo q̄-P (rad):     {metrics['query_plane_angle']:.4f}")
        print(f"    Angolo q̄-P (gradi):   {metrics['query_plane_angle'] * 180 / 3.14159:.1f}°")
        print(f"    σ1/σ2 (anisotropia):  {metrics.get('query_anisotropy_ratio', 'N/A')}")
    
    print("\n" + "="*60)