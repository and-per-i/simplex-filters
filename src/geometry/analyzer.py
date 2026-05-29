"""
analyzer.py — Pipeline completa di analisi geometrica su modello addestrato.

Carica un checkpoint, estrae K1/K2/Q via hook, calcola metriche geometriche.
"""

import torch
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.geometry.hooks import ActivationSaver, batch_to_planes
from src.geometry.grassmann import frechet_mean_planes, geodesic_variance, frechet_mean_queries, query_plane_relation


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
            for _ in range(2):  # batch_size=2
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
        
        # 1. Media di Frechet dei piani
        U_mean, P_mean = frechet_mean_planes(U_all, n_iter=10)
        
        # 2. Varianza geodesica
        var_g, distances = geodesic_variance(U_all, U_mean)
        
        # 3. Media delle query
        q_mean = frechet_mean_queries(q_all)
        
        # 4. Relazione query-piano medio
        proj_norm, angle = query_plane_relation(q_mean, U_mean)
        
        results[layer_idx] = {
            "num_vectors": N,
            "U_mean": U_mean.cpu(),
            "P_mean": P_mean.cpu(),
            "geodesic_variance": var_g.item(),
            "geodesic_distances": distances.cpu(),
            "q_mean": q_mean.cpu(),
            "query_plane_proj_norm": proj_norm.item(),
            "query_plane_angle": angle.item(),
        }
        
        if verbose:
            print(f"    Varianza geodesica:  {var_g.item():.6f}")
            print(f"    Proiezione q su P:   {proj_norm.item():.6f}")
            print(f"    Angolo q-P (rad):    {angle.item():.4f}")
    
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
    
    print("\n" + "="*60)