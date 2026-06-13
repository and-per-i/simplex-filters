"""
analyzer.py — Pipeline completa di analisi geometrica su modello addestrato.

Carica un checkpoint, estrae K1/K2/Q via hook, calcola metriche geometriche.
"""

import torch
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.geometry.hooks import ActivationSaver, batch_to_planes, batch_to_planes_gram_det
from src.geometry.grassmann import (
    frechet_mean_planes,
    geodesic_variance,
    frechet_mean_queries,
    q_filters_query_mean,
    query_plane_relation,
)


# Baseline Monte Carlo per varianza geodesica su Gr(2, d)
# Calcolata con scripts/grassmann_baseline.py --dim <d> --runs 10
GRASSMANN_BASELINE = {
    64: 3.8507,   # Gr(2,64) — LLaMA 3.2 1B (head_dim=64)
    128: 4.0906,  # Gr(2,128) — LLaMA 3.1 8B (head_dim=128)
}


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
    q_proj = U_mean.T @ q_all.float().T  # [2, N]
    
    # SVD sulla matrice delle proiezioni
    # SVD non supporta BFloat16 su CUDA
    U, sigma, Vh = torch.linalg.svd(q_proj.float(), full_matrices=False)
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
    base_model_path: str = "meta-llama/Llama-3.1-8B",
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
    import os
    
    # device="cpu" → device_map=None (device_map non accetta "cpu" come stringa)
    model_device_map = None if device == "cpu" else device
    
    # ================================================================
    # FASE 1: Carica modello LLaMA base (pesi originali da disco locale)
    # ================================================================
    # Il checkpoint salvato usa architettura ibrida (k1_proj/k2_proj/v1_proj/v2_proj),
    # ma AutoModelForCausalLM.from_pretrained carica solo architettura LLaMA standard.
    # Quindi: carichiamo LLaMA base, convertiamo in ibrido, e sovrascriviamo i pesi
    # K1/V1/K2/V2 con quelli addestrati dal checkpoint.
    # Usa base_model_path passato come parametro
    if not os.path.exists(base_model_path):
        # Fallback: carica con local_files_only=False
        pass
    
    if verbose:
        print(f"\nFASE 1/3: Caricamento modello LLaMA base da {base_model_path}")
    
    # Su CPU, BFloat16 non supporta linalg_svd → usiamo float32
    model_dtype = torch.float32 if device == "cpu" else torch.bfloat16
    hf_token = os.environ.get("HF_TOKEN")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=model_dtype,
        device_map=model_device_map,
        attn_implementation="eager",
        token=hf_token,
    )
    model.train()
    
    # ================================================================
    # FASE 2: Converti in ibrido (crea i layer SimplicialAttention)
    # ================================================================
    from src.modeling.convert_to_hybrid import convert_llama_to_hybrid
    from transformers import LlamaConfig
    
    if verbose:
        print(f"FASE 2/3: Conversione in ibrido ({attention_type})...")
    
    model, converted = convert_llama_to_hybrid(
        model,
        simplicial_indices=simplicial_indices,
        alpha=0.01,
        w1=32,
        w2=256,
        attention_type=attention_type,
        gram_window=8,
    )
    
    # ================================================================
    # FASE 3: Sovrascrivi con pesi addestrati dal checkpoint
    # ================================================================
    if verbose:
        print(f"FASE 3/3: Caricamento pesi addestrati da {checkpoint_path}...")
    
    from safetensors.torch import load_file as safetensors_load
    import glob
    
    # Cerca file .safetensors nel checkpoint
    safetensor_files = glob.glob(os.path.join(checkpoint_path, "*.safetensors"))
    if not safetensor_files:
        # Cerca model.safetensors.index.json per capire i nomi file
        index_path = os.path.join(checkpoint_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            import json
            with open(index_path) as f:
                index = json.load(f)
            safetensor_files = list(set(index.get("weight_map", {}).values()))
            safetensor_files = [os.path.join(checkpoint_path, f) for f in safetensor_files]
    
    if not safetensor_files:
        raise FileNotFoundError(f"Nessun file .safetensors trovato in {checkpoint_path}")
    
    state_dict = {}
    for sf in safetensor_files:
        state_dict.update(safetensors_load(sf))
    
    # Sovrascrivi solo i pesi dei layer simpliciali
    loaded_count = 0
    for name, param in model.named_parameters():
        if name in state_dict:
            param.data.copy_(state_dict[name].to(param.device))
            loaded_count += 1
    
    if verbose:
        print(f"  Caricati {loaded_count} pesi dal checkpoint.")
    
    model.eval()
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Dataset di analisi (da disco locale se disponibile, per bypassare proxy Vast)
    wikitext_local = "./data/wikitext_test"
    if os.path.exists(wikitext_local):
        if verbose:
            print(f"  Dataset di analisi da disco: {wikitext_local}")
        from datasets import load_from_disk
        dataset = load_from_disk(wikitext_local)
    else:
        if verbose:
            print(f"  Dataset di analisi da HF: Salesforce/wikitext")
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test", streaming=True)
    
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
            
            # Estrai piani e query (diverso per trilineare vs GramDet)
            if attention_type == "gram_det":
                U_list, q_vectors = batch_to_planes_gram_det(
                    activations, layer_idx, num_heads, head_dim, device=device, num_pairs=500,
                )
            else:
                U_list, q_vectors = batch_to_planes(
                    activations, layer_idx, num_heads, head_dim, device=device,
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
        U_mean, P_mean = frechet_mean_planes(U_all, n_iter=10, verbose=verbose)
        
        # 2. Varianza geodesica
        var_g, distances = geodesic_variance(U_all, U_mean)
        
        # 3. Media delle query — metodo Q-filters (SVD su matrici raw, non normalizzate)
        #    Ref: github.com/NathanGodey/qfilters -> make_filters.py (righe 94-101)
        q_mean = q_filters_query_mean(q_all)
        
        # 4. Relazione query-piano medio
        proj_norm, angle_from_normal, angle_from_plane = query_plane_relation(q_mean, U_mean)
        
        # 5. Analisi distribuzione query nel piano medio
        query_dist = analyze_query_distribution(q_all, U_mean)
        
        angle_fn_deg = angle_from_normal.item() * 180 / 3.14159
        angle_fp_deg = angle_from_plane.item() * 180 / 3.14159
        
        results[layer_idx] = {
            "num_vectors": N,
            "U_mean": U_mean.cpu(),
            "P_mean": P_mean.cpu(),
            "geodesic_variance": var_g.item(),
            "geodesic_distances": distances.cpu(),
            "q_mean": q_mean.cpu(),
            "query_plane_proj_norm": proj_norm.item(),
            "query_angle_from_normal_rad": angle_from_normal.item(),
            "query_angle_from_plane_rad": angle_from_plane.item(),
            "query_angle_from_plane_deg": angle_fp_deg,
            "query_sigma1": query_dist["query_sigma1"],
            "query_sigma2": query_dist["query_sigma2"],
            "query_anisotropy_ratio": query_dist["query_anisotropy_ratio"],
        }
        
        # Baseline e riduzione percentuale (per head_dim)
        baseline = GRASSMANN_BASELINE.get(head_dim, None)
        if baseline:
            reduction_pct = (baseline - var_g.item()) / baseline * 100
            results[layer_idx]["baseline_variance"] = baseline
            results[layer_idx]["reduction_pct"] = round(reduction_pct, 1)

        if verbose:
            print(f"    Varianza geodesica:  {var_g.item():.6f}")
            if baseline:
                print(f"    Baseline random:     {baseline:.4f} (Gr(2,{head_dim}))")
                print(f"    Riduzione vs random: {reduction_pct:.1f}%")
            print(f"    ||P q̄||:            {proj_norm.item():.6f}")
            print(f"    q̄ dalla normale:     {angle_fn_deg:.1f}° (0° = q ∥ normale → volume massimo)")
            print(f"    q̄ dal piano:         {angle_fp_deg:.1f}° (90° = q ⟂ piano → volume massimo)")
            print(f"    σ1/σ2 (anisotropia): {query_dist['query_anisotropy_ratio']:.2f}")
    
    return results


def summarize_results(results: Dict):
    """Stampa un riepilogo dei risultati di analisi."""
    print("\n" + "="*60)
    print("  RIEPILOGO ANALISI GEOMETRICA")
    print("="*60)
    
    for layer_idx, metrics in sorted(results.items()):
        angle_fp = metrics.get('query_angle_from_plane_deg', 'N/A')
        print(f"\n  Layer {layer_idx}:")
        print(f"    Vettori:              {metrics['num_vectors']}")
        print(f"    Varianza geodesica:   {metrics['geodesic_variance']:.6f}")
        print(f"    ||P q̄||:              {metrics['query_plane_proj_norm']:.6f}")
        print(f"    q̄ dal piano:          {angle_fp}° (90° = ⟂ piano → volume max)")
        print(f"    σ1/σ2 (anisotropia):  {metrics.get('query_anisotropy_ratio', 'N/A')}")
    
    print("\n" + "="*60)
