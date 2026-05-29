"""
convert_to_hybrid — Converte LLaMA 3.1 8B in un modello ibrido.
 
Sostituisce l'attenzione standard nei layer selezionati con
attenzione 2-simpliciale (SimplicialAttention).

Inizializzazione:
- k1_proj, v1_proj: pesi di k_proj, v_proj originali espansi 8→32 teste
- k2_proj, v2_proj: k1_proj + alpha * noise, v1_proj + alpha * noise
"""

import torch
import torch.nn as nn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention
from src.modeling.simplicial_attention import SimplicialAttention


def expand_kv_weight(kv_weight: torch.Tensor, num_repeats: int = 4) -> torch.Tensor:
    """
    Espande il peso di una proiezione KV da num_kv_heads a num_heads.
    
    Llama 3.1 8B ha 32 Q heads e 8 KV heads (GQA 4:1 = num_kv_heads 8).
    k_proj e v_proj sono nn.Linear(hidden_size, num_kv_heads * head_dim) = (4096, 1024).
    
    Per portarli a 32 teste, replichiamo il peso 4 volte:
    (4096, 1024) → repeat 4× → (4096, 4096)
    
    Args:
        kv_weight: peso originale [hidden_size, num_kv_heads * head_dim]
        num_repeats: fattore di ripetizione (default 4 per GQA 4:1)
    
    Returns:
        peso espanso [hidden_size, num_heads * head_dim]
    """
    # kv_weight.shape: [hidden_size, num_kv_heads * head_dim]
    # Vogliamo: [hidden_size, num_heads * head_dim]
    # Dove num_heads = num_kv_heads * num_repeats
    
    # Separa lungo le teste e replica
    hidden_size = kv_weight.shape[0]
    kv_head_dim = kv_weight.shape[1]  # num_kv_heads * head_dim
    num_kv_heads = kv_head_dim // (kv_head_dim // num_repeats)
    
    weight_expanded = kv_weight.repeat(1, num_repeats)
    # [hidden_size, num_kv_heads * head_dim * num_repeats]
    # = [hidden_size, num_heads * head_dim]
    
    return weight_expanded


def convert_llama_to_hybrid(
    model,
    simplicial_indices: list = [16, 20, 24, 28],
    alpha: float = 0.01,
    w1: int = 32,
    w2: int = 256,
):
    """
    Converte un LLaMAForCausalLM in un modello ibrido.
    
    Per ogni layer in simplicial_indices, sostituisce self_attn originale
    con SimplicialAttention.
    
    Args:
        model: LlamaForCausalLM caricato da HuggingFace
        simplicial_indices: indici dei layer da convertire (default: seconda metà, ogni 4)
        alpha: coefficiente di perturbazione per K2/V2 (default: 0.01)
        w1: finestra K1 (default: 32)
        w2: finestra K2 (default: 256)
    
    Returns:
        model: modello ibrido
        converted_layers: lista degli indici convertiti
    """
    print(f"[convert_to_hybrid] Conversione di {len(simplicial_indices)} layer in SimplicialAttention")
    print(f"  α={alpha}, w1={w1}, w2={w2}")
    print(f"  Layer indices: {simplicial_indices}")
    
    converted_layers = []
    
    for layer_idx in simplicial_indices:
        old_attn = model.model.layers[layer_idx].self_attn
        
        # Verifica che sia un LlamaAttention standard
        assert isinstance(old_attn, LlamaAttention), \
            f"Layer {layer_idx}: self_attn non è LlamaAttention ma {type(old_attn)}"
        
        # Estrai pesi
        q_weight = old_attn.q_proj.weight.data.clone()
        k_weight = old_attn.k_proj.weight.data.clone()
        v_weight = old_attn.v_proj.weight.data.clone()
        o_weight = old_attn.o_proj.weight.data.clone()
        
        # Espandi K/V da 8 a 32 teste
        k_weight_32 = expand_kv_weight(k_weight)
        v_weight_32 = expand_kv_weight(v_weight)
        
        # Crea nuova attenzione simpliciale
        new_attn = SimplicialAttention(model.config, layer_idx, w1=w1, w2=w2)
        
        # Copia q_proj, o_proj (identici all'originale)
        new_attn.q_proj.weight.data = q_weight
        new_attn.o_proj.weight.data = o_weight
        
        # Copia k1_proj, v1_proj (pesi originali espansi a 32 teste)
        new_attn.k1_proj.weight.data = k_weight_32
        new_attn.v1_proj.weight.data = v_weight_32
        
        # Inizializza k2_proj, v2_proj come k1/v1 + alpha * noise
        torch.manual_seed(42 + layer_idx)  # seed riproducibile per layer
        noise_k2 = torch.randn_like(k_weight_32)
        noise_v2 = torch.randn_like(v_weight_32)
        
        new_attn.k2_proj.weight.data = k_weight_32 + alpha * noise_k2
        new_attn.v2_proj.weight.data = v_weight_32 + alpha * noise_v2
        
        # Gestisci bias (se presenti)
        if hasattr(old_attn.k_proj, 'bias') and old_attn.k_proj.bias is not None:
            k_bias = old_attn.k_proj.bias.data.clone()
            k_bias_32 = k_bias.repeat(4)
            new_attn.k1_proj.bias.data = k_bias_32
            torch.manual_seed(42 + layer_idx)
            new_attn.k2_proj.bias.data = k_bias_32 + alpha * torch.randn_like(k_bias_32)
        
        if hasattr(old_attn.v_proj, 'bias') and old_attn.v_proj.bias is not None:
            v_bias = old_attn.v_proj.bias.data.clone()
            v_bias_32 = v_bias.repeat(4)
            new_attn.v1_proj.bias.data = v_bias_32
            torch.manual_seed(42 + layer_idx)
            new_attn.v2_proj.bias.data = v_bias_32 + alpha * torch.randn_like(v_bias_32)
        
        # Sostituisci
        model.model.layers[layer_idx].self_attn = new_attn
        converted_layers.append(layer_idx)
        
        # Statistiche
        with torch.no_grad():
            diff_k2_k1 = (new_attn.k2_proj.weight - new_attn.k1_proj.weight).abs().mean().item()
            diff_v2_v1 = (new_attn.v2_proj.weight - new_attn.v1_proj.weight).abs().mean().item()
        
        print(f"  ✓ Layer {layer_idx}: SimplicialAttention installato")
        print(f"    k1={list(new_attn.k1_proj.weight.shape)}, "
              f"k2={list(new_attn.k2_proj.weight.shape)}")
        print(f"    |k2-k1|_mean = {diff_k2_k1:.6f}, |v2-v1|_mean = {diff_v2_v1:.6f}")
    
    print(f"[convert_to_hybrid] Conversione completata. {len(converted_layers)} layer modificati.")
    return model, converted_layers


def freeze_parameters(model, simplicial_indices, lr_k1v1=2e-5, lr_k2v2=2e-4):
    """
    Congela tutti i parametri tranne K1/V1/K2/V2 nei layer simpliciali.
    
    Restituisce i gruppi di parametri per l'ottimizzatore.
    """
    from collections import OrderedDict
    
    frozen_params = []
    k1v1_params = []
    k2v2_params = []
    
    for name, param in model.named_parameters():
        # Determina se il parametro è in un layer simpliciale
        in_simplicial = False
        is_k1v1 = False
        is_k2v2 = False
        
        for idx in simplicial_indices:
            if f"layers.{idx}.self_attn" in name:
                in_simplicial = True
                if "k1_proj" in name or "v1_proj" in name:
                    is_k1v1 = True
                elif "k2_proj" in name or "v2_proj" in name:
                    is_k2v2 = True
                break
        
        if not in_simplicial:
            # Frozen
            param.requires_grad = False
            frozen_params.append(param)
        elif is_k2v2:
            # K2/V2 → learning rate normale
            param.requires_grad = True
            k2v2_params.append(param)
        elif is_k1v1:
            # K1/V1 → learning rate piccolo
            param.requires_grad = True
            k1v1_params.append(param)
        else:
            # q_proj, o_proj nei layer simpliciali → frozen
            param.requires_grad = False
            frozen_params.append(param)
    
    # Stampa riepilogo
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n[freeze_parameters]")
    print(f"  Parametri totali:         {total_params:,}")
    print(f"  Parametri trainable:      {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"    - K1/V1 (lr={lr_k1v1}):  {sum(p.numel() for p in k1v1_params):,}")
    print(f"    - K2/V2 (lr={lr_k2v2}):  {sum(p.numel() for p in k2v2_params):,}")
    print(f"  Parametri frozen:         {total_params - trainable_params:,}")
    
    return [
        {"params": frozen_params, "lr": 0.0},
        {"params": k1v1_params, "lr": lr_k1v1},
        {"params": k2v2_params, "lr": lr_k2v2},
    ]