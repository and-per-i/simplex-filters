"""
convert_to_hybrid — Converte LLaMA 3.1 8B in un modello ibrido.
 
Sostituisce l'attenzione standard nei layer selezionati con
attenzione 2-simpliciale, in due modalita':
  - "simplicial": trilineare ottimizzata (kernel Triton, 520 TFLOPS)
  - "gram_det":   determinante di Gram vettorizzato (puro PyTorch)

Inizializzazione:
- k1_proj, v1_proj: pesi di k_proj, v_proj originali espansi 8→32 teste
- k2_proj, v2_proj: k1_proj + alpha * noise, v1_proj + alpha * noise
  (solo per modalita' "simplicial")
"""

import torch
import torch.nn as nn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention

from src.modeling.simplicial_attention import SimplicialAttention
from src.modeling.gram_det_attention import GramDetAttention


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


def _init_simplicial_layer(layer, config, layer_idx, q_weight, k_weight_32, v_weight_32, o_weight, alpha, w1, w2):
    """
    Inizializza un layer SimplicialAttention con 5 proiezioni:
    q, k1, v1 (pre-trained, espansi), k2, v2 (pre-trained + α·noise), o.
    """
    new_attn = SimplicialAttention(config, layer_idx, w1=w1, w2=w2)
    new_attn.q_proj.weight.data = q_weight
    new_attn.o_proj.weight.data = o_weight
    new_attn.k1_proj.weight.data = k_weight_32
    new_attn.v1_proj.weight.data = v_weight_32

    torch.manual_seed(42 + layer_idx)
    new_attn.k2_proj.weight.data = k_weight_32 + alpha * torch.randn_like(k_weight_32)
    new_attn.v2_proj.weight.data = v_weight_32 + alpha * torch.randn_like(v_weight_32)
    return new_attn


def _init_gram_det_layer(layer, config, layer_idx, q_weight, k_weight_32, v_weight_32, o_weight, w):
    """
    Inizializza un layer GramDetAttention.
    Usa una sola proiezione Q/K/V (32 teste, da originale espanso).
    K2/V2 non servono: lo score e' det(Gram(q, k1, k2)).
    """
    new_attn = GramDetAttention(
        d_model=config.hidden_size,
        n_heads=config.num_attention_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        window_size=w,
    )
    new_attn.q_proj.weight.data = q_weight
    new_attn.k_proj.weight.data = k_weight_32
    new_attn.v_proj.weight.data = v_weight_32
    new_attn.o_proj.weight.data = o_weight
    return new_attn


def convert_llama_to_hybrid(
    model,
    simplicial_indices: list = [16, 20, 24, 28],
    alpha: float = 0.01,
    w1: int = 32,
    w2: int = 256,
    attention_type: str = "simplicial",
    gram_window: int = 8,
):
    """
    Converte un LLaMAForCausalLM in un modello ibrido.
    
    Per ogni layer in simplicial_indices, sostituisce self_attn originale
    con un layer 2-simpliciale.
    
    Modalità disponibili:
      - "simplicial" (default): trilineare ottimizzata con kernel Triton.
        5 proiezioni (q, k1, k2, v1, v2, o). K2/V2 = K1/V1 + α·noise.
        w1, w2: finestre per K1, K2.
        
      - "gram_det": determinante di Gram vettorizzato (puro PyTorch).
        3 proiezioni (q, k, v, o). Score = det(Gram(q_i, k_j1, k_j2)).
        gram_window: half-window W per la finestra [i-W, i+W].
    
    Args:
        model: LlamaForCausalLM
        simplicial_indices: indici dei layer da convertire
        alpha: perturbazione per K2/V2 (solo simplicial)
        w1: finestra K1 (solo simplicial)
        w2: finestra K2 (solo simplicial)
        attention_type: "simplicial" | "gram_det"
        gram_window: half-window per gram_det (default: 8)
    
    Returns:
        model: modello ibrido
        converted_layers: lista degli indici convertiti
    """
    assert attention_type in ("simplicial", "gram_det"), \
        f"attention_type deve essere 'simplicial' o 'gram_det', got '{attention_type}'"
    
    print(f"[convert_to_hybrid] Conversione di {len(simplicial_indices)} layer in {attention_type}")
    print(f"  α={alpha}, w1={w1}, w2={w2}" if attention_type == "simplicial" else f"  gram_window={gram_window}")
    print(f"  Layer indices: {simplicial_indices}")
    
    converted_layers = []
    
    for layer_idx in simplicial_indices:
        old_attn = model.model.layers[layer_idx].self_attn
        assert isinstance(old_attn, LlamaAttention), \
            f"Layer {layer_idx}: non e' LlamaAttention ma {type(old_attn)}"
        
        q_weight = old_attn.q_proj.weight.data.clone()
        k_weight = old_attn.k_proj.weight.data.clone()
        v_weight = old_attn.v_proj.weight.data.clone()
        o_weight = old_attn.o_proj.weight.data.clone()
        
        k_weight_32 = expand_kv_weight(k_weight)
        v_weight_32 = expand_kv_weight(v_weight)
        
        if attention_type == "simplicial":
            new_attn = _init_simplicial_layer(
                old_attn, model.config, layer_idx,
                q_weight, k_weight_32, v_weight_32, o_weight,
                alpha, w1, w2,
            )
        else:  # gram_det
            new_attn = _init_gram_det_layer(
                old_attn, model.config, layer_idx,
                q_weight, k_weight_32, v_weight_32, o_weight,
                gram_window,
            )
        
        model.model.layers[layer_idx].self_attn = new_attn
        converted_layers.append(layer_idx)
        
        params_s = f"k1={list(k_weight_32.shape)}"
        if attention_type == "simplicial":
            with torch.no_grad():
                params_s += f", k2={list(new_attn.k2_proj.weight.shape)}"
        
        print(f"  ✓ Layer {layer_idx}: {params_s}")
    
    print(f"[convert_to_hybrid] Conversione completata. {len(converted_layers)} layer modificati.")
    return model, converted_layers


def freeze_parameters(model, simplicial_indices, lr_k1v1=2e-5, lr_k2v2=2e-4, attention_type="simplicial"):
    """
    Congela tutti i parametri tranne K1/V1/K2/V2 nei layer simpliciali.
    
    Per "gram_det": tutti i parametri dei layer GramDet sono trainable
    (solo 3 proiezioni invece di 5 — nessun K2/V2).
    
    Restituisce i gruppi di parametri per l'ottimizzatore.
    """
    from collections import OrderedDict
    
    frozen_params = []
    k1v1_params = []
    k2v2_params = []
    gram_det_params = []
    
    for name, param in model.named_parameters():
        in_simplicial = any(f"layers.{idx}.self_attn" in name for idx in simplicial_indices)
        
        if not in_simplicial:
            param.requires_grad = False
            frozen_params.append(param)
            continue
        
        if attention_type == "gram_det":
            # GramDet → tutti i parametri sono trainable
            param.requires_grad = True
            gram_det_params.append(param)
        else:
            if "k1_proj" in name or "v1_proj" in name:
                param.requires_grad = True
                k1v1_params.append(param)
            elif "k2_proj" in name or "v2_proj" in name:
                param.requires_grad = True
                k2v2_params.append(param)
            else:
                param.requires_grad = False
                frozen_params.append(param)
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n[freeze_parameters] attention_type={attention_type}")
    print(f"  Total: {total:,}, Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
    
    if attention_type == "gram_det":
        return [
            {"params": frozen_params, "lr": 0.0},
            {"params": gram_det_params, "lr": lr_k2v2},
        ]
    else:
        return [
            {"params": frozen_params, "lr": 0.0},
            {"params": k1v1_params, "lr": lr_k1v1},
            {"params": k2v2_params, "lr": lr_k2v2},
        ]


