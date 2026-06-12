"""
SimplicialAttention — Sottoclasse di LlamaAttention.

Sostituisce l'attenzione dot-product standard con attenzione 2-simpliciale
usando il kernel Triton ottimizzato di FBGEMM.

Architettura:
- 5 proiezioni invece di 3:
  - q_proj, k1_proj, k2_proj, v1_proj, v2_proj, o_proj
- k1_proj, v1_proj sono i pesi originali di Llama (da k_proj, v_proj) espansi da 8 a 32 teste
- k2_proj, v2_proj sono nuovi, inizializzati come k1/v1 + alpha * noise
- Il forward chiama il kernel 2-simplicial invece di scaled_dot_product_attention
"""

from typing import Optional, Callable
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb

from simplicial.ops.pytorch.two_simplicial_attention import SimplicialAttentionFunction
# Kernel Triton opzionali — se non installato, si usa solo fallback PyTorch
try:
    from simplicial.ops.triton.fwd import triton_fwd
    from simplicial.ops.triton.bwd import triton_bwd
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


class SimplicialAttention(LlamaAttention):
    """
    Attenzione 2-simpliciale per LLaMA.
    
    Usa 5 proiezioni lineari (Q, K1, K2, V1, V2) invece di 3 (Q, K, V).
    Il forward calcola:
        logits_ijk = sum_h(Q[i,h] * K1[j,h] * K2[k,h])
        attn = softmax(logits, dim=[j, k])
        out[i,h] = sum_jk attn_ijk * V1[j,h] * V2[k,h]
    
    Supporta eviction KV cache via Q-filter score:
    - Imposta self.eviction_params prima del forward per attivarla
    - I parametri includono sigma1, sigma2, U_mean, budget, strategy
    - Le chiavi con score piu' basso vengono azzerate
    
    Args:
        config: LlamaConfig
        layer_idx: indice del layer
        w1: finestra per K1 (default: 32)
        w2: finestra per K2 (default: 256)
    """
    
    def __init__(self, config, layer_idx: int, w1: int = 32, w2: int = 256):
        # Chiama __init__ di LlamaAttention che crea q_proj, k_proj, v_proj, o_proj
        super().__init__(config, layer_idx)
        
        # Salva iperparametri
        self.w1 = w1
        self.w2 = w2
        self.num_heads = config.num_attention_heads  # 32
        self.head_dim = config.hidden_size // config.num_attention_heads  # 128
        
        # Parametri eviction (impostati dall'esterno per il benchmark)
        self.eviction_params = None  # dict con sigma1, sigma2, U_mean, budget, strategy
        
        # Sostituisci k_proj con k1_proj (32 teste invece di 8)
        self.k1_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim,
            bias=config.attention_bias
        )
        
        # Crea k2_proj (32 teste)
        self.k2_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim,
            bias=config.attention_bias
        )
        
        # Sostituisci v_proj con v1_proj (32 teste invece di 8)
        self.v1_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim,
            bias=config.attention_bias
        )
        
        # Crea v2_proj (32 teste)
        self.v2_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim,
            bias=config.attention_bias
        )
        
        # Rimuovi i riferimenti alle proiezioni originali (non piu' usate)
        del self.k_proj
        del self.v_proj
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward dell'attenzione 2-simpliciale.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            position_embeddings: (cos, sin) per RoPE
            attention_mask: maschera causale (opzionale, gestita internamente dal kernel)
            past_key_values: non supportato (KV cache disabilitata)
            
        Returns:
            attn_output: [batch_size, seq_len, hidden_size]
            None: per compatibilità con LlamaDecoderLayer
        """
        input_shape = hidden_states.shape[:-1]  # [B, S]
        hidden_shape = (*input_shape, -1, self.head_dim)  # [B, S, num_heads, head_dim]
        
        # 1. Proiezioni
        # Q: [B, S, 32*128] → view → [B, S, 32, 128] → transpose → [B, 32, S, 128]
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key1_states = self.k1_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key2_states = self.k2_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value1_states = self.v1_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value2_states = self.v2_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        
        # 2. RoPE su Q, K1, K2
        # apply_rotary_pos_emb attende [B, H, S, D]
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key1_states = apply_rotary_pos_emb(query_states, key1_states, cos, sin)
            # K2 viene ruotato come K1 (stessa posizione relativa)
            _, key2_states = apply_rotary_pos_emb(query_states, key2_states, cos, sin)
        
        # 3. KV cache non supportata (ignorata, transformers la passa sempre)
        # transformers v5+ passa past_key_value=... in forward anche senza cache
        # Basta ignorarlo: il kernel 2-simplicial calcola tutto da zero
        
        # 4. Trasponi per il kernel Triton: [B, H, S, D] → [B, S, H, D]
        B, H, S, D = query_states.shape
        q = query_states.transpose(1, 2).contiguous()    # [B, S, H, D]
        k1 = key1_states.transpose(1, 2).contiguous()     # [B, S, H, D]
        k2 = key2_states.transpose(1, 2).contiguous()     # [B, S, H, D]
        v1 = value1_states.transpose(1, 2).contiguous()   # [B, S, H, D]
        v2 = value2_states.transpose(1, 2).contiguous()   # [B, S, H, D]
        
        # ================================================================
        # EVICTION Q-FILTER (opzionale, solo se self.eviction_params e' impostato)
        # ================================================================
        # Le chiavi K1 e K2 con score Q-filter basso vengono azzerate,
        # quindi non contribuiscono ai logits trilineari Q·K1·K2.
        # I valori V1/V2 corrispondenti vengono azzerati allo stesso modo.
        if self.eviction_params is not None:
            params = self.eviction_params
            sigma1 = params["sigma1"]
            sigma2 = params["sigma2"]
            U_mean = params["U_mean"]  # [d, 2]
            budget = params["budget"]
            strategy = params["strategy"]
            
            # Calcola score Q-filter per ogni posizione in K1
            # k1: [B, S, H, D] → prendiamo tutte le head
            # Per ogni head calcoliamo score separatamente (U_mean e' per tutte le head)
            with torch.no_grad():
                # k1_norm = k1 / (k1.norm(dim=-1, keepdim=True) + 1e-10)
                for b in range(B):
                    # k1_b: [S, H, D] — per ogni head, calcola score
                    k1_b = k1[b]  # [S, H, D]
                    k1_flat = k1_b.reshape(-1, D)  # [S*H, D]
                    
                    # Q-filter score
                    k_proj = k1_flat @ U_mean  # [S*H, 2]
                    k_e1 = k_proj[:, 0]
                    k_e2 = k_proj[:, 1]
                    scores = torch.sqrt(sigma1**2 * k_e2**2 + sigma2**2 * k_e1**2)
                    
                    if strategy == "random":
                        N_scores = scores.shape[0]
                        B_keep = max(1, int(N_scores * budget))
                        keep_indices = torch.randperm(N_scores, device=scores.device)[:B_keep]
                    else:  # "qfilter"
                        N_scores = scores.shape[0]
                        B_keep = max(1, int(N_scores * budget))
                        keep_indices = torch.argsort(scores, descending=True)[:B_keep]
                    
                    # Crea maschera: 1 per chiavi tenute, 0 per eliminate
                    mask = torch.zeros(S * H, 1, device=k1.device, dtype=k1.dtype)
                    mask[keep_indices] = 1.0
                    mask = mask.reshape(S, H, 1)  # [S, H, 1]
                    
                    # Applica maschera a K1, V1, K2, V2
                    k1[b] = k1[b] * mask
                    v1[b] = v1[b] * mask
                    k2[b] = k2[b] * mask
                    v2[b] = v2[b] * mask
        
        # 5. Kernel 2-simplicial tramite custom autograd Function
        attn_output = SimplicialAttentionFunction.apply(q, k1, k2, v1, v2, self.w1, self.w2)
        # attn_output: [B, S, H, D]
        
        # 6. Output projection
        attn_output = attn_output.reshape(*input_shape, -1)  # [B, S, H*D]
        attn_output = self.o_proj(attn_output)                # [B, S, hidden_size]
        
        return attn_output, None