"""
hooks.py — Forward hook per estrarre K1, K2, Q dai layer simpliciali.

Durante un forward pass, attacchiamo hook ai moduli di proiezione
q_proj, k1_proj, k2_proj nei layer simpliciali (16, 20, 24, 28).
L'hook cattura l'output della proiezione (le attivazioni) per ogni token.

Attenzione: i layer simpliciali (SimplicialAttention) hanno k1_proj e k2_proj.
I layer Gram Det (GramDetAttention) hanno solo k_proj — in quel caso
le coppie (k, k) sono identiche e il piano e' degenere.
"""

import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Optional


class ActivationSaver:
    """
    Gestore di forward hook che salva le attivazioni di proiezioni specifiche.
    
    Usage:
        saver = ActivationSaver(model, layers=[16, 20, 24, 28])
        saver.register_hooks()
        output = model(input_ids)  # forward pass normale
        data = saver.get_data()    # dict con K1, K2, Q per ogni layer
        saver.clear()
    """
    
    def __init__(
        self,
        model,
        simplicial_indices: List[int] = [16, 20, 24, 28],
        attention_type: str = "simplicial",
    ):
        self.model = model
        self.simplicial_indices = simplicial_indices
        self.attention_type = attention_type
        self.hooks = []
        self.data = defaultdict(list)
    
    def _make_hook(self, layer_idx: int, key: str):
        """Crea una closure che cattura l'output della proiezione."""
        def hook(module, input, output):
            # output: [B, S, H*d] o [B, S, H, d] dopo la proiezione
            # Salva come [S, d] per ogni elemento del batch e head
            self.data[(layer_idx, key)].append(output.detach().cpu())
        return hook
    
    def register_hooks(self):
        """Registra forward hook su tutte le proiezioni dei layer simpliciali."""
        for idx in self.simplicial_indices:
            layer = self.model.model.layers[idx]
            attn = layer.self_attn
            
            if self.attention_type == "simplicial":
                # SimplicialAttention: k1_proj, k2_proj, q_proj
                if hasattr(attn, 'k1_proj'):
                    h = attn.k1_proj.register_forward_hook(self._make_hook(idx, 'k1'))
                    self.hooks.append(h)
                if hasattr(attn, 'k2_proj'):
                    h = attn.k2_proj.register_forward_hook(self._make_hook(idx, 'k2'))
                    self.hooks.append(h)
                if hasattr(attn, 'q_proj'):
                    h = attn.q_proj.register_forward_hook(self._make_hook(idx, 'q'))
                    self.hooks.append(h)
            else:
                # GramDetAttention: k_proj (usata sia per k1 che k2), q_proj
                if hasattr(attn, 'k_proj'):
                    h = attn.k_proj.register_forward_hook(self._make_hook(idx, 'k1'))
                    self.hooks.append(h)
                    h = attn.k_proj.register_forward_hook(self._make_hook(idx, 'k2'))
                    self.hooks.append(h)
                if hasattr(attn, 'q_proj'):
                    h = attn.q_proj.register_forward_hook(self._make_hook(idx, 'q'))
                    self.hooks.append(h)
    
    def remove_hooks(self):
        """Rimuove tutti gli hook registrati."""
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def get_data(self) -> Dict:
        """
        Raccoglie i dati catturati dagli hook.
        
        Returns:
            dict: {(layer_idx, proj): tensor [B, S, H*d]}
        """
        result = {}
        for (idx, key), tensors in self.data.items():
            if tensors:
                result[(idx, key)] = torch.cat(tensors, dim=0)
        return result
    
    def clear(self):
        """Pulisce i dati accumulati."""
        self.data.clear()


def extract_key_vectors(
    activations: Dict,
    layer_idx: int,
    proj_key: str,
    num_heads: int = 32,
    head_dim: int = 128,
) -> torch.Tensor:
    """
    Da un tensore di attivazioni [B, S, H*d], estrai i vettori per ogni
    token, head, elemento del batch. Restituisce [N, d] dove N = B*S*H.

    Args:
        activations: dict dall'ActivationSaver
        layer_idx: indice del layer
        proj_key: 'k1', 'k2', 'q'
        num_heads: numero di teste
        head_dim: dimensione di ogni testa

    Returns:
        vettori appiattiti [N, head_dim]
    """
    key = (layer_idx, proj_key)
    if key not in activations:
        raise KeyError(f"Nessuna attivazione per layer {layer_idx}, {proj_key}")
    
    x = activations[key]  # [B, S, H*d]
    B, S, _ = x.shape
    
    # Reshape: [B, S, H, d] -> [B*S*H, d]
    x = x.view(B, S, num_heads, head_dim)
    x = x.reshape(-1, head_dim)
    
    return x


def batch_to_planes(
    activations: Dict,
    layer_idx: int,
    num_heads: int = 32,
    head_dim: int = 128,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Estrae K1, K2 per un layer e calcola le basi ortonormali dei piani.
    
    Args:
        activations: dict dall'ActivationSaver
        layer_idx: indice del layer
        num_heads: numero di teste
        head_dim: dimensione testa
        device: device per il calcolo

    Returns:
        U_list: basi ortonormali [N, d, 2] per ogni coppia (batch, seq, head)
        Q_vectors: vettori query corrispondenti [N, d]
    """
    from src.geometry.plane import plane_projector_and_basis
    
    # Estrai K1, K2, Q
    k1 = extract_key_vectors(activations, layer_idx, 'k1', num_heads, head_dim).to(device)
    k2 = extract_key_vectors(activations, layer_idx, 'k2', num_heads, head_dim).to(device)
    q = extract_key_vectors(activations, layer_idx, 'q', num_heads, head_dim).to(device)
    
    # Calcola piani per ogni coppia (k1_i, k2_i)
    N = k1.shape[0]
    U_list = torch.zeros(N, head_dim, 2, device=device)
    
    for i in range(N):
        _, U, _ = plane_projector_and_basis(k1[i], k2[i])
        U_list[i] = U
    
    return U_list, q


def batch_to_planes_gram_det(
    activations: Dict,
    layer_idx: int,
    num_heads: int = 32,
    head_dim: int = 128,
    device: str = "cpu",
    num_pairs: int = 500,
) -> torch.Tensor:
    """
    Estrae K e Q per un layer GramDet e costruisce piani da coppie random.
    
    GramDet ha una sola proiezione K → k1 == k2 per ogni token → piani degeneri.
    Invece campioniamo coppie (k[j1], k[j2]) con j1 ≠ j2 da posizioni diverse,
    costruendo piani 2D validi per l'analisi sulla Grassmanniana.
    
    Args:
        activations: dict dall'ActivationSaver
        layer_idx: indice del layer
        num_heads: numero di teste
        head_dim: dimensione testa
        device: device per il calcolo
        num_pairs: numero di coppie da campionare (default: 500)
        
    Returns:
        U_list: basi ortonormali [num_pairs, d, 2]
        Q_vectors: vettori query corrispondenti [num_pairs, d]
    """
    from src.geometry.plane import plane_projector_and_basis
    
    K = extract_key_vectors(activations, layer_idx, 'k1', num_heads, head_dim).to(device)
    Q = extract_key_vectors(activations, layer_idx, 'q', num_heads, head_dim).to(device)
    
    N = K.shape[0]
    # Limita il numero di coppie a N (non di piu' — non abbiamo abbastanza token)
    actual_pairs = min(num_pairs, N, N * (N - 1) // 2)
    
    # Genera coppie (j1, j2) garantendo j1 ≠ j2 via permutazione + shift ciclico
    all_indices = torch.randperm(N, device=device)
    idx1 = all_indices
    idx2 = all_indices[(torch.arange(N, device=device) + N // 2) % N]
    # Garantisce j1 != j2 anche quando N=1
    if N > 1 and torch.all(idx1 == idx2):
        idx2 = all_indices[(torch.arange(N, device=device) + N // 2 + 1) % N]
    
    U_list = torch.zeros(actual_pairs, head_dim, 2, device=device)
    q_sampled = torch.zeros(actual_pairs, head_dim, device=device)
    
    for p in range(actual_pairs):
        j1, j2 = idx1[p].item(), idx2[p].item()
        if j1 == j2:
            # Fallback: prendi il ciclo successivo
            j2 = (j2 + 1) % N
        _, U, _ = plane_projector_and_basis(K[j1], K[j2])
        U_list[p] = U
        q_sampled[p] = Q[j1]  # query associata al primo token della coppia
    
    return U_list, q_sampled
