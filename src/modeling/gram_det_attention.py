"""
GramDetAttention — Attenzione 2-simpliciale via determinante di Gram.
Versione completamente vettorizzata (zero loop Python nel forward pass).

Formula (Sarrus per Gram 3x3 simmetrica):
    G = [[qq,   qk1,  qk2 ],
         [qk1, k1k1, k1k2],
         [qk2, k1k2, k2k2]]

    det(G) = qq·(k1k1·k2k2 - k1k2^2)
           - qk1·(qk1·k2k2 - k1k2·qk2)
           + qk2·(qk1·k1k2 - k1k1·qk2)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_pair_indices(W: int) -> torch.Tensor:
    """
    Pre-calcola tutti gli indici delle coppie (j1, j2) con j1 <= j2
    all'interno di una finestra di 2W+1 posizioni.

    Args:
        W: half-window size

    Returns:
        Tensor [P, 2] dove P = (2W+1)(2W+2)//2
    """
    pairs = []
    for w1 in range(2 * W + 1):
        for w2 in range(w1, 2 * W + 1):
            pairs.append([w1, w2])
    return torch.tensor(pairs, dtype=torch.long)


class GramDetAttention(nn.Module):
    """
    Attenzione 2-simpliciale con score = determinante della matrice di Gram.
    Forward completamente vettorizzato (zero loop Python).

    Args:
        d_model:  dimensione del modello (hidden)
        n_heads:  numero di teste d'attenzione
        head_dim: dimensione di ogni testa (default: d_model // n_heads)
        window_size: half-window W. La finestra e' [i-W, i+W].
        dropout:   dropout dopo softmax (default: 0.0)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int | None = None,
        window_size: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim or (d_model // n_heads)
        self.W = window_size  # half-window
        self.W_full = 2 * self.W + 1  # dimensione totale finestra
        self.scaling = self.head_dim ** -0.5

        # Pre-calcola e registra gli indici delle coppie come buffer
        pair_indices = _build_pair_indices(self.W)  # [P, 2]
        self.register_buffer('pair_indices', pair_indices)
        self.num_pairs = pair_indices.shape[0]  # P

        # Proiezioni
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass vettorizzato (zero loop).

        Args:
            x: [B, N, d_model]
            return_weights: se True, restituisce anche i pesi d'attenzione

        Returns:
            output: [B, N, d_model]
            oppure (output, attn_weights) se return_weights=True
        """
        B, N, D = x.shape
        H = self.n_heads
        d = self.head_dim
        W = self.W

        # 1. Proietta in [B, H, N, d]
        q = self.q_proj(x).view(B, N, H, d).transpose(1, 2)  # [B, H, N, d]
        k = self.k_proj(x).view(B, N, H, d).transpose(1, 2)  # [B, H, N, d]
        v = self.v_proj(x).view(B, N, H, d).transpose(1, 2)  # [B, H, N, d]

        # 2. Window extraction vettorizzata
        # Paddiamo con W zeri a sinistra e destra della sequenza
        k_pad = F.pad(k, (0, 0, W, W))  # [B, H, N+2W, d]
        v_pad = F.pad(v, (0, 0, W, W))  # [B, H, N+2W, d]

        # Indici delle finestre: [N, 2W+1]
        win_idx = (
            torch.arange(N, device=x.device)[:, None]
            + torch.arange(2 * W + 1, device=x.device)[None, :]
        )  # [N, 2W+1]

        # Estrai finestre: [B, H, N, 2W+1, d]
        k_windows = k_pad[:, :, win_idx, :]  # [B, H, N, 2W+1, d]
        v_windows = v_pad[:, :, win_idx, :]  # [B, H, N, 2W+1, d]

        # 3. Pair indexing vettorizzato
        # pair_indices: [P, 2]  →  indici nella dim della finestra
        pi = self.pair_indices  # [P, 2]

        # k1: [B, H, N, P, d], k2: [B, H, N, P, d]
        k1 = k_windows[:, :, :, pi[:, 0], :]  # [B, H, N, P, d]
        k2 = k_windows[:, :, :, pi[:, 1], :]  # [B, H, N, P, d]
        v1 = v_windows[:, :, :, pi[:, 0], :]  # [B, H, N, P, d]
        v2 = v_windows[:, :, :, pi[:, 1], :]  # [B, H, N, P, d]

        # 4. Gram determinante completamente in batch
        # q espanso: [B, H, N, 1, d] per broadcasting su P
        q_exp = q.unsqueeze(-2)  # [B, H, N, 1, d]

        qq   = (q_exp * q_exp).sum(dim=-1).squeeze(-2)               # [B, H, N]
        k1k1 = (k1 * k1).sum(dim=-1)                                  # [B, H, N, P]
        k2k2 = (k2 * k2).sum(dim=-1)                                  # [B, H, N, P]
        qk1  = (q_exp * k1).sum(dim=-1).squeeze(-2)                   # [B, H, N, P]
        qk2  = (q_exp * k2).sum(dim=-1).squeeze(-2)                   # [B, H, N, P]
        k1k2 = (k1 * k2).sum(dim=-1)                                  # [B, H, N, P]

        # Sarrus: tutti [B, H, N, P]
        term1 = qq.unsqueeze(-1) * (k1k1 * k2k2 - k1k2.pow(2))
        term2 = qk1 * (qk1 * k2k2 - k1k2 * qk2)
        term3 = qk2 * (qk1 * k1k2 - k1k1 * qk2)

        scores = (term1 - term2 + term3) * self.scaling  # [B, H, N, P]

        # 5. Softmax su P
        attn_weights = F.softmax(scores, dim=-1)  # [B, H, N, P]
        attn_weights = self.dropout(attn_weights)

        # 6. Aggregazione pesata: output_i = sum_p weight_p * (v1_p * v2_p)
        # v_hadamard: [B, H, N, P, d]
        v_hadamard = v1 * v2  # [B, H, N, P, d]

        # output: [B, H, N, d] = sum_p [B, H, N, P, 1] * [B, H, N, P, d]
        output = (attn_weights.unsqueeze(-1) * v_hadamard).sum(dim=-2)

        # 7. Output projection
        output = output.transpose(1, 2).reshape(B, N, -1)  # [B, N, H*d]
        output = self.o_proj(output)                         # [B, N, d_model]

        if return_weights:
            return output, attn_weights
        return output

    # ==========================================================================
    # Forward naive (con loop) — mantenuto per test di correttezza
    # ==========================================================================

    @torch.no_grad()
    def forward_naive(
        self,
        x: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Versione naive con loop Python (mantenuta per test).
        Dovrebbe produrre output identico a forward() entro tolleranza float.
        """
        B, N, D = x.shape
        H = self.n_heads
        d = self.head_dim
        W = self.W

        q = self.q_proj(x).view(B, N, H, d).transpose(1, 2)
        k = self.k_proj(x).view(B, N, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, d).transpose(1, 2)

        output = torch.zeros_like(q)
        all_weights_list = [] if return_weights else None

        for i in range(N):
            start = max(0, i - W)
            end = min(N, i + W + 1)
            pos_in_window = torch.arange(start, end, device=x.device)

            q_i = q[:, :, i:i+1, :]
            k_win = k[:, :, start:end, :]
            v_win = v[:, :, start:end, :]

            # Genera coppie on-the-fly
            pair_scores_list = []
            pair_v1_list = []
            pair_v2_list = []

            for j1_idx, j1 in enumerate(pos_in_window):
                for j2_idx, j2 in enumerate(pos_in_window):
                    if j1_idx > j2_idx:
                        continue  # solo j1 <= j2

                    k1j = k_win[:, :, j1_idx, :]
                    k2j = k_win[:, :, j2_idx, :]
                    v1j = v_win[:, :, j1_idx, :]
                    v2j = v_win[:, :, j2_idx, :]

                    qv = q_i.squeeze(-2)  # [B, H, d]
                    gram = torch.zeros(B, H, 3, 3, device=x.device)
                    gram[:,:,0,0] = (qv*qv).sum(-1)
                    gram[:,:,1,1] = (k1j*k1j).sum(-1)
                    gram[:,:,2,2] = (k2j*k2j).sum(-1)
                    gram[:,:,0,1] = gram[:,:,1,0] = (qv*k1j).sum(-1)
                    gram[:,:,0,2] = gram[:,:,2,0] = (qv*k2j).sum(-1)
                    gram[:,:,1,2] = gram[:,:,2,1] = (k1j*k2j).sum(-1)

                    det = torch.det(gram)
                    pair_scores_list.append(det)
                    pair_v1_list.append(v1j)
                    pair_v2_list.append(v2j)

            if not pair_scores_list:
                output[:, :, i, :] = 0
                if return_weights:
                    all_weights_list.append(torch.zeros(B, H, 1, 1, device=x.device))
                continue

            scores = torch.stack(pair_scores_list, dim=-1) * self.scaling
            attn = F.softmax(scores, dim=-1)

            v1s = torch.stack(pair_v1_list, dim=-2)
            v2s = torch.stack(pair_v2_list, dim=-2)
            v_h = v1s * v2s
            o_i = (attn.unsqueeze(-1) * v_h).sum(dim=-2)
            output[:, :, i, :] = o_i

            if return_weights:
                all_weights_list.append(attn.detach().cpu())

        output = output.transpose(1, 2).reshape(B, N, -1)
        output = self.o_proj(output)

        if return_weights:
            return output, all_weights_list
        return output