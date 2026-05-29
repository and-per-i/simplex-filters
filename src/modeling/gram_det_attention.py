"""
GramDetAttention — Attenzione 2-simpliciale via determinante di Gram.

Calcola l'attenzione usando come score il determinante della matrice
di Gram 3×3 per ogni tripletta (query_i, key_j1, key_j2).

Implementa la Sezione 5 del paper "Fast and Simplex" (determinant-based
trilinear forms). Puro PyTorch, autograd automatico, nessun kernel custom.

Formula (Sarrus per Gram 3×3 simmetrica):
    G = [[qq,   qk1,  qk2 ],
         [qk1, k1k1, k1k2],
         [qk2, k1k2, k2k2]]

    det(G) = qq·(k1k1·k2k2 - k1k2²)
           − qk1·(qk1·k2k2 - k1k2·qk2)
           + qk2·(qk1·k1k2 - k1k1·qk2)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sarrus_determinant(
    qq: torch.Tensor,
    k1k1: torch.Tensor,
    k2k2: torch.Tensor,
    qk1: torch.Tensor,
    qk2: torch.Tensor,
    k1k2: torch.Tensor,
) -> torch.Tensor:
    """
    Calcola il determinante di una matrice di Gram 3×3 simmetrica
    usando la regola di Sarrus.

    Args:
        qq:   dot(q, q)           [B, H] o scalare
        k1k1: dot(k_j1, k_j1)     [B, H, P] o [B, H, 1, P]
        k2k2: dot(k_j2, k_j2)     [B, H, P] o [B, H, P, 1]
        qk1:  dot(q, k_j1)        [B, H, P]
        qk2:  dot(q, k_j2)        [B, H, P]
        k1k2: dot(k_j1, k_j2)     [B, H, P, P]

    Returns:
        det:  [B, H, P, P]    determinante per ogni coppia (j1, j2)
    """
    # qq·(k1k1·k2k2 - k1k2²)
    term1 = qq.unsqueeze(-1).unsqueeze(-1) * (k1k1.unsqueeze(-1) * k2k2.unsqueeze(-2) - k1k2.pow(2))

    # − qk1·(qk1·k2k2 - k1k2·qk2)
    term2 = qk1.unsqueeze(-1) * (qk1.unsqueeze(-1) * k2k2.unsqueeze(-2) - k1k2 * qk2.unsqueeze(-2))

    # + qk2·(qk1·k1k2 - k1k1·qk2)
    term3 = qk2.unsqueeze(-2) * (qk1.unsqueeze(-1) * k1k2 - k1k1.unsqueeze(-1) * qk2.unsqueeze(-2))

    return term1 - term2 + term3


class GramDetAttention(nn.Module):
    """
    Attenzione 2-simpliciale con score = determinante della matrice di Gram.

    Per ogni query i, calcola lo score per tutte le coppie (j1, j2) nella
    finestra [i-W, i+W] (half-window W) come determinante della Gram 3×3
    costruita da q_i, k_j1, k_j2. Softmax 2D sulle coppie, output come
    media pesata di prodotti Hadamard di valori.

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
        self.window_size = window_size  # half-window
        self.scaling = self.head_dim ** -0.5

        # Proiezioni (come in LlamaAttention)
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
        Forward pass.

        Args:
            x: [B, N, d_model] hidden states
            return_weights: se True, restituisce anche i pesi d'attenzione

        Returns:
            output: [B, N, d_model]
            oppure (output, attn_weights) se return_weights=True
        """
        B, N, D = x.shape
        H = self.n_heads
        d = self.head_dim
        W = self.window_size

        # 1. Proietta e reshapa in [B, H, N, d]
        q = self.q_proj(x).view(B, N, H, d).transpose(1, 2)  # [B, H, N, d]
        k = self.k_proj(x).view(B, N, H, d).transpose(1, 2)  # [B, H, N, d]
        v = self.v_proj(x).view(B, N, H, d).transpose(1, 2)  # [B, H, N, d]

        # Output accumulator
        output = torch.zeros_like(q)  # [B, H, N, d]

        # Per memorizzare i pesi d'attenzione (se richiesto)
        all_weights = None
        if return_weights:
            all_weights = []

        # 2. Per ogni posizione query i
        for i in range(N):
            # Finestra: [i-W, i+W] clampata a [0, N-1]
            start = max(0, i - W)
            end = min(N, i + W + 1)
            P = end - start  # dimensionee della finestra

            # q_i: [B, H, 1, d]
            q_i = q[:, :, i:i+1, :]

            # k_window: [B, H, P, d]
            k_win = k[:, :, start:end, :]
            v_win = v[:, :, start:end, :]

            # Pre-calcola dot products vettorizzati
            # qq: [B, H, 1, 1]
            qq = (q_i * q_i).sum(dim=-1, keepdim=True)  # [B, H, 1, 1]

            # qk: [B, H, 1, P]
            qk = torch.matmul(q_i, k_win.transpose(-2, -1))  # [B, H, 1, P]

            # kk: [B, H, P, P]
            kk = torch.matmul(k_win, k_win.transpose(-2, -1))  # [B, H, P, P]

            # k1k1: [B, H, 1, P]  (diagonale di kk, broadcast su j1)
            k1k1 = kk.diagonal(dim1=-2, dim2=-1).unsqueeze(-2)  # [B, H, 1, P]

            # k2k2: [B, H, P, 1]  (diagonale di kk, broadcast su j2)
            k2k2 = kk.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)  # [B, H, P, 1]

            # Calcola determinante per TUTTE le coppie (j1, j2)
            # scores_raw: [B, H, P, P]
            scores_raw = sarrus_determinant(
                qq=qq,
                k1k1=k1k1,
                k2k2=k2k2,
                qk1=qk,         # [B, H, 1, P]
                qk2=qk,         # [B, H, 1, P]
                k1k2=kk,        # [B, H, P, P]
            )

            # Maschera: solo j1 < j2 (coppie ordinate non ridondanti)
            # ed eventualmente j1 != j2 (se vogliamo evitare triplette degeneri)
            mask = torch.triu(torch.ones(P, P, device=x.device), diagonal=1)
            scores_raw = scores_raw * mask  # azzera j1 >= j2

            # Scaling
            scores_scaled = scores_raw * self.scaling

            # Maschera le posizioni non valide (fuori o j1>j2)
            scores_masked = scores_scaled.masked_fill(mask == 0, float('-inf'))

            # 3. Softmax sulle coppie (appiattisci ultime 2 dim)
            # scores_masked: [B, H, P, P] → [B, H, P*P]
            B_h, _, Pp, _ = scores_masked.shape
            scores_flat = scores_masked.reshape(B, H, -1)  # [B, H, P*P]
            attn_flat = F.softmax(scores_flat, dim=-1)      # [B, H, P*P]
            attn_2d = attn_flat.reshape(B, H, P, P)         # [B, H, P, P]

            attn_2d = self.dropout(attn_2d)

            if return_weights:
                all_weights.append(attn_2d.detach().cpu())

            # 4. Aggrega: output_i = sum_{j1,j2} weight * (v_j1 ⊙ v_j2)
            # v1: [B, H, P, 1, d]  v2: [B, H, 1, P, d]  →  v1*v2: [B, H, P, P, d]
            v1 = v_win.unsqueeze(-2)   # [B, H, P, 1, d]
            v2 = v_win.unsqueeze(-3)   # [B, H, 1, P, d]
            v_hadamard = v1 * v2        # [B, H, P, P, d]

            # attn_2d: [B, H, P, P, 1]  →  broadcast
            o_i = (attn_2d.unsqueeze(-1) * v_hadamard).sum(dim=(-3, -2))  # [B, H, d]

            output[:, :, i, :] = o_i

        # 5. Output projection
        output = output.transpose(1, 2).reshape(B, N, -1)  # [B, N, H*d]
        output = self.o_proj(output)                        # [B, N, d_model]

        if return_weights:
            return output, all_weights
        return output