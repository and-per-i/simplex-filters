"""
grassmann_press.py — GrassmannianPress per kvpress.

Implementa `GrassmannianPress` come estensione di `ScorerPress` da kvpress.
Usa il piano medio Grassmanniano (U_mean) per calcolare uno score di eviction
basato sulla componente ortogonale delle chiavi al piano tipico.

Formula: Score(k_j) = ‖k_j - P̄ k_j‖, P̄ = U_mean @ U_mean^T
Validata empiricamente: Spearman ρ = +0.61.

Riferimenti:
    - src/geometry/analyzer.py — calcolo di U_mean
    - src/kv_cache/qfilter_score.py — implementazione originale dello score
    - validate_proxy.py — validazione empirica
"""

import torch
from dataclasses import dataclass

try:
    from kvpress import ScorerPress
except ImportError:
    # Fallback per ambiente locale senza kvpress
    ScorerPress = object


@dataclass
class GrassmannianPress(ScorerPress):
    """
    Score di eviction basato sulla componente ortogonale al piano medio Grassmanniano.

    Per ogni chiave k_j calcola:
        Score(k_j) = ‖k_j - P̄ k_j‖
    dove P̄ = U_mean @ U_mean^T è il proiettore ortogonale sul piano medio.

    Score alto = chiave atipica (lontana dal piano tipico) → da conservare.
    Score basso = chiave tipica (vicina al piano) → candidata all'eviction.

    Args:
        U_mean: base del piano medio [head_dim, k=2], tipicamente float32
        compression_ratio: frazione di chiavi da EVINCERE (es. 0.5 = 50%)
        layer_indices: lista di layer a cui applicare la press (None = tutti)
    """
    U_mean: torch.Tensor = None
    compression_ratio: float = 0.5
    # layer_indices ereditato da ScorerPress

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        """
        Calcola lo score per ogni posizione nel KV cache.

        keys: [B, H, S, d] — chiavi del layer corrente
        Restituisce: [B, H, S] — score per ogni chiave

        Nota: pressure_type='snip' significa che S corrisponde alla
        lunghezza totale del KV cache corrente (non solo il nuovo token).
        """
        # keys: [B, H, S, d]
        B, H, S, d = keys.shape

        # Proiettore sul piano medio
        with torch.no_grad():
            P = self.U_mean @ self.U_mean.T  # [d, d] float32

            # Appiattisci tutte le teste per lo scoring
            k_flat = keys.reshape(-1, d)  # [B*H*S, d]

            # Allinea dtype e device
            P_cast = P.to(k_flat.dtype).to(k_flat.device)

            # Componente proiettata sul piano
            k_proj = k_flat @ P_cast  # [B*H*S, d]

            # Componente ortogonale
            k_orth = k_flat - k_proj  # [B*H*S, d]

            # Score = norma della componente ortogonale
            scores = torch.norm(k_orth, dim=-1)  # [B*H*S]

        return scores.reshape(B, H, S)  # [B, H, S]