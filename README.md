# simplex-filters

## Fast 2-Simplicial Attention

Implementazione di kernel GPU ad alte prestazioni per **2-Simplicial Attention**, basata sui paper:

- [Logic and the 2-Simplicial Transformer](https://arxiv.org/abs/1909.00668) — Clift et al., 2019
- [Fast and Simplex: 2-Simplicial Attention in Triton](https://arxiv.org/pdf/2507.02754) — Roy et al., 2025

### Cos'è l'Attenzione 2-Simpliciale?

L'attenzione 2-simpliciale generalizza il **dot-product attention** standard (bilineare tra Q e K) a una **forma trilineare** tra Q, K e K':

```
A_ijk = ⟨q_i, k_j, k'_k⟩
```

Questo produce un tensore di attenzione di terzo ordine invece della matrice 2D standard, consentendo interazioni triadiche tra i token.

### Architettura del Progetto

```
simplex-filters/
├── simplicial_attention/         # Package principale
│   ├── simplicial/               # Codice sorgente
│   │   ├── __init__.py
│   │   ├── utils.py              # Utility (TFLOPS, SQNR, assert)
│   │   └── ops/                  # Implementazioni kernel
│   │       ├── triton/           # Kernel Triton (forward + backward)
│   │       │   ├── fwd.py        # Forward pass (online softmax, sliding window)
│   │       │   └── bwd.py        # Backward pass (two-pass senza atomiche)
│   │       ├── tlx/              # Kernel TLX (Triton eXtended)
│   │       │   ├── fwd_ws.py             # TLX sliding window forward
│   │       │   ├── fwd_ws_pipelined.py   # TLX con pipelining
│   │       │   └── fwd_ws_pingpong.py    # TLX con ping-pong buffering
│   │       └── pytorch/          # Implementazione di riferimento PyTorch
│   │           └── two_simplicial_attention.py  # Module + autograd Function
│   ├── benchmarks/               # Benchmark delle performance
│   │   ├── bench_fwd.py
│   │   └── _proton/              # Benchmark con Proton profiler
│   ├── tests/                    # Test suite
│   ├── scripts/                  # Script di utilità
│   └── setup.py                  # Installazione package
```

### Implementazioni Disponibili

#### 1. Kernel Triton (`simplicial.ops.triton`)
- **Forward**: online softmax (stile Flash Attention), sliding window (w1=512, w2=32), GQA packing
- **Backward**: due kernel separati per evitare atomiche (kv1 kernel + kv2q kernel two-pass)
- Raggiunge ~520 TFLOPS su GPU

#### 2. Kernel TLX (`simplicial.ops.tlx`)
- Implementazione ad alte prestazioni con Tensor Descriptor e TMA
- Tre varianti: sliding window base, pipelined, ping-pong buffering
- Configurazioni per num_heads=64 e num_heads=128

#### 3. Riferimento PyTorch (`simplicial.ops.pytorch`)
- `SimplicialAttention` module (wrappa forward + backward Triton)
- `torch_fwd_ref` — forward di riferimento con einsum O(s³)
- `torch_bwd_ref` — backward via autograd
- `torch_simplicial_bwd` — backward manuale
- Varianti: `strassen` e `rank1` per forme alternative dei logits

### Requisiti

- PyTorch (GPU)
- Triton (branch `tlx` per kernel TLX)
- GPU con supporto CUDA

### Utilizzo Rapido

```python
import torch
from simplicial.ops.triton.fwd import triton_fwd

# Setup tensori
batch_size, seq_len, num_heads, head_dim = 4, 1024, 64, 128
device = torch.cuda.current_device()

Q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
K1 = torch.randn(batch_size, seq_len, 1, head_dim, device=device, dtype=torch.bfloat16)
K2 = torch.randn(batch_size, seq_len, 1, head_dim, device=device, dtype=torch.bfloat16)
V1 = torch.randn(batch_size, seq_len, 1, head_dim, device=device, dtype=torch.bfloat16)
V2 = torch.randn(batch_size, seq_len, 1, head_dim, device=device, dtype=torch.bfloat16)

# Forward pass con sliding window (w1=32, w2=256)
output, max_plus_lse = triton_fwd(Q, K1, K2, V1, V2, w1=32, w2=256)

# Oppure con il modulo PyTorch completo (include backward)
from simplicial.ops.pytorch.two_simplicial_attention import SimplicialAttention
attn = SimplicialAttention(head_dim=128, n_heads=64, w1=32, w2=256)
output = attn(xq, xk1, xk2, xv1, xv2)
```

### Dettagli Kernel Forward (Triton)

Il forward pass implementa:
```
logits = einsum("btnh,bsnh,brnh->bntsr", Q, K, K')
attention = softmax(logits + causal_mask, axis=[-1, -2])
output = einsum("bntsr,bsnh,brnh->btnh", attention, V, V')
```

Con sliding window: ogni query Q[i] attende a w1 posizioni di K e w2 posizioni di K', riducendo la complessità da O(n³) a O(n × w1 × w2).

### Dettagli Kernel Backward (Triton)

Il backward decomponi il calcolo in due kernel separati:
1. **kv1 kernel**: calcola dK1, dV1 (senza atomiche su K1, V1)
2. **kv2q kernel**: calcola dQ, dK2, dV2 usando un approccio two-pass (even/odd tiles) che evita operazioni atomiche

Pre-calcolo di `d = sum(O * dO, dim=-1)` per il gradiente del softmax.