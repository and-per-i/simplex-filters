# simplex-filters

## 2-Simplicial Attention: Geometrica e KV Cache Eviction

Implementazione di **attenzione 2-simpliciale** per Transformer, con analisi geometrica basata sulla **Grassmanniana** e **KV cache eviction** guidata dal **Q-filter**. Parte della tesi di laurea.

## Architettura

```
simplex-filters/
├── main.py                              # Entry point (6 modalità)
├── src/
│   ├── modeling/                        # Attenzione 2-simpliciale
│   │   ├── simplicial_attention.py       # Trilineare (kernel Triton, 520 TFLOPS)
│   │   ├── gram_det_attention.py         # Gram Det vettorizzato (puro PyTorch)
│   │   └── convert_to_hybrid.py          # Converte LLaMA in ibrido
│   ├── geometry/                         # Analisi Grassmanniana
│   │   ├── plane.py                      # Piano via SVD, angoli, distanza geodesica
│   │   ├── grassmann.py                  # Media di Frechet, Q-filters query mean
│   │   ├── hooks.py                      # Forward hook per K1/K2/Q
│   │   └── analyzer.py                   # Pipeline analisi geometrica completa
│   └── kv_cache/                         # KV Cache eviction
│       ├── qfilter_score.py              # Score = √(σ₁²⟨k,e₂⟩² + σ₂²⟨k,e₁⟩²)
│       ├── eviction.py                   # Top-B eviction + random baseline
│       ├── benchmark.py                  # Perplexity vs budget B
│       └── ruler/niah_benchmark.py       # RULER NIAH (8K/16K)
├── finetuning/                           # Training su C4
│   ├── config.yaml                       # Iperparametri
│   ├── train_hybrid.py                   # Loop manuale + 3 LR gruppi
│   └── utils/                            # Data, optimizer, metrics, wandb
├── tests/                                # 70+ test
└── simplicial_attention/                 # Kernel FBGEMM (da Meta)
```

## Istruzioni Rapide

```bash
# 1. Installa dipendenze
pip install -r requirements.txt

# 2. LLaMA config (scaricato automaticamente al primo run)
#    Configura HF_TOKEN per pesi reali:
export HF_TOKEN="hf_yourtoken"

# 3. Sei modalità
python main.py                              # Test suite (~70 test CPU)
python main.py --real-weights               # Test con LLaMA reale (GPU)
python main.py --finetune                   # Training su C4
python main.py --both                       # Trilineare + Gram Det in sequenza
python main.py --analyze ./checkpoints/...  # Analisi geometrica
python main.py --benchmark ./checkpoints/.. # Benchmark eviction PPL
python main.py --ruler ./checkpoints/...    # Benchmark NIAH
```

## Modalità nel dettaglio

| Comando | Cosa fa |
|---------|---------|
| `python main.py` | Test suite automatica su modello random (CPU). Salva `test_results.txt` |
| `python main.py --real-weights` | Test con LLaMA 3.1 8B reale (scarica ~30 GB da HuggingFace) |
| `python main.py --finetune` | Training su C4 con 3 LR gruppi (frozen, K1/V1, K2/V2) + WandB logging |
| `python main.py --both` | Pipeline completa: baseline LLaMA + fine-tuning trilineare + Gram Det |
| `python main.py --analyze ./ckpt/final` | Analisi Grassmanniana: piano medio, varianza, Q-filter, anisotropia |
| `python main.py --benchmark ./ckpt/final` | Perplexity vs budget B (50/30/10%) con Q-filter vs random |
| `python main.py --ruler ./ckpt/final` | RULER NIAH accuracy a 8K/16K contesto con Q-filter vs random |

## Prerequisiti

- Python 3.10+
- GPU con almeno 24 GB (per training, 48+ GB per --both)
- Token HuggingFace per scaricare LLaMA 3.1 8B
- WandB API key per logging training

## Installazione della GPU vast.ai

```bash
# Vast.ai template: pytorch/pytorch:2.4.0-cuda12.1-cudnn8-devel
git clone https://github.com/and-per-i/simplex-filters.git
cd simplex-filters
pip install -r requirements.txt

# Variabili d'ambiente
export HF_TOKEN="hf_yourtoken"
export WANDB_API_KEY="your_wandb_key"

# Training completo
python main.py --both --verbose
```

## Componenti chiave

### Attenzione 2-simpliciale
Due meccanismi di scoring per la tripletta (query, key1, key2):

| Meccanismo | Score | Kernel | RF LOPS |
|-----------|-------|--------|---------|
| Trilineare | `Σ h q[h]·k1[h]·k2[h]` | Triton custom | 520 |
| Gram Det | `det(Gram(q, k1, k2))` | PyTorch vettorizzato | - |

### Analisi geometrica (Grassmanniana)
Ogni coppia (k1, k2) definisce un **punto sulla Grassmanniana Gr(2,d)**. L'analisi calcola:
- **Piano medio**: media di Frechet sulla Grassmanniana
- **Varianza geodesica**: dispersione dei piani attorno al piano medio
- **Q-filter score per eviction**: `√(σ₁²⟨k, e₂⟩² + σ₂²⟨k, e₁⟩²)`
- **Relazione query-piano**: `||P q̄||` — se ≈ 0, query ortogonale al piano → volume massimizzato
- **Anisotropia**: distribuzione delle query projetate nel piano medio (σ₁/σ₂)

### KV Cache Eviction
Lo score Q-filter viene usato per selezionare le top-B chiavi nella finestra K1 (w1=512), riducendo il costo computazionale dell'attenzione 2-simpliciale mantenendo l'accuratezza. Benchmark su:
- **Perplexity** su Wikitext-2 a vari budget B (100%, 50%, 30%, 10%)
- **RULER NIAH** (Needle-In-A-Haystack) a 8K e 16K token
- Confronto con **random eviction** come baseline

## Tests

```bash
# Tutti i test CPU
python -m pytest tests/ -k "not requires_gpu" -v

# Tutti i test (GPU richiesto per alcuni)
./tests/run_all.sh

# Report automatico su test_results.txt (ogni run di python main.py)
```

## Citazione

Basato su:
- Clift et al., "Logic and the 2-Simplicial Transformer", 2019
- Roy et al., "Fast and Simplex: 2-Simplicial Attention in Triton", 2025
- Godey et al., "Q-filters", 2024