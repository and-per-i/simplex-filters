# simplex-filters

**2-Simplicial Attention: Geometria sulla Grassmanniana e Filtraggio della KV Cache**

> Progetto di tesi: implementazione di attenzione 2-simpliciale per Transformer (trilineare e determinante di Gram), analisi geometrica dei piani di chiavi sulla Grassmanniana, e filtraggio della KV cache tramite Q-filter score.

[![HuggingFace Models](https://img.shields.io/badge/🤗_HuggingFace-and--per-blue)](https://huggingface.co/and-per)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Perché questo progetto

L'attenzione standard dei Transformer è un **prodotto scalare** tra query Q e key K (forma bilineare). Questo progetto esplora due generalizzazioni a forma **trilineare**:

| Tipo | Formula | Cosa misura |
|---|---|---|
| **Dot-product** (standard) | ⟨q, k⟩ | Similarità tra 2 vettori |
| **Trilineare** | ⟨q, k₁, k₂⟩ | Interazione a 3 corpi |
| **Gram determinante** | det(Gram(q, k₁, k₂)) | Volume del parallelepipedo span(q, k₁, k₂) |

La versione **GramDet** ha un'interpretazione geometrica diretta: più il determinante è alto, più i tre vettori sono linearmente indipendenti — e quindi trasportano informazione complementare.

## Modelli pre-addestrati

I checkpoint addestrati sono disponibili su HuggingFace:

| Modello | Descrizione | Link |
|---|---|---|
| `llama-trilinear-step4000` | Trilineare, step 4000, w1=32, w2=256 | [🤗](https://huggingface.co/and-per/llama-trilinear-step4000) |
| `llama-gram-det-step6000` | GramDet, step 6000, gram_window=8, scaling=10.0 | [🤗](https://huggingface.co/and-per/llama-gram-det-step6000) |

**Profilo HuggingFace**: [https://huggingface.co/and-per](https://huggingface.co/and-per)

---

## Installazione

```bash
git clone https://github.com/and-per-i/simplex-filters.git
cd simplex-filters
pip install -r requirements.txt
pip install -e simplicial_attention/   # kernel opzionali Triton/TLX
```

**Requisiti**: Python 3.10+, CUDA GPU con ≥48 GB VRAM (per LLaMA 3.1 8B in bfloat16), `HF_TOKEN` per scaricare i pesi da HuggingFace.

```bash
export HF_TOKEN=hf_il_tuo_token
```

---

## Cosa fa il progetto

### 1. Due meccanismi di attenzione 2-simpliciale

Sostituisce l'attenzione dot-product standard nei layer {16, 20, 24, 28} di LLaMA 3.1 8B con due varianti:

```python
from src.modeling.gram_det_attention import GramDetAttention

# GramDet: score = det(Gram(q, k1, k2)) con finestra di 17 token
# k_proj e v_proj vengono espansi da 8 a 32 teste (GQA 4:1)
```

La conversione è gestita da `convert_llama_to_hybrid()`:
- **Trilineare** (`--attention-type simplicial`): 5 proiezioni (q, k1, k2, v1, v2, o), training su K2/V2 con α·noise
- **GramDet** (`--attention-type gram_det`): 3 proiezioni (q, k, v, o), training diretto su q/k/v/o

### 2. Training su C4

Il training finetuna **solo** i 4 layer simpliciali (268M parametri su 8B totali = 3.3%) su C4 streaming:

```bash
python finetuning/train_hybrid.py --attention-type gram_det        # GramDet
python finetuning/train_hybrid.py --attention-type gram_det --resume ./checkpoints/checkpoint-6000  # resume
```

Parametri chiave in `finetuning/config.yaml`:
- `gram_window: 8` — finestra di 17 token, 153 coppie per query
- `scaling: 10.0` — amplifica la differenza tra determinanti per softmark selettiva
- `lr_k2v2: 5e-4`, `lr_k1v1: 2e-5` — LR diversa per simpliciali vs backbone
- `warmup_steps: 100`, `max_steps: 10000`

### 3. Diagnostica dell'attenzione

Per verificare che l'attenzione funzioni correttamente PRIMA del training:

```bash
python scripts/diagnose_checkpoint.py --gram-window 8 --section attention
```

Metriche diagnostiche:
- **Entropia**: distribuzione dei pesi softmax sulle 153 coppie (5.03 = uniforme, 0 = one-hot)
- **Max weight medio**: peso massimo medio per token
- **Gini**: coefficiente di concentrazione (0 = uniforme, 1 = one-hot)
- **Pre-softmax mean**: grandezza dei logit prima della softmax

### 4. Analisi geometrica sulla Grassmanniana

Ogni coppia di chiavi (kⱼ₁, kⱼ₂) definisce un **punto sulla Grassmanniana Gr(2,128)** — lo spazio dei piani 2D in ℝ¹²⁸. L'analisi geometrica calcola:

```bash
python main.py --analyze ./checkpoints/checkpoint-6000 --attention-type gram_det
```

**Cosa misura**:

| Metrica | Formula | Cosa dice |
|---|---|---|
| **Varianza geodesica** | (1/N)Σ d_g(Uᵢ, U_mean)² | Quanto sono dispersi i piani (4.09 = random, <2.5 = strutturati) |
| **Angolo query-piano** | arcsin(‖P q̄‖ / ‖q̄‖) | Quanto la query media è ortogonale al piano (90° = volume massimo) |
| **Anisotropia σ₁/σ₂** | dalla SVD delle proiezioni | Se la distribuzione è concentrata lungo un asse |
| **Q-filter score** | √(σ₁²⟨k,e₂⟩² + σ₂²⟨k,e₁⟩²) | Peso della chiave per l'evizione |

**Risultati principali** (confronto con baseline Monte Carlo):

| Modello | Varianza geodesica | Riduzione vs random |
|---|---|---|
| Random Haar (Gr(2,128)) | 4.09 ± 0.004 | — |
| Trilineare | ~3.1 | ~24% |
| **GramDet** | **2.34–2.52** | **~40%** |

GramDet sviluppa quasi il **doppio della struttura geometrica** del trilineare, con specializzazione gerarchica per profondità: layer intermedi cercano complanarità (det≈0), layer profondi cercano volume massimo (det≈1).

### 5. Valutazione PPL out-of-domain

```bash
python scripts/eval_wikitext.py --checkpoints ./checkpoints/checkpoint-6000 --max-samples 50 --seq-length 256
```

**Risultato**: PPL 868K su Wikitext-2 → il modello NON generalizza fuori dal dominio di training. Il collo di bottiglia è pratico (3.3% parametri trainabili), non teorico.

### 6. Baseline Monte Carlo per Grassmanniana

```bash
python scripts/grassmann_baseline.py --dim 128 --n-planes 320 --runs 5
```

Genera piani random su Gr(2,128) e calcola la varianza geodesica attesa per confronto.

---

## Struttura del progetto

```
simplex-filters/
├── main.py                              # Entry point (7 modalità: test, finetune, analyze, benchmark, ruler, test-checkpoint, both)
├── finetuning/
│   ├── config.yaml                       # Iperparametri (gram_window, scaling, LR, warmup)
│   ├── train_hybrid.py                   # Training → C4 streaming, checkpoint, resume
│   └── utils/                            # Data loader, optimizer (3 gruppi), wandb logging, metrics
├── src/
│   ├── modeling/                         # Attenzione 2-simpliciale
│   │   ├── simplicial_attention.py        # Trilineare (kernel Triton, 520 TFLOPS)
│   │   ├── gram_det_attention.py          # Gram Det vettorizzato (puro PyTorch, finestra, coppie)
│   │   └── convert_to_hybrid.py           # Converte LLaMA → ibrido (espansione GQA 4:1)
│   ├── geometry/                          # Analisi Grassmanniana
│   │   ├── plane.py                       # SVD, proiettore, angoli principali, distanza geodesica
│   │   ├── grassmann.py                   # Media di Fréchet, varianza geodesica, Q-filters query mean
│   │   ├── hooks.py                       # Forward hook per estrarre K1/K2/Q dai layer
│   │   └── analyzer.py                    # Pipeline completa (5 batch × 2 testi × 256 token)
│   └── kv_cache/                          # KV Cache eviction (da completare dopo training)
│       ├── qfilter_score.py               # Score basato su geometria del piano
│       ├── eviction.py                    # Top-B eviction
│       └── benchmark.py                   # Perplexity vs budget
├── scripts/
│   ├── diagnose_checkpoint.py             # Diagnostica attenzione (entropia, Gini, softmax)
│   ├── eval_wikitext.py                   # PPL out-of-domain su Wikitext-2
│   └── grassmann_baseline.py              # Monte Carlo per baseline random
├── tests/                                 # 70+ test (strutturali, forward/backward, numerici)
└── simplicial_attention/                  # Kernel FBGEMM/Triton/TLX (da Meta, opzionali)
```

---

## Risultati sperimentali (sintesi)

| Modello | Varianza geodesica | Specializzazione | PPL Wikitext-2 |
|---|---|---|---|
| Random baseline | 4.09 | — | — |
| Trilineare | ~3.1 | Nessuna | — |
| **GramDet** | **2.34–2.52** | **Layer 20∥piano, Layer 28⟂piano** | **868K** (non generalizza) |

**Interpretazione**: GramDet sviluppa struttura geometrica genuina e specializzazione gerarchica, ma 268M parametri (3.3%) non bastano per generalizzare fuori dal dominio di training. La tesi documenta sia il successo geometrico che la limitazione pratica.

## Citazione

Basato su:
- Clift et al., "Logic and the 2-Simplicial Transformer", 2019
- Roy et al., "Fast and Simplex: 2-Simplicial Attention in Triton", 2025
- Godey et al. (Q-filters), 2024