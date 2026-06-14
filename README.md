# simplex-filters

**Attenzione 2-Simpliciale: Geometria sulla Grassmanniana, Filtraggio della KV Cache**

> Progetto di tesi: implementazione di attenzione 2-simpliciale per Transformer (trilineare e determinante di Gram), analisi geometrica dei piani di chiavi sulla Grassmanniana di LLaMA puro, e filtraggio della KV cache tramite Q-filter score ispirato alla geometria dei piani.

[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Perché questo progetto

L'attenzione standard dei Transformer è un **prodotto scalare** tra query Q e key K (forma bilineare). Questo progetto esplora due generalizzazioni a forma **trilineare**:

| Tipo | Formula | Cosa misura |
|---|---|---|
| **Dot-product** (standard) | \(\langle q, k \rangle\) | Similarità tra 2 vettori |
| **Trilineare** | \(\langle q, k_1, k_2 \rangle\) | Interazione a 3 corpi |
| **Gram determinante** | det(Gram(q, k₁, k₂)) | Volume del parallelepipedo span(q, k₁, k₂) |

La versione **GramDet** ha un'interpretazione geometrica diretta: più il determinante è alto, più i tre vettori sono linearmente indipendenti — e quindi trasportano informazione complementare.

**Scoperta chiave**: la struttura geometrica (piani di Fréchet) non è solo un artefatto del training 2-simpliciale — esiste già, in forma latente, nei **key vectors di LLaMA 3.1/3.2 standard**, con varianza geodesica 38-47% inferiore al caso casuale.

---

## Installazione

```bash
git clone https://github.com/and-per-i/simplex-filters.git
cd simplex-filters
pip install -r requirements.txt
pip install -e simplicial_attention/   # kernel opzionali Triton/TLX
```

**Requisiti**: Python 3.10+, CUDA GPU, `HF_TOKEN` per scaricare i pesi da HuggingFace.

```bash
export HF_TOKEN=hf_il_tuo_token
```

---

## Cosa fa il progetto

### 1. Due meccanismi di attenzione 2-simpliciale

Sostituisce l'attenzione dot-product standard in layer selezionati di LLaMA con due varianti:

```python
from src.modeling.gram_det_attention import GramDetAttention

# GramDet: score = det(Gram(q, k1, k2)) con finestra configurabile
# Finestra di 17 token (W=8) → 153 coppie, 65 token (W=32) → 2145 coppie
```

La conversione è gestita da `convert_llama_to_hybrid()`:
- **Trilineare** (`--attention-type simplicial`): 5 proiezioni (q, k1, k2, v1, v2, o)
- **GramDet** (`--attention-type gram_det`): 3 proiezioni (q, k, v, o)

### 2. Training su C4 (LLaMA 3.2 1B)

Training leggero su 4 layer GramDet [8, 10, 12, 14]:

```bash
python finetuning/train_hybrid.py --config finetuning/config_gramdet.yaml
```

Config `finetuning/config_gramdet.yaml`:
- `gram_window: 8` (default) o `32` per finestra più ampia
- Solo 67M parametri trainabili su 1.2B totali
- Learning rate 5e-4, cosine annealing, batch effettivo 16
- Checkpoint ogni 1000 step in `./gramdet_1b/`

### 3. Analisi geometrica sulla Grassmanniana

**Su LLaMA puro** (nessuna conversione, nessun training):

```bash
python main.py --analyze-llama
```

Oppure cross-model:
```bash
python scripts/analyze_qwen.py
python scripts/analyze_olmo.py
```

Ogni chiave kⱼ è un vettore in ℝᵈ. Due chiavi (kⱼ₁, kⱼ₂) definiscono un **punto sulla Grassmanniana Gr(2,d)** — lo spazio dei piani 2D in ℝᵈ. L'analisi calcola:

| Metrica | Cosa dice |
|---|---|
| **Varianza geodesica** | Quanto sono dispersi i piani (low = strutturati) |
| **Angolo query-piano** | Quanto la query è ortogonale al piano medio |
| **Anisotropia σ₁/σ₂** | Concentrazione della distribuzione lungo un asse |

**Risultati principali** su LLaMA 3.2 1B:

| Dataset | Varianza geodesica | Riduzione vs random |
|---|---|---|
| Random Haar | 4.09 | — |
| Wikitext | 2.14-2.55 | **38-48%** |
| C4 | 2.14-2.50 | **38-49%** |

La struttura è **marginale** (non relazionale): ogni chiave individuale ha una posizione preferita nello spazio, indipendentemente dalle altre. Lo shuffle test (mescolamento delle coppie) produce varianza identica all'originale (ratio ≈ 1.0×).

**Cross-model robustness**: Qwen 2.5 7B mostra pattern simile (varianza 2.24-2.58), confermando che non è artefatto di un modello specifico.

### 4. KV Cache Eviction Benchmark (kvpress)

Benchmark reale su LLaMA puro senza conversioni architetturali. Usa **kvpress** (NVIDIA) per applicare eviction via hooks:

```bash
pip install kvpress
python scripts/benchmark_kvpress.py                    # LLaMA puro
python scripts/benchmark_kvpress.py --gramdet           # GramDet step 0
python scripts/benchmark_kvpress.py --gramdet --gram-window 32  # finestra 65 token
```

**Strategie confrontate**:
- **QFilterPress** (NVIDIA): score standard per attenzione dot-product
- **GrassmannianPress** (nostro): ‖k − P̄k‖, componente ortogonale al piano medio di Fréchet
- **RandomPress**: baseline casuale

**Risultati su LLaMA puro**:

| Budget | Grassmann | QFilter | Random |
|---|---|---|---|
| 100% | 11.42 | 11.42 | 11.42 |
| 50% | 697.06 | **16.20** | 157.00 |
| 30% | 703.63 | **15.21** | 217.29 |
| 10% | 445.86 | **14.38** | 363.90 |

**QFilter domina su LLaMA standard** perché lo score ‖k − P̄k‖ è stato validato per GramDet (ρ=+0.61), mentre LLaMA standard usa softmax(q·k/√d) — la rilevanza di un token dipende dall'allineamento con la query corrente, non dalla distanza dal piano medio.

**Risultati su GramDet step 0 (W=32)**:

| Budget | Grassmann | Random |
|---|---|---|
| 100% | 28.95 | 28.95 |
| 50% | 28.97 | 29.01 |
| 30% | 29.17 | 29.24 |
| 10% | 29.60 | 30.11 |

Grassmann batte random a tutti i budget. Il segnale è piccolo (max Δ=0.51) ma sistematico — lo score ortogonale funzione correttamente per GramDet attention.

### 5. Cross-dataset validation (Wikitext vs C4)

```bash
python main.py --analyze-llama --dataset-name wikitext
python main.py --analyze-llama --dataset-name c4
```

La struttura geometrica è **robusta cross-dataset**: varianza geodesica ±0.1 tra Wikitext e C4 su tutti i layer. Lo shuffle test conferma che la struttura è marginale (~1.0×) su entrambi i dataset.

### 6. Cross-model analysis

```bash
python scripts/analyze_qwen.py  # Qwen 2.5 7B
python scripts/analyze_olmo.py  # OLMo 2 7B
```

Supporto automatico per diverse architetture (GQA, MHA, RoPE, etc.).

### 7. Baseline Monte Carlo per Grassmanniana

```bash
python scripts/grassmann_baseline.py --dim 64 --n-planes 320 --runs 5
```

### 8. RULER Benchmark (Needle-in-a-Haystack)

```bash
python src/kv_cache/ruler/niah_benchmark.py --model meta-llama/Llama-3.2-1B
```

---

## Struttura del progetto

```
simplex-filters/
├── main.py                              # Entry point (analyze-llama, analyze, finetune, ...)
├── finetuning/
│   ├── config_gramdet.yaml               # Config per GramDet 1B
│   ├── config.yaml                       # Config per Trilnear 8B (legacy)
│   ├── train_hybrid.py                   # Training loop
│   └── utils/                            # Data loader, optimizer, metrics, wandb
├── src/
│   ├── modeling/
│   │   ├── gram_det_attention.py          # GramDet vettorizzato (puro PyTorch)
│   │   ├── simplicial_attention.py        # Trilineare (kernel Triton)
│   │   └── convert_to_hybrid.py           # Converte LLaMA → ibrido GQA 4:1
│   ├── geometry/
│   │   ├── plane.py                       # SVD, proiettore, distanza geodesica
│   │   ├── grassmann.py                   # Media di Fréchet, varianza
│   │   ├── hooks.py                       # Forward hook per estrarre attivazioni
│   │   └── analyzer.py                    # Pipeline analisi cross-model
│   └── kv_cache/
│       ├── grassmann_press.py             # GrassmannianPress per kvpress
│       ├── qfilter_score.py               # Score ortogonale ‖k − P̄k‖
│       ├── eviction.py                    # Eviction via eviction_params
│       ├── benchmark.py                   # Benchmark (legacy)
│       └── ruler/                         # Needle-in-a-Haystack
├── scripts/
│   ├── benchmark_kvpress.py               # Benchmark eviction con kvpress
│   ├── benchmark_gramdet_step0.py         # Benchmark GramDet step 0 (legacy)
│   ├── analyze_qwen.py                    # Analisi geometrica su Qwen
│   ├── analyze_olmo.py                    # Analisi geometrica su OLMo
│   ├── baseline_ppl.py                    # Calcolo PPL baseline
│   ├── download_c4.py                     # Download C4 locale
│   ├── validate_proxy.py                  # Validazione proxy score
│   └── grassmann_baseline.py              # Monte Carlo
├── tests/                                 # Test strutturali, forward, numerici
└── simplicial_attention/                  # Kernel Triton/TLX (opzionali)
```

---

## Evidenze sperimentali (sintesi)

| Scoperta | Dettaglio |
|---|---|
| **LLaMA ha struttura grassmanniana latente** | Varianza 38-48% sotto random, robusta cross-dataset e cross-modello |
| **Struttura è marginale, non relazionale** | Shuffle test ratio ≈ 1.0× |
| **GramDet crea più struttura del trilineare** | Varianza 2.34 vs 3.1 |
| **Q-filter ortogonale è il proxy corretto** | Spearman ρ=+0.61 per GramDet |
| **QFilterPress domina su LLaMA standard** | PPL 14.38 vs 363.90 al 10% budget |
| **GrassmannianPress funziona su GramDet** | Δ PPL +0.51 vs random al 10% |

## Citazione

Basato su:
- Clift et al., "Logic and the 2-Simplicial Transformer", 2019
- Roy et al., "Fast and Simplex: 2-Simplicial Attention in Triton", 2025
- Godey et al. (Q-filters), 2024
- Nvidia kvpress library (2025)