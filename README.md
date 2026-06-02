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
│   │   ├── grassmann.py                  # Media di Fréchet, Q-filters query mean
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

## Cosa fa

### 1. Due meccanismi di attenzione 2-simpliciale

Sostituisce l'attenzione dot-product standard in layer selezionati di LLaMA 3.1 8B con forme trilineari — sia la **trilineare classica** (Q·K1·K2) sia il **determinante di Gram** (det di Gram(q, k1, k2)). Convertendo 4 layer su 32 in modalità ibrida, si ottiene un modello che calcola interazioni a tre corpi mantenendo il 96.75% dei parametri congelati.

### 2. Analisi geometrica sulla Grassmanniana

Ogni coppia (k1, k2) prodotta dall'attenzione definisce un **punto sulla Grassmanniana Gr(2,d)** — lo spazio dei piani di dimensione 2 in ℝᵈ. L'analisi calcola:

- **Piano medio**: media di Fréchet iterativa sulla Grassmanniana (10 iterazioni, SVD a ogni passo)
- **Varianza geodesica**: √(θ₁² + θ₂²) dove θᵢ sono gli angoli principali tra ogni piano e il piano medio. Misura quanto i piani sono dispersi.
- **Q-filter score**: √(σ₁²·⟨k, e₂⟩² + σ₂²·⟨k, e₁⟩²) — pesa la proiezione di ogni chiave sul piano medio usando i valori singolari delle query. Chiavi con score alto contribuiscono di più al volume del parallelepipedo span(q, k1, k2).
- **Relazione query-piano**: ||P q̄|| — l'angolo tra la query media e il piano. Se l'angolo è ~90° (query quasi perpendicolare al piano), il volume dello spanned parallelepipedo è massimizzato.
- **Anisotropia delle query**: rapporto σ₁/σ₂ della distribuzione delle query proiettate sul piano medio. σ₁ ≫ σ₂ significa distribuzione concentrata lungo un asse.

### 3. KV Cache Eviction

Lo score Q-filter (calcolato dall'analisi geometrica) guida la **eviction della KV cache**: in ogni passo, solo le top-B chiavi con Q-filter score più alto vengono mantenute nella sliding window K1 (w1=32). Le chiavi eliminate vengono azzerate prima del kernel di attenzione, riducendo il costo computazionale dell'attenzione 2-simpliciale. Il benchmark confronta:

- **Perplexity** su C4 validation a vari budget B (100%, 50%, 30%, 10%)
- **RULER NIAH** (Needle-In-A-Haystack) a 8K e 16K token, verificando se il modello recupera correttamente un "ago" nel "pagliaio" anche con eviction.

Il contributo della tesi è geometrico, non di performance assoluta: ciò che conta non è la PPL assoluta ma la **differenza relativa** tra Q-filter e random eviction sullo stesso modello, e la **struttura geometrica** dei piani e delle query.

## Tests

```
# Tutti i test CPU
python -m pytest tests/ -k "not requires_gpu" -v

# Tutti i test (GPU richiesto per alcuni)
./tests/run_all.sh
```

## Citazione

Basato su:
- Clift et al., "Logic and the 2-Simplicial Transformer", 2019
- Roy et al., "Fast and Simplex: 2-Simplicial Attention in Triton", 2025
- Godey et al., "Q-filters", 2024