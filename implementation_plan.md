# Implementation Plan

[Overview]
Migrare il progetto da LLaMA 3.1 8B-hardcodato a modello-agnostico con target primario LLaMA 3.2 1B, full finetuning a 3 gruppi di learning rate, early stopping, e backward compatibility garantita.

Il progetto ha decine di hardcoding di MODEL_NAME, SIMPLICIAL_INDICES, HEAD_DIM, num_q_heads=32, CONFIG_DIR sparsi in 11 file Python più il config YAML. L'obiettivo è centralizzare la derivazione dell'architettura: tutto ciò che è architetturale (head_dim, hidden_size, num_layers, num_heads, GQA ratio) viene letto da `model.config` a runtime. Il config YAML contiene solo parametri configurabili dall'utente (model_name, simplicial_indices, iperparametri training). I default rimangono LLaMA 3.1 8B → nessun workflow esistente si rompe.

Per il training del 1B, si usano 3 gruppi di ottimizzazione: simpliciali (lr=2e-4), standard (lr=5e-6, tutti i parametri dei layer non-simpliciali inclusi FFN), frozen (embedding + lm_head). Early stopping se PPL su Wikitext-2 > 15. Niente LoRA: l'architettura è sufficientemente piccola (1B) per permettere full finetuning di tutti i layer con lr differenziato.

[Types]

Nessun nuovo tipo. Il dict `config` (da YAML) riceve due nuove chiavi opzionali:
```yaml
lr_standard: 5e-6              # learning rate per layer non-simpliciali (default: 1e-5)
early_stop_ppl: 15.0            # early stopping se PPL > soglia (default: null = disabilitato)
baseline_ppl: null               # se null, calcolato a runtime; se float, usato direttamente
```

Costanti di progetto (unico dizionario globale ammesso):
```python
GRASSMANN_BASELINE = {128: 4.09, 64: 3.85}  # head_dim → varianza geodesica attesa Gr(2,d)
```

Derivazione architettura standard (identica in ogni file che ne ha bisogno):
```python
head_dim = getattr(model.config, "head_dim", None) or (model.config.hidden_size // model.config.num_attention_heads)
hidden_size = model.config.hidden_size
num_layers = model.config.num_hidden_layers
num_q_heads = model.config.num_attention_heads
num_kv_heads = getattr(model.config, "num_key_value_heads", num_q_heads)
num_repeats = num_q_heads // num_kv_heads  # per espansione GQA
```

[Files]

Un nuovo file, 11 file modificati, nessun file cancellato.

### Nuovo file
- `scripts/baseline_ppl.py` — calcola PPL baseline su C4 o Wikitext-2 per qualsiasi modello, senza training. Utile per smoke test e per ottenere il valore di `baseline_ppl`.

### File modificati (in ordine di implementazione)

1. **`finetuning/config.yaml`** — aggiungere `lr_standard: 5e-6`, `early_stop_ppl: 15.0`. Rimuovere commenti che citano "8B". Non togliere campi esistenti.

2. **`finetuning/utils/optimizer.py`** — modificare `create_optimizer_groups()`:
   - Nuovo parametro: `lr_standard: float = 1e-5`
   - Nuovo gruppo `standard_params`: tutti i parametri `requires_grad=True` che NON sono in layer simpliciali e NON sono embedding/lm_head
   - Freezare esplicitamente `embed_tokens`, `lm_head` (qualsiasi nome contenga "embed" o "lm_head")
   - Logica esatta (step-by-step per ogni parametro):
     a) `not requires_grad` → frozen
     b) `"embed" in name or "lm_head" in name` → freeze, frozen
     c) `any(f"layers.{idx}." in name for idx in simplicial_indices)` → simplicial (k1v1/k2v2 o gram_det come prima)
     d) altrimenti → standard_params
   - Restituire lista di 4 dict: frozen (lr=0), standard (lr=lr_standard), k1v1 (lr=lr_k1v1), k2v2 (lr=lr_k2v2) per simplicial; o 3 per gram_det (frozen, standard, gram_det)

3. **`finetuning/train_hybrid.py`** — modifiche multiple:
   - Rimuovere `LLAMA_BASELINE_PPL = 9.45` e tutto il commento associato
   - In `train()`:
     a) Dopo `model = AutoModelForCausalLM.from_pretrained(...)`: estrarre `head_dim`, `hidden_size`, `num_heads` dal model.config
     b) `baseline_ppl = config.get("baseline_ppl")`; se None, chiamare `compute_ppl_on_c4()` (da definire localmente con forward pass su batch C4)
     c) Stampare "Baseline LLaMA su C4: PPL = {baseline_ppl:.2f}"
   - Chiamata a `create_optimizer_groups()`: passare anche `lr_standard=config["lr_standard"]`
   - Dopo validation: se `val_ppl > config.get("early_stop_ppl", float("inf"))`:
     ```python
     print(f"Early stopping: PPL {val_ppl:.2f} > {config['early_stop_ppl']}")
     break
     ```

4. **`scripts/validate_proxy.py`** — modifiche:
   - Rimuovere `MODEL_NAME`, `SIMPLICIAL_INDICES`, `HEAD_DIM`, `SCALE` dalle costanti globali
   - In `main()`: aggiungere `--model` (default `meta-llama/Llama-3.1-8B`), `--simplicial-indices` (default `16,20,24,28`)
   - `load_model()`: accettare MODEL_NAME e SIMPLICIAL_INDICES come parametri
   - `compute_true_score()`: `HEAD_DIM` → derivato da `Q.shape[-1]`; `num_q_heads=32` → derivato da hook (prima istanza: `K.shape[0] // batch_size // seq_length` o dal pattern dei dati hookati)
   - `SCALE` → derivato da `head_dim`: `SCALE = 1.0 / math.sqrt(head_dim)`

5. **`scripts/diagnose_checkpoint.py`** — modifiche:
   - Rimuovere `MODEL_NAME`, `SIMPLICIAL_INDICES` dalle costanti globali
   - In `main()`: aggiungere `--model`, `--simplicial-indices` (stessi default)
   - Tutte le funzioni che usano `SIMPLICIAL_INDICES` o `MODEL_NAME`: passarli come parametri

6. **`scripts/eval_wikitext.py`** — modifiche:
   - Rimuovere `MODEL_NAME`, `SIMPLICIAL_INDICES` da dentro la funzione `_load_and_convert_model()`
   - Passarli come parametri della funzione
   - In `main()`: aggiungere `--model`, `--simplicial-indices` (stessi default)

7. **`main.py`** — modifiche:
   - `MODEL_NAME`, `CONFIG_DIR`, `SIMPLICIAL_INDICES` → da argparse in `main()`
   - Ovunque sia usato `"meta-llama/Llama-3.1-8B"` (righe ~441, 448, 677, 684, 732): sostituire con `args.model` o variabile parametrizzata
   - `CONFIG_DIR` → derivare da `MODEL_NAME` o da argparse
   - Aggiungere costante: `GRASSMANN_BASELINE = {128: 4.09, 64: 3.85}` prima del `main()`
   - In `run_analyze()`: usare `GRASSMANN_BASELINE` per determinare baseline

8. **`src/geometry/analyzer.py`** — modifiche:
   - `base_model_path = "meta-llama/Llama-3.1-8B"` → parametro della funzione `analyze_model()`
   - `tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")` → parametro o derivato da model_name passato
   - Aggiungere mapping baseline: `GRASSMANN_BASELINE.get(head_dim, 4.0)` invece di hardcodato

9. **`src/training/finetune.py`** — modifiche:
   - `model_name: str = "meta-llama/Llama-3.1-8B"` → rimane uguale (è un default, giusto)

10. **`src/kv_cache/benchmark.py`** — modifiche:
    - `tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")` → parametro

11. **`scripts/grassmann_baseline.py`** — modifiche:
    - `parser.add_argument("--dim", type=int, default=128)` → `default=64`

12. **`tests/conftest.py`** — modifiche:
    - `MODEL_NAME = "meta-llama/Llama-3.1-8B"` → `os.environ.get("LLAMA_MODEL_NAME", "meta-llama/Llama-3.1-8B")`
    - `CONFIG_PATH` → `os.environ.get("LLAMA_CONFIG_DIR", "llama-3.1-8b")`
    - `SIMPLICIAL_INDICES = [16, 20, 24, 28]` → da env var

13. **`tests/level_2_forward/test_forward_backward.py`** — modifiche:
    - Trovare assert con `4096` e sostituire con `model.config.hidden_size`

[Functions]

### Nuove funzioni

**`scripts/baseline_ppl.py::compute_ppl()`**
```python
def compute_ppl(
    model_name: str,
    dataset: str,        # "c4" o "wikitext"
    max_samples: int = 50,
    seq_length: int = 256,
    device: str = "cuda",
    output: Optional[str] = None,
) -> float:
```
Forward pass only su max_samples batch, calcola NLL media, restituisce 2^NLL.

**`scripts/baseline_ppl.py::main()`** — CLI con `--model`, `--dataset`, `--max-samples`, `--seq-length`, `--output`.

### Funzioni modificate

**`finetuning/utils/optimizer.py::create_optimizer_groups()`**
- Nuovo parametro: `lr_standard: float = 1e-5` (dopo `lr_k1v1`)
- Aggiunge gruppo `standard_params` e freeza embedding+lm_head
- Restituisce 3-4 gruppi invece di 2-3

**`finetuning/train_hybrid.py::train()`**
- Aggiunge: derivazione architettura, baseline PPL runtime, early stopping, passaggio `lr_standard` all'optimizer

**`scripts/validate_proxy.py::load_model()`** — firma: `load_model(ckpt_path, attention_type, model_name, simplicial_indices, device)`

**`scripts/validate_proxy.py::compute_true_score()`** — head_dim e num_q_heads derivati da dati hookati

**`scripts/validate_proxy.py::main()`** — 2 nuovi argomenti CLI

**`scripts/diagnose_checkpoint.py::main()`** — 2 nuovi argomenti CLI

**`scripts/eval_wikitext.py::_load_and_convert_model()`** — MODIFIED_NAME e INDICES parametrizzati

**`main.py::run_analyze()`** — model_name, simplicial_indices, CONFIG_DIR parametrizzati

**`main.py::run_llama_base_baseline()`** — model_name parametrizzato

**`src/geometry/analyzer.py::analyze_model()`** — `base_model_path` parametrizzato

**`src/kv_cache/benchmark.py::benchmark_eviction()`** — model_name parametrizzato

### Funzioni rimosse
- `LLAMA_BASELINE_PPL` costante in `train_hybrid.py` (6 righe di commento + assegnazione)

[Classes]

Nessuna modifica a classi.

[Dependencies]

Nessuna nuova dipendenza. `accelerate` è già presente come dipendenza di transformers. `peft` NON serve più.

[Testing]

### Test esistenti modificati

**`tests/conftest.py`**: MODEL_NAME, CONFIG_PATH, SIMPLICIAL_INDICES → environment variables:
```bash
LLAMA_CONFIG_DIR=llama-3.1-8b LLAMA_MODEL_NAME=meta-llama/Llama-3.1-8B SIMPLICIAL_INDICES=16,20,24,28 pytest
```

**`tests/level_2_forward/test_forward_backward.py`**: sostituire tutti gli hardcoding di `4096` e `1024` con accesso a `model.config.hidden_size` e `model.config.hidden_size // 4`.

### Smoke test manuali (su Vast.ai dopo push)

1. `python scripts/baseline_ppl.py --model meta-llama/Llama-3.2-1B --dataset wikitext --max-samples 50` → atteso ~11.57
2. `pytest tests/ -x` → tutti verdi (8B backward compat)
3. Forward pass 1B: forward su batch piccolo dopo conversione ibrida
4. Backward pass 1B: gradienti non nulli su simpliciali E standard
5. `python finetuning/train_hybrid.py --config finetuning/config.yaml --max-steps 5` → 8B funziona
6. Training 100 step 1B con CLI args: loss scende

[Implementation Order]

1. `finetuning/config.yaml` — aggiungere `lr_standard`, `early_stop_ppl`
2. `finetuning/utils/optimizer.py` — 3 gruppi, freeze embedding+lm_head
3. `finetuning/train_hybrid.py` — architettura, early stopping, baseline runtime
4. `scripts/baseline_ppl.py` — nuovo file
5. `scripts/validate_proxy.py` — CLI args
6. `scripts/diagnose_checkpoint.py` — CLI args
7. `scripts/eval_wikitext.py` — CLI args
8. `main.py` — parametrizzazione globale + `GRASSMANN_BASELINE`
9. `src/geometry/analyzer.py` — parametrizzazione base_model_path
10. `src/kv_cache/benchmark.py` — parametrizzazione tokenizer
11. `scripts/grassmann_baseline.py` — default `--dim 64`
12. `tests/conftest.py` — env vars
13. `tests/level_2_forward/test_forward_backward.py` — assert generici
14. `src/training/finetune.py` — default aggiornato (solo se serve)
15. Git commit + push