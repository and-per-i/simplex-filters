#!/usr/bin/env python3
"""
main.py — Entry point principale per simplex-filters.

Modalità:
  Default (--test): Carica modello → Converti → Freeze → Test suite
  --finetune:       Carica modello → Converti → Freeze → Training su C4 → Valutazione

Usage:
    python main.py                              # solo test
    python main.py --finetune                   # finetuning completo
    python main.py --finetune --max-steps 5000  # finetuning con override
    python main.py --finetune --attention-type gram_det
    python main.py --level 1                     # solo test strutturali
    python main.py --verbose                     # output verboso
    python main.py --analyze ./checkpoints/trilinear/final  # analisi geometrica
    python main.py --benchmark ./checkpoints/trilinear/final  # benchmark eviction
    python main.py --ruler ./checkpoints/trilinear/final      # RULER NIAH benchmark
"""

import argparse
import os
import sys
import platform
import subprocess
from pathlib import Path

# ==========================================================================
# Configurazione
# ==========================================================================

MODEL_NAME = "meta-llama/Llama-3.1-8B"
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "llama-3.1-8b")
SIMPLICIAL_INDICES = [16, 20, 24, 28]

# Colori (ANSI)
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
BLUE = "\033[0;34m"
BOLD = "\033[1m"
NC = "\033[0m"


def print_step(label, msg, color=BLUE):
    print(f"\n  {color}{BOLD}[{label}]{NC} {msg}")


def print_ok(msg):
    print(f"  {GREEN}[OK]{NC} {msg}")


def print_warn(msg):
    print(f"  {YELLOW}[WARN]{NC} {msg}")


def print_err(msg):
    print(f"  {RED}[ERR]{NC} {msg}")


# ==========================================================================
# Step 1: Carica config e modello
# ==========================================================================

def ensure_config():
    """Verifica che la config locale sia presente."""
    config_file = os.path.join(CONFIG_DIR, "config.json")
    if not os.path.exists(config_file):
        print_step("1/5", f"Config non trovata in {CONFIG_DIR}, scarico...")
        from huggingface_hub import snapshot_download
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token is None:
            print_warn("HF_TOKEN non impostata. Imposta con: export HF_TOKEN=<token>")
            print_warn("Provo login da cache...")
        os.makedirs(CONFIG_DIR, exist_ok=True)
        snapshot_download(MODEL_NAME, local_dir=CONFIG_DIR, token=hf_token)
        print_ok(f"Config scaricata in {CONFIG_DIR}")
    else:
        print_step("1/5", f"Config trovata in {CONFIG_DIR}")
    return True


def load_model(real_weights=False):
    """Carica LLaMA 3.1 8B."""
    from transformers import LlamaConfig, AutoModelForCausalLM

    config = LlamaConfig.from_pretrained(CONFIG_DIR)
    print_ok("Config LLaMA 3.1 8B caricata")

    if real_weights:
        print_step("2/5", "Caricamento modello con pesi reali da HuggingFace (~30 GB)...")
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
            token=hf_token,
        )
        print_ok("Modello con pesi reali caricato")
    else:
        print_step("2/5", "Creazione modello con pesi casuali (nessun download 30 GB)")
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)
        print_ok("Modello con pesi casuali creato")

    return model


# ==========================================================================
# Step 2: Salva pesi originali
# ==========================================================================

def save_original_weights(model):
    """Salva i pesi di k_proj e v_proj PRIMA della conversione."""
    print_step("3/5", "Salvataggio pesi originali per test...")
    original_weights = {}
    for idx in SIMPLICIAL_INDICES:
        attn = model.model.layers[idx].self_attn
        original_weights[idx] = {
            "k_proj": attn.k_proj.weight.data.clone(),
            "v_proj": attn.v_proj.weight.data.clone(),
        }
    print_ok(f"Pesi di {len(SIMPLICIAL_INDICES)} layer salvati")
    return original_weights


# ==========================================================================
# Step 3: Converti in ibrido
# ==========================================================================

def convert_model(model, attention_type, alpha, w1, w2, gram_window):
    """Converte il modello in ibrido."""
    from src.modeling.convert_to_hybrid import convert_llama_to_hybrid

    print_step("4/5", f"Conversione in modello ibrido ({attention_type})...")
    model, converted = convert_llama_to_hybrid(
        model,
        simplicial_indices=SIMPLICIAL_INDICES,
        alpha=alpha,
        w1=w1,
        w2=w2,
        attention_type=attention_type,
        gram_window=gram_window,
    )
    print_ok(f"{len(converted)} layer convertiti: {converted}")
    return model, converted


# ==========================================================================
# Step 4: Freeze parametri
# ==========================================================================

def freeze_model(model, attention_type):
    """Applica freeze dei parametri."""
    from src.modeling.convert_to_hybrid import freeze_parameters

    print_step("5/5", f"Congelamento parametri ({attention_type})...")
    param_groups = freeze_parameters(
        model,
        simplicial_indices=SIMPLICIAL_INDICES,
        attention_type=attention_type,
    )
    print_ok(f"Parametri congelati: {len(param_groups)} gruppi")
    return param_groups


# ==========================================================================
# Step 5: Esegui test
# ==========================================================================

def run_tests(levels, verbose=False, stop_on_failure=False):
    """Esegue i test specificati."""
    import pytest

    targets = []
    if 1 in levels:
        targets.append("tests/level_1_structural/")
        targets.append("tests/test_gram_det_attention.py -k \"not requires_gpu\"")
    if 2 in levels:
        targets.append("tests/level_2_forward/")
    if 3 in levels:
        targets.append("tests/level_3_numerical/")

    print_step("5/5", f"Esecuzione test: livelli {levels}...")
    print()

    pytest_args = ["-v"] if verbose else []
    if stop_on_failure:
        pytest_args.append("-x")

    all_passed = True
    for target in targets:
        cmd = ["pytest", target] + pytest_args
        print(f"  {BLUE}→{NC} {' '.join(cmd)}")
        ret = subprocess.run(cmd, capture_output=not verbose)
        if ret.returncode != 0:
            all_passed = False
            if not verbose:
                print(ret.stdout.decode()[-500:])
                print(ret.stderr.decode()[-500:])

    return all_passed


# ==========================================================================
# Step 5: Analisi geometrica
# ==========================================================================

def run_analysis(checkpoint_path: str, verbose: bool = False):
    """
    Esegue l'analisi geometrica completa su un checkpoint addestrato.
    
    Carica il modello, estrae K1/K2/Q via hook su Wikitext-2,
    calcola: piano medio (Frechet), varianza geodesica, query media
    (Q-filters), relazione query-piano, anisotropia nel piano.
    """
    from src.geometry.analyzer import analyze_checkpoint, summarize_results

    print(f"\n{BOLD}{'=' * 60}{NC}")
    print(f"{BOLD}  ANALISI GEOMETRICA{NC}")
    print(f"{BOLD}  Checkpoint: {checkpoint_path}{NC}")
    print(f"{BOLD}{'=' * 60}{NC}\n")

    try:
        results = analyze_checkpoint(
            checkpoint_path=checkpoint_path,
            attention_type="simplicial",
            num_analysis_batches=5,
            seq_length=256,
            verbose=verbose,
        )
        summarize_results(results)
        return 0
    except Exception as e:
        print_err(f"Analisi geometrica fallita: {e}")
        return 1


# ==========================================================================
# Step 6: Finetuning
# ==========================================================================

def run_finetuning(args, output_subdir=None):
    """
    Esegue il finetuning del modello ibrido su C4.
    Carica pesi reali, converte, addestra, salva checkpoint.
    
    Args:
        args: namespace con tutti i parametri
        output_subdir: sottocartella per checkpoint (es. "trilinear", "gram_det")
    """
    from finetuning.train_hybrid import train
    from finetuning.utils.optimizer import create_optimizer_groups
    from finetuning.utils.wandb_utils import init_wandb, finish_wandb
    from transformers import AutoTokenizer

    attn_label = "TRILINEARE" if args.attention_type == "simplicial" else "GRAM DET"

    print(f"\n{BOLD}{'=' * 60}{NC}")
    print(f"{BOLD}  FINETUNING: {attn_label}{NC}")
    print(f"{BOLD}  Attenzione: {args.attention_type}{NC}")
    print(f"{BOLD}{'=' * 60}{NC}\n")

    # Carica config di training
    import yaml
    config_path = args.finetune_config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Override parametri da CLI
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    if args.lr_k2v2 is not None:
        config["lr_k2v2"] = args.lr_k2v2
    config["attention_type"] = args.attention_type
    config["alpha"] = args.alpha
    config["simplicial_indices"] = SIMPLICIAL_INDICES

    # Checkpoint in sottocartella separata
    if output_subdir:
        config["checkpoint_dir"] = os.path.join(
            os.path.dirname(config["checkpoint_dir"]), output_subdir
        )

    # Esegue training (carica modello, converte, freeze, dataset, loop)
    train(config)

    return 0


# ==========================================================================
# Main
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="simplex-filters: validazione e finetuning modello ibrido LLaMA + 2-Simplicial"
    )

    # Modalità
    parser.add_argument("--finetune", action="store_true",
                        help="Esegue il finetuning su C4 (implica --real-weights)")
    parser.add_argument("--both", action="store_true",
                        help="Esegue entrambi i training in sequenza: trilineare + Gram Det")
    parser.add_argument("--analyze", type=str, default=None,
                        help="Path a checkpoint da analizzare geometricamente")
    parser.add_argument("--benchmark", type=str, default=None,
                        help="Path a checkpoint per benchmark eviction Q-filter")
    parser.add_argument("--ruler", type=str, default=None,
                        help="Path a checkpoint per benchmark RULER NIAH")
    parser.add_argument("--finetune-config", type=str,
                        default="./finetuning/config.yaml",
                        help="Path configurazione finetuning (default: finetuning/config.yaml)")

    # Parametri modello
    parser.add_argument("--real-weights", action="store_true",
                        help="Scarica pesi reali da HuggingFace (~30 GB)")
    parser.add_argument("--attention-type", type=str, default="simplicial",
                        choices=["simplicial", "gram_det"],
                        help="Tipo di attenzione 2-simpliciale")

    # Parametri conversione
    parser.add_argument("--alpha", type=float, default=0.01,
                        help="Perturbazione K2/V2 (solo simplicial)")
    parser.add_argument("--w1", type=int, default=32,
                        help="Finestra K1 (solo simplicial)")
    parser.add_argument("--w2", type=int, default=256,
                        help="Finestra K2 (solo simplicial)")
    parser.add_argument("--gram-window", type=int, default=8,
                        help="Half-window per Gram Det")

    # Parametri finetuning (override della config YAML)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max_steps per finetuning")
    parser.add_argument("--lr-k2v2", type=float, default=None,
                        help="Override learning rate per K2/V2")
    parser.add_argument("--lr-k1v1", type=float, default=None,
                        help="Override learning rate per K1/V1")

    # Parametri test
    parser.add_argument("--level", type=int, nargs="+", default=[1, 2, 3],
                        choices=[1, 2, 3],
                        help="Livelli di test da eseguire (default: 1 2 3)")
    parser.add_argument("--verbose", action="store_true",
                        help="Output verboso dei test")
    parser.add_argument("--stop-on-failure", action="store_true",
                        help="Ferma al primo fallimento")

    args = parser.parse_args()

    # ======================================================================
    # MODALITA' FINETUNING
    # ======================================================================
    if args.finetune:
        ensure_config()
        return run_finetuning(args)

    # ======================================================================
    # MODALITA' BOTH: TRILINEARE + GRAM DET
    # ======================================================================
    if args.both:
        ensure_config()

        print(f"\n{BOLD}══════════════════════════════════════════════════════════════{NC}")
        print(f"{BOLD}  RUN 1/2: ATTENZIONE TRILINEARE{NC}")
        print(f"{BOLD}══════════════════════════════════════════════════════════════{NC}\n")
        args.attention_type = "simplicial"
        run_finetuning(args, output_subdir="trilinear")

        print(f"\n{BOLD}══════════════════════════════════════════════════════════════{NC}")
        print(f"{BOLD}  RUN 2/2: ATTENZIONE GRAM DET{NC}")
        print(f"{BOLD}══════════════════════════════════════════════════════════════{NC}\n")
        args.attention_type = "gram_det"
        run_finetuning(args, output_subdir="gram_det")

        return 0

    # ======================================================================
    # MODALITA' ANALISI GEOMETRICA
    # ======================================================================
    if args.analyze is not None:
        return run_analysis(args.analyze, verbose=args.verbose)

    # ======================================================================
    # MODALITA' BENCHMARK EVICTION
    # ======================================================================
    if args.benchmark is not None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import math

        print(f"\n{BOLD}{'=' * 60}{NC}")
        print(f"{BOLD}  BENCHMARK EVICTION{NC}")
        print(f"{BOLD}  Checkpoint: {args.benchmark}{NC}")
        print(f"{BOLD}{'=' * 60}{NC}\n")

        # Carica LLaMA base UNA SOLA VOLTA (fuori dal benchmark loop)
        print("Caricamento LLaMA base per baseline...")
        llama_base = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
        llama_base.eval()

        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        tokenizer.pad_token = tokenizer.eos_token

        # Calcola PPL LLaMA base sul validation set
        from src.kv_cache.benchmark import _eval_ppl_with_eviction
        from datasets import load_dataset
        
        val_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", streaming=True)
        ppl_llama_base = _eval_ppl_with_eviction(
            llama_base, tokenizer, val_dataset, None, 0.0, 0.0,
            budget=1.0, strategy="qfilter",
            seq_length=256, num_batches=5, device="cuda",
        )
        print(f"  Baseline LLaMA base PPL: {ppl_llama_base:.2f}")

        from src.kv_cache.benchmark import benchmark_checkpoint
        result = benchmark_checkpoint(
            checkpoint_path=args.benchmark,
            attention_type=args.attention_type,
            seq_length=256,
            num_batches=5,
            device="cuda",
            wandb_active=False,
            llama_base_ppl=ppl_llama_base,
            tokenizer=tokenizer,
        )
        return 0

    # ======================================================================
    # MODALITA' RULER BENCHMARK
    # ======================================================================
    if args.ruler is not None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.kv_cache.ruler.niah_benchmark import run_niah_benchmark
        
        print(f"\n{BOLD}{'=' * 60}{NC}")
        print(f"{BOLD}  RULER NIAH BENCHMARK{NC}")
        print(f"{BOLD}  Checkpoint: {args.ruler}{NC}")
        print(f"{BOLD}{'=' * 60}{NC}\n")

        model = AutoModelForCausalLM.from_pretrained(
            args.ruler,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        tokenizer.pad_token = tokenizer.eos_token

        result = run_niah_benchmark(
            model=model,
            tokenizer=tokenizer,
            checkpoint_path=args.ruler,
            attention_type=args.attention_type,
            device="cuda",
            wandb_active=False,
        )
        print(result.summary())
        return 0

    # ======================================================================
    # MODALITA' TEST / VALIDAZIONE
    # ======================================================================
    print(f"\n{BOLD}{'=' * 60}{NC}")
    print(f"{BOLD}  simplex-filters — Validazione modello ibrido{NC}")
    print(f"{BOLD}  Attenzione: {args.attention_type}{NC}")
    print(f"{BOLD}  Pesi reali: {args.real_weights}{NC}")
    print(f"{BOLD}{'=' * 60}{NC}\n")

    # Step 1: Config
    ensure_config()

    # Step 2: Modello
    try:
        model = load_model(real_weights=args.real_weights)
    except Exception as e:
        print_err(f"Caricamento modello fallito: {e}")
        return 1

    # Step 3: Pesi originali
    original_weights = save_original_weights(model)

    # Step 4: Conversione
    try:
        model, converted = convert_model(
            model, args.attention_type, args.alpha, args.w1, args.w2, args.gram_window,
        )
    except Exception as e:
        print_err(f"Conversione modello fallita: {e}")
        return 1

    # Step 5: Freeze
    try:
        freeze_model(model, args.attention_type)
    except Exception as e:
        print_warn(f"Freeze parametri fallito: {e} (non bloccante)")

    # Step 6: Test
    verbose_flag = args.verbose or (args.level == [1])
    ok = run_tests(args.level, verbose=verbose_flag, stop_on_failure=args.stop_on_failure)

    # Riepilogo
    print(f"\n{BOLD}{'=' * 60}{NC}")
    if ok:
        print(f"  {GREEN}{BOLD}✅ TUTTI I TEST SONO PASSATI.{NC}")
        ret = 0
    else:
        print(f"  {RED}{BOLD}❌ QUALCHE TEST HA FALLITO.{NC}")
        ret = 1
    print(f"{BOLD}{'=' * 60}{NC}\n")

    return ret


if __name__ == "__main__":
    sys.exit(main())