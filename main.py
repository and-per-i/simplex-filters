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

import torch

# ==========================================================================
# Costanti globali
# ==========================================================================

# Baseline Monte Carlo per varianza geodesica su Gr(2, d)
# Calcolata con scripts/grassmann_baseline.py --dim <d> --runs 10
GRASSMANN_BASELINE = {128: 4.09, 64: 3.85}

# Colori (ANSI)
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
BLUE = "\033[0;34m"
BOLD = "\033[1m"
NC = "\033[0m"


def free_gpu_memory(*models):
    """Libera memoria GPU tra modelli diversi per evitare OOM."""
    import gc
    import torch
    for m in models:
        if m is not None:
            del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print_ok("Memoria GPU liberata")


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

def ensure_config(model_name: str, config_dir: str):
    """Verifica che la config locale sia presente."""
    config_file = os.path.join(config_dir, "config.json")
    if not os.path.exists(config_file):
        print_step("1/5", f"Config non trovata in {config_dir}, scarico...")
        from huggingface_hub import snapshot_download
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token is None:
            print_warn("HF_TOKEN non impostata. Imposta con: export HF_TOKEN=<token>")
            print_warn("Provo login da cache...")
        os.makedirs(config_dir, exist_ok=True)
        snapshot_download(model_name, local_dir=config_dir, token=hf_token)
        print_ok(f"Config scaricata in {config_dir}")
    else:
        print_step("1/5", f"Config trovata in {config_dir}")
    return True


def load_model(model_name: str, config_dir: str, real_weights=False):
    """Carica modello LLaMA."""
    from transformers import LlamaConfig, AutoModelForCausalLM

    config = LlamaConfig.from_pretrained(config_dir)
    print_ok(f"Config {model_name} caricata")

    if real_weights:
        print_step("2/5", "Caricamento modello con pesi reali da HuggingFace (~30 GB)...")
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
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

def save_original_weights(model, simplicial_indices):
    """Salva i pesi di k_proj e v_proj PRIMA della conversione."""
    print_step("3/5", "Salvataggio pesi originali per test...")
    original_weights = {}
    for idx in simplicial_indices:
        attn = model.model.layers[idx].self_attn
        original_weights[idx] = {
            "k_proj": attn.k_proj.weight.data.clone(),
            "v_proj": attn.v_proj.weight.data.clone(),
        }
    print_ok(f"Pesi di {len(simplicial_indices)} layer salvati")
    return original_weights


# ==========================================================================
# Step 3: Converti in ibrido
# ==========================================================================

def convert_model(model, simplicial_indices, attention_type, alpha, w1, w2, gram_window):
    """Converte il modello in ibrido."""
    from src.modeling.convert_to_hybrid import convert_llama_to_hybrid

    print_step("4/5", f"Conversione in modello ibrido ({attention_type})...")
    model, converted = convert_llama_to_hybrid(
        model,
        simplicial_indices=simplicial_indices,
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

def freeze_model(model, simplicial_indices, attention_type):
    """Applica freeze dei parametri."""
    from src.modeling.convert_to_hybrid import freeze_parameters

    print_step("5/5", f"Congelamento parametri ({attention_type})...")
    param_groups = freeze_parameters(
        model,
        simplicial_indices=simplicial_indices,
        attention_type=attention_type,
    )
    print_ok(f"Parametri congelati: {len(param_groups)} gruppi")
    return param_groups


# ==========================================================================
# Step 5: Esegui test
# ==========================================================================

def run_tests(levels, verbose=False, stop_on_failure=False, model_name="random", model_type="hybrid"):
    """
    Esegue i test specificati e scrive un report su test_results.txt.
    """
    import pytest
    import re

    if model_type == "llama_base":
        print_step("5/5", "LLaMA base: nessun test strutturale (skip)")
        print(f"  {YELLOW}[INFO]{NC} I test per LLaMA base sono solo perplexity + RULER,")
        print(f"  eseguiti dalle modalita' dedicate.")
        return True

    targets = []
    sections = []
    
    if 1 in levels:
        targets.append("tests/level_1_structural/")
        sections.append("LEVEL 1 — STRUTTURALE")
        targets.append("tests/test_gram_det_attention.py -k \"not requires_gpu\"")
        sections.append("LEVEL 1 — GRAMDET")
        targets.append("tests/test_geometry.py")
        sections.append("LEVEL 1 — GEOMETRIA")
        targets.append("tests/test_kv_cache.py")
        sections.append("LEVEL 1 — KV CACHE")
        targets.append("tests/test_ruler.py")
        sections.append("LEVEL 1 — RULER")
    if 2 in levels:
        targets.append("tests/level_2_forward/")
        sections.append("LEVEL 2 — FORWARD + BACKWARD")
    if 3 in levels:
        targets.append("tests/level_3_numerical/")
        sections.append("LEVEL 3 — NUMERICO")

    model_tag = {"trilinear": "TRILINEARE", "gram_det": "GRAM DET", "hybrid": "IBRIDO"}.get(model_type, model_name)
    print_step("5/5", f"Esecuzione test: {model_tag} — livelli {levels}...")
    print()

    all_output = []
    total_passed = 0
    total_failed = 0
    total_skipped = 0

    for target in targets:
        cmd = ["pytest", target, "--tb=short", "-q"] + (["-x"] if stop_on_failure else []) + (["-v"] if verbose else [])
        
        if verbose:
            print(f"  {BLUE}→{NC} {' '.join(cmd)}")
        
        ret = subprocess.run(cmd, capture_output=True, text=True)
        output = ret.stdout + ret.stderr
        
        for line in output.split("\n"):
            m = re.search(r'(PASSED|FAILED|SKIPPED|ERROR)', line)
            if m and ("::" in line or "test_" in line):
                status = m.group(1)
                if status == "PASSED":
                    total_passed += 1
                elif status in ("FAILED", "ERROR"):
                    total_failed += 1
                else:
                    total_skipped += 1
        
        all_output.append(output)
        
        if verbose:
            print(output[-1000:] if len(output) > 1000 else output)

        if ret.returncode != 0 and not stop_on_failure:
            error_lines = [l for l in output.split("\n") if "FAILED" in l or "ERROR" in l]
            if error_lines:
                print(f"  {RED}Errori:{NC}")
                for err in error_lines[:5]:
                    print(f"    {err}")

    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report_path = os.path.join(os.path.dirname(__file__), "test_results.txt")
    separator = "=" * 74
    
    with open(report_path, "a") as f:
        f.write(f"\n{separator}\n")
        f.write(f"  simplex-filters — Test Results\n")
        f.write(f"  Modello: {model_name}\n")
        f.write(f"  Data: {now}\n")
        f.write(f"{separator}\n")
        
        for idx, output in enumerate(all_output):
            section_name = sections[idx] if idx < len(sections) else f"TEST {idx}"
            f.write(f"\n[{section_name}]\n")
            
            for line in output.split("\n"):
                line_stripped = line.strip()
                if "::" in line_stripped and ("PASSED" in line_stripped or "FAILED" in line_stripped or "SKIPPED" in line_stripped):
                    parts = line_stripped.split("::")
                    test_name = parts[-1] if len(parts) > 1 else line_stripped
                    symbol = {"PASSED": "✔", "FAILED": "✘", "SKIPPED": "⊘", "ERROR": "✘"}
                    status_symbol = symbol.get(re.search(r'(PASSED|FAILED|SKIPPED|ERROR)', line_stripped).group(1), "?")
                    f.write(f"  {status_symbol} {test_name}\n")
        
        total = total_passed + total_failed + total_skipped
        f.write(f"\n{separator}\n")
        f.write(f"  SUMMARY\n")
        f.write(f"  Total: {total} | Passed: {total_passed} | Failed: {total_failed} | Skipped: {total_skipped}\n")
        f.write(f"{separator}\n\n")
    
    all_passed = total_failed == 0
    
    print(f"\n  {GREEN}Passed: {total_passed}{NC} {RED}Failed: {total_failed}{NC} {YELLOW}Skipped: {total_skipped}{NC}")
    print(f"  Report salvato: {report_path}")
    
    return all_passed


# ==========================================================================
# Step 5: Analisi geometrica
# ==========================================================================

def run_analysis(checkpoint_path: str, verbose: bool = False, device: str = "cuda", attention_type: str = "simplicial",
                 model_name: str = "meta-llama/Llama-3.1-8B", simplicial_indices: list = [16, 20, 24, 28]):
    """
    Esegue l'analisi geometrica completa su un checkpoint addestrato.
    """
    from src.geometry.analyzer import analyze_checkpoint, summarize_results

    abs_path = os.path.abspath(checkpoint_path)

    print(f"\n{BOLD}{'=' * 60}{NC}")
    print(f"{BOLD}  ANALISI GEOMETRICA{NC}")
    print(f"{BOLD}  Checkpoint: {abs_path}{NC}")
    print(f"{BOLD}  Modello: {model_name}{NC}")
    print(f"{BOLD}  Layer simpliciali: {simplicial_indices}{NC}")
    print(f"{BOLD}  Attention type: {attention_type}{NC}")
    print(f"{BOLD}  Device: {device}{NC}")
    print(f"{BOLD}{'=' * 60}{NC}\n")

    print(f"  {YELLOW}[INFO]{NC} L'analisi su CPU e' piu' lenta (~2-3 minuti invece di 30 secondi)")
    print(f"  {YELLOW}[INFO]{NC} ma non compete per VRAM col training in corso.\n")

    try:
        results = analyze_checkpoint(
            checkpoint_path=abs_path,
            attention_type=attention_type,
            base_model_path=model_name,
            simplicial_indices=simplicial_indices,
            num_analysis_batches=5,
            seq_length=256,
            device=device,
            verbose=verbose,
        )
        summarize_results(results)
        return 0
    except Exception as e:
        import traceback
        traceback.print_exc()
        print_err(f"Analisi geometrica fallita: {e}")
        return 1


# ==========================================================================
# Step 6: Test su checkpoint addestrato
# ==========================================================================

def run_test_checkpoint(checkpoint_path: str, attention_type: str, verbose: bool = False):
    """
    Carica un checkpoint addestrato, esegue i test strutturali/forward/numerici,
    e scrive il report su test_results.txt.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.modeling.convert_to_hybrid import convert_llama_to_hybrid, freeze_parameters

    abs_path = os.path.abspath(checkpoint_path)

    print(f"\n{BOLD}{'=' * 60}{NC}")
    print(f"{BOLD}  TEST SU CHECKPOINT: {abs_path}{NC}")
    print(f"{BOLD}{'=' * 60}{NC}\n")

    model = AutoModelForCausalLM.from_pretrained(
        abs_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        local_files_only=True,
    )
    model.eval()

    model_type = "trilinear" if attention_type == "simplicial" else "gram_det"
    ok = run_tests(
        levels=[1, 2, 3],
        model_name=f"checkpoint_{model_type}",
        model_type=model_type,
        verbose=verbose,
        stop_on_failure=False,
    )

    return 0 if ok else 1


def run_llama_base_baseline(model_name: str):
    """
    Calcola baseline perplexity + RULER per modello base puro.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.kv_cache.benchmark import _eval_ppl_with_eviction
    from src.kv_cache.ruler.niah_benchmark import generate_single_niah, extract_answer, check_answer
    from datasets import load_dataset
    import math

    print(f"\n{BOLD}{'=' * 60}{NC}")
    print(f"{BOLD}  BASELINE LLaMA BASE{NC}")
    print(f"{BOLD}{'=' * 60}{NC}\n")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    val_dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test", streaming=True)
    ppl_llama = _eval_ppl_with_eviction(
        model, tokenizer, val_dataset, None, 0.0, 0.0,
        budget=1.0, strategy="qfilter",
        seq_length=256, num_batches=5, device="cuda",
    )
    print(f"  Baseline LLaMA PPL: {ppl_llama:.2f}")

    correct = 0
    total = 10
    for test_idx in range(total):
        input_ids, expected_answer, prompt = generate_single_niah(
            tokenizer, context_len=8192, needle_pos=0.5,
        )
        input_ids = input_ids.unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            logits = model(input_ids).logits
        predicted = extract_answer(logits, tokenizer)
        if check_answer(predicted, expected_answer):
            correct += 1
    ruler_acc = correct / total
    print(f"  Baseline RULER accuracy (8K): {ruler_acc*100:.1f}% ({correct}/{total})")

    free_gpu_memory(model)
    return ppl_llama, ruler_acc


# ==========================================================================
# Step 7: Finetuning
# ==========================================================================
def run_finetuning(args, model_name, simplicial_indices, output_subdir=None):
    """
    Esegue il finetuning del modello ibrido su C4.
    """
    if output_subdir is None:
        output_subdir = "gram_det" if args.attention_type == "gram_det" else "trilinear"
    
    from finetuning.train_hybrid import train
    from finetuning.utils.wandb_utils import init_wandb, finish_wandb

    attn_label = "TRILINEARE" if args.attention_type == "simplicial" else "GRAM DET"

    print(f"\n{BOLD}{'=' * 60}{NC}")
    print(f"{BOLD}  FINETUNING: {attn_label}{NC}")
    print(f"{BOLD}  Attenzione: {args.attention_type}{NC}")
    print(f"{BOLD}{'=' * 60}{NC}\n")

    import yaml
    config_path = args.finetune_config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    if args.lr_k2v2 is not None:
        config["lr_k2v2"] = args.lr_k2v2
    config["attention_type"] = args.attention_type
    config["alpha"] = args.alpha
    config["model_name"] = model_name
    config["simplicial_indices"] = simplicial_indices
    if args.resume is not None:
        config["resume_path"] = os.path.abspath(args.resume)

    if output_subdir:
        config["checkpoint_dir"] = os.path.join(
            os.path.dirname(config["checkpoint_dir"]), output_subdir
        )

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
    parser.add_argument("--analyze-llama", action="store_true",
                        help="Analisi geometrica su LLaMA puro (nessuna conversione)")
    parser.add_argument("--benchmark", type=str, default=None,
                        help="Path a checkpoint per benchmark eviction Q-filter")
    parser.add_argument("--ruler", type=str, default=None,
                        help="Path a checkpoint per benchmark RULER NIAH")
    parser.add_argument("--test-checkpoint", type=str, default=None,
                        help="Path a checkpoint da testare (strutturali/forward/numerico)")
    parser.add_argument("--finetune-config", type=str,
                        default="./finetuning/config.yaml",
                        help="Path configurazione finetuning (default: finetuning/config.yaml)")

    # Parametri modello
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B",
                        help="Nome del modello HuggingFace (default: meta-llama/Llama-3.1-8B)")
    parser.add_argument("--config-dir", type=str, default=None,
                        help="Directory config modello (default: derivata da nome modello)")
    parser.add_argument("--simplicial-indices", type=str, default="16,20,24,28",
                        help="Indici layer simpliciali, separati da virgola (default: 16,20,24,28)")
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

    # Parametri finetuning
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max_steps per finetuning")
    parser.add_argument("--lr-k2v2", type=float, default=None,
                        help="Override learning rate per K2/V2")
    parser.add_argument("--lr-k1v1", type=float, default=None,
                        help="Override learning rate per K1/V1")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path a checkpoint da cui riprendere training")

    # Parametri test
    parser.add_argument("--level", type=int, nargs="+", default=[1, 2, 3],
                        choices=[1, 2, 3],
                        help="Livelli di test da eseguire (default: 1 2 3)")
    parser.add_argument("--verbose", action="store_true",
                        help="Output verboso dei test")
    parser.add_argument("--stop-on-failure", action="store_true",
                        help="Ferma al primo fallimento")
    parser.add_argument("--analyze-cpu", action="store_true",
                        help="Forza analisi geometrica su CPU (quando GPU e' occupata dal training)")

    args = parser.parse_args()

    # Deriva parametri
    simplicial_indices = [int(x) for x in args.simplicial_indices.split(",")]
    model_name = args.model
    config_dir = args.config_dir
    if config_dir is None:
        # Deriva directory config dal nome modello (ultimo segmento dopo /)
        config_dir = os.path.join(
            os.path.dirname(__file__),
            model_name.split("/")[-1].lower()
        )

    # ======================================================================
    # MODALITA' FINETUNING
    # ======================================================================
    if args.finetune:
        ensure_config(model_name, config_dir)
        return run_finetuning(args, model_name, simplicial_indices)

    # ======================================================================
    # MODALITA' BOTH: TRILINEARE + GRAM DET
    # ======================================================================
    if args.both:
        ensure_config(model_name, config_dir)

        print(f"\n{BOLD}══════════════════════════════════════════════════════════════{NC}")
        print(f"{BOLD}  FASE 1/2: FINETUNING TRILINEARE{NC}")
        print(f"{BOLD}══════════════════════════════════════════════════════════════{NC}\n")
        args.attention_type = "simplicial"
        run_finetuning(args, model_name, simplicial_indices, output_subdir="trilinear")

        ckpt_path = os.path.abspath(os.path.join(".", "trilinear", "final"))
        run_analysis(ckpt_path, verbose=args.verbose, model_name=model_name, simplicial_indices=simplicial_indices)
        run_test_checkpoint(ckpt_path, "trilinear", verbose=args.verbose)
        free_gpu_memory()

        print(f"\n{BOLD}══════════════════════════════════════════════════════════════{NC}")
        print(f"{BOLD}  FASE 2/2: FINETUNING GRAM DET{NC}")
        print(f"{BOLD}══════════════════════════════════════════════════════════════{NC}\n")
        args.attention_type = "gram_det"
        run_finetuning(args, model_name, simplicial_indices, output_subdir="gram_det")

        ckpt_path_gd = os.path.abspath(os.path.join(".", "gram_det", "final"))
        run_analysis(ckpt_path_gd, verbose=args.verbose, model_name=model_name, simplicial_indices=simplicial_indices)
        run_test_checkpoint(ckpt_path_gd, "gram_det", verbose=args.verbose)
        free_gpu_memory()

        return 0

    # ======================================================================
    # MODALITA' ANALISI LLaMA PURO
    # ======================================================================
    if args.analyze_llama:
        from src.geometry.analyzer import analyze_llama_pure, summarize_results

        # Auto-deriva layer indices: prendi i 4 layer centrali
        if args.simplicial_indices == "16,20,24,28":
            # Solo per LLaMA 1B (16 layer), auto-deriva
            from transformers import AutoConfig
            try:
                cfg = AutoConfig.from_pretrained(model_name)
                num_layers = cfg.num_hidden_layers
                mid = num_layers // 2
                simplicial_indices = [mid - 3, mid - 1, mid + 1, mid + 3]
                if args.verbose:
                    print(f"  [INFO] Auto-derivati layer simpliciali: {simplicial_indices} ({num_layers} layer totali)")
            except Exception:
                pass  # usa default

        device = "cpu" if args.analyze_cpu else "cuda"
        print(f"\n{BOLD}{'=' * 60}{NC}")
        print(f"{BOLD}  ANALISI LLaMA PURO (nessuna conversione){NC}")
        print(f"{BOLD}  Modello: {model_name}{NC}")
        print(f"{BOLD}  Layer: {simplicial_indices}{NC}")
        print(f"{BOLD}  Device: {device}{NC}")
        print(f"{BOLD}{'=' * 60}{NC}\n")

        try:
            results = analyze_llama_pure(
                model_name=model_name,
                simplicial_indices=simplicial_indices,
                num_analysis_batches=5,
                seq_length=256,
                device=device,
                verbose=args.verbose,
            )
            summarize_results(results)
            return 0
        except Exception as e:
            import traceback
            traceback.print_exc()
            print_err(f"Analisi LLaMA puro fallita: {e}")
            return 1

    # ======================================================================
    # MODALITA' ANALISI GEOMETRICA
    # ======================================================================
    if args.analyze is not None:
        device = "cpu" if args.analyze_cpu else "cuda"
        return run_analysis(
            args.analyze,
            verbose=args.verbose,
            device=device,
            attention_type=args.attention_type,
            model_name=model_name,
            simplicial_indices=simplicial_indices,
        )

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

        print("Caricamento LLaMA base per baseline...")
        llama_base = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
        llama_base.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        from src.kv_cache.benchmark import _eval_ppl_with_eviction
        from datasets import load_dataset
        
        val_dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test", streaming=True)
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

        tokenizer = AutoTokenizer.from_pretrained(model_name)
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
    # MODALITA' TEST SU CHECKPOINT
    # ======================================================================
    if args.test_checkpoint is not None:
        return run_test_checkpoint(
            args.test_checkpoint,
            attention_type=args.attention_type,
            verbose=args.verbose,
        )

    # ======================================================================
    # MODALITA' TEST / VALIDAZIONE
    # ======================================================================
    print(f"\n{BOLD}{'=' * 60}{NC}")
    print(f"{BOLD}  simplex-filters — Validazione modello ibrido{NC}")
    print(f"{BOLD}  Modello: {model_name}{NC}")
    print(f"{BOLD}  Attenzione: {args.attention_type}{NC}")
    print(f"{BOLD}  Pesi reali: {args.real_weights}{NC}")
    print(f"{BOLD}{'=' * 60}{NC}\n")

    # Step 1: Config
    ensure_config(model_name, config_dir)

    # Step 2: Modello
    try:
        model = load_model(model_name, config_dir, real_weights=args.real_weights)
    except Exception as e:
        print_err(f"Caricamento modello fallito: {e}")
        return 1

    # Step 3: Pesi originali
    original_weights = save_original_weights(model, simplicial_indices)

    # Step 4: Conversione
    try:
        model, converted = convert_model(
            model, simplicial_indices, args.attention_type, args.alpha, args.w1, args.w2, args.gram_window,
        )
    except Exception as e:
        print_err(f"Conversione modello fallita: {e}")
        return 1

    # Step 5: Freeze
    try:
        freeze_model(model, simplicial_indices, args.attention_type)
    except Exception as e:
        print_warn(f"Freeze parametri fallito: {e} (non bloccante)")

    # Step 6: Test
    verbose_flag = args.verbose or (args.level == [1])
    model_type = "llama_base" if not args.real_weights else args.attention_type
    ok = run_tests(args.level, verbose=verbose_flag, stop_on_failure=args.stop_on_failure,
                   model_type=model_type)

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