#!/bin/bash
# run_pipeline.sh — Pipeline completa simplex-filters su Vast.ai
#
# FASI:
#   0. Setup: installa requirements, Triton TLX, FBGEMM
#   1. Login HuggingFace + WandB (da variabili d'ambiente)
#   2. Pipeline completa: python main.py --both
#
# Usage:
#   export HF_TOKEN="hf_..." WANDB_API_KEY="..."
#   bash scripts/run_pipeline.sh
#
# Vast.ai template: pytorch/pytorch:2.4.0-cuda12.1-cudnn8-devel

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo ""
echo "=========================================="
echo "  simplex-filters — Pipeline completa"
echo "=========================================="
echo ""

# ==========================================================================
# FASE 0: Setup
# ==========================================================================
echo ""
echo "=========================================="
echo "  FASE 0: Setup"
echo "=========================================="

echo "  0a. Installazione requirements..."
pip install -r requirements.txt 2>&1 | tail -3

echo "  0b. Clonazione Triton TLX (kernel FBGEMM)..."
if [ ! -d "triton" ]; then
    git clone -b tlx https://github.com/facebookexperimental/triton.git 2>&1 | tail -3
    cd triton && pip install -e . --no-build-isolation 2>&1 | tail -3 && cd ..
    echo "  Triton TLX installato"
else
    echo "  Triton TLX gia' presente"
fi

echo "  0c. Verifica GPU..."
python3 -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}, Mem: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')"

# ==========================================================================
# FASE 1: Login
# ==========================================================================
echo ""
echo "=========================================="
echo "  FASE 1: Login HuggingFace + WandB"
echo "=========================================="

# HF_TOKEN (obbligatorio per scaricare LLaMA)
if [ -z "${HF_TOKEN:-}" ]; then
    echo "  ERRORE: HF_TOKEN non impostato"
    echo "  export HF_TOKEN=hf_yourtoken"
    exit 1
fi
echo "  [OK] HF_TOKEN impostato"

# WANDB_API_KEY (opzionale, ma raccomandato)
if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "  [OK] WANDB_API_KEY impostato"
    wandb login "$WANDB_API_KEY" 2>/dev/null || true
else
    echo "  [WARN] WANDB_API_KEY non impostato — solo logging stdout"
fi

# ==========================================================================
# FASE 2: Pipeline completa
# ==========================================================================
echo ""
echo "=========================================="
echo "  FASE 2: Training + Analisi + Benchmark"
echo "=========================================="
echo ""

python main.py --both --verbose

EXIT_CODE=$?

# ==========================================================================
# RIEPILOGO
# =========================================================================
echo ""
echo "=========================================="
echo "  PIPELINE COMPLETATA"
echo "  Exit code: $EXIT_CODE"
echo "  Checkpoint: ./checkpoints/trilinear/ e ./checkpoints/gram_det/"
echo "  Report test: ./test_results.txt"
echo "=========================================="
echo ""

exit $EXIT_CODE