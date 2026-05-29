#!/bin/bash
# run_pipeline.sh — Pipeline completa simplex-filters su Vast.ai
#
# FASI:
#   PRE-FLIGHT: Verifica HF_TOKEN e WANDB_API_KEY (PRIMA di qualsiasi setup)
#   FASE 0: Setup requirements, Triton TLX
#   FASE 1: Verifica GPU
#   FASE 2: python main.py --both
#
# Usage:
#   export HF_TOKEN="hf_..." WANDB_API_KEY="..."
#   bash scripts/run_pipeline.sh
#
# NOTA: I token vengono controllati PRIMA dell'installazione,
#       per non sprecare tempo/crediti GPU se mancano.
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
# PRE-FLIGHT: Controlla credenziali PRIMA di qualsiasi installazione
# ==========================================================================
echo "  PRE-FLIGHT: Verifica credenziali..."

if [ -z "${HF_TOKEN:-}" ]; then
    echo ""
    echo "  ========================================"
    echo "  ERRORE: HF_TOKEN non impostato"
    echo "  ========================================"
    echo ""
    echo "  Il token serve per scaricare LLaMA 3.1 8B da HuggingFace."
    echo "  Ottienilo da: https://huggingface.co/settings/tokens"
    echo ""
    echo "  Poi: export HF_TOKEN=hf_yourtoken"
    echo ""
    exit 1
fi
echo "  [OK] HF_TOKEN impostato"

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo ""
    echo "  ========================================"
    echo "  ERRORE: WANDB_API_KEY non impostato"
    echo "  ========================================"
    echo ""
    echo "  La chiave serve per loggare i risultati del training su WandB."
    echo "  Ottienila da: https://wandb.ai/authorize"
    echo ""
    echo "  Poi: export WANDB_API_KEY=your_wandb_key"
    echo ""
    exit 1
fi
echo "  [OK] WANDB_API_KEY impostato"
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

echo "  0b. Login WandB..."
wandb login "$WANDB_API_KEY" 2>/dev/null || echo "  [WARN] wandb login fallito, continuo..."

echo "  0c. Clonazione Triton TLX (kernel FBGEMM)..."
if [ ! -d "triton" ]; then
    git clone -b tlx https://github.com/facebookexperimental/triton.git 2>&1 | tail -3
    cd triton && pip install -e . --no-build-isolation 2>&1 | tail -3 && cd ..
    echo "  Triton TLX installato"
else
    echo "  Triton TLX gia' presente"
fi

echo "  0d. Verifica GPU..."
python3 -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}, Mem: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')"

# ==========================================================================
# FASE 1: Pipeline completa
# ==========================================================================
echo ""
echo "=========================================="
echo "  FASE 1: Training + Analisi + Benchmark"
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