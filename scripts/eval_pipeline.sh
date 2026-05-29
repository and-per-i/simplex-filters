#!/bin/bash
# eval_pipeline.sh — Valutazione completa su checkpoint gia' addestrati
#
# Per ogni checkpoint, esegue:
#   1. Analisi geometrica
#   2. Benchmark eviction (perplexity vs budget)
#   3. RULER NIAH
#   4. Test strutturali
#
# Usage:
#   bash scripts/eval_pipeline.sh ./checkpoints/trilinear/final ./checkpoints/gram_det/final

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <checkpoint1> [checkpoint2 ...]"
    echo "       bash scripts/eval_pipeline.sh ./checkpoints/trilinear/final"
    exit 1
fi

for CKPT in "$@"; do
    if [ ! -d "$CKPT" ]; then
        echo "  ERRORE: $CKPT non trovato. Salto."
        continue
    fi

    # Determina attention_type dal path
    if echo "$CKPT" | grep -q "trilinear"; then
        ATT_TYPE="simplicial"
    elif echo "$CKPT" | grep -q "gram_det"; then
        ATT_TYPE="gram_det"
    else
        ATT_TYPE="simplicial"
    fi

    echo ""
    echo "=========================================="
    echo "  Checkpoint: $CKPT ($ATT_TYPE)"
    echo "=========================================="

    python main.py --analyze "$CKPT"
    python main.py --benchmark "$CKPT" --attention-type "$ATT_TYPE"
    python main.py --ruler "$CKPT" --attention-type "$ATT_TYPE"
    python main.py --test-model "$CKPT" --attention-type "$ATT_TYPE" --verbose
done

echo ""
echo "=========================================="
echo "  Valutazione completata"
echo "  Report: ./test_results.txt"
echo "=========================================="