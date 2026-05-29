#!/bin/bash
# ==========================================================================
# run_all.sh - Test suite master per simplex-filters
#
# Esegue tutti i test in ordine di complessita' crescente.
# Le fasi critiche (strutturale, GramDet) bloccano il resto.
# I test GPU vengono skippati automaticamente se non disponibili.
#
# Usage:
#   ./tests/run_all.sh                    # esegue tutto
#   ./tests/run_all.sh --verbose          # output verboso
#   ./tests/run_all.sh --stop-on-failure  # ferma al primo fallimento
#   ./tests/run_all.sh --cpu-only         # solo test CPU (salta GPU)
# ==========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Configurazione
VERBOSE=""
STOP_ON_FAILURE=false
CPU_ONLY=false
PYTEST_OPTS=()

# Parsing argomenti
for arg in "$@"; do
    case "$arg" in
        --verbose) VERBOSE="-v" ;;
        --stop-on-failure) STOP_ON_FAILURE=true ;;
        --cpu-only) CPU_ONLY=true ;;
        *) PYTEST_OPTS+=("$arg") ;;
    esac
done

if [ "$STOP_ON_FAILURE" = true ]; then
    PYTEST_OPTS+=("-x")
fi

# Contatori
TOTAL=0
PASSED=0
FAILED=0
SKIPPED=0

# ==========================================================================
# Funzioni di utilita'
# ==========================================================================

print_header() {
    echo ""
    echo "=========================================="
    echo "  $1"
    echo "=========================================="
    echo ""
}

print_subheader() {
    echo "  -> $1"
}

print_pass() {
    echo "  [PASS] $1"
    PASSED=$((PASSED + 1))
}

print_fail() {
    echo "  [FAIL] $1"
    FAILED=$((FAILED + 1))
}

print_skip() {
    echo "  [SKIP] $1"
    SKIPPED=$((SKIPPED + 1))
}

run_test_group() {
    local name="$1"
    local target="$2"
    local is_critical="${3:-false}"
    local extra_opts="$4"

    TOTAL=$((TOTAL + 1))
    print_subheader "$name"

    local cmd="python -m pytest $target $VERBOSE ${PYTEST_OPTS[*]} $extra_opts"

    if [ "$VERBOSE" = "-v" ]; then
        echo "  $cmd"
        echo ""
    fi

    if eval "$cmd" 2>&1; then
        print_pass "$name"
    else
        print_fail "$name"
        if [ "$is_critical" = true ]; then
            echo ""
            echo "  FASE CRITICA FALLITA: $name"
            echo "  I test successivi non verranno eseguiti."
            return 1
        fi
    fi
    return 0
}

# ==========================================================================
# Inizio
# ==========================================================================

echo ""
echo "=========================================="
echo "  simplex-filters - Test Suite Master"
echo "  $(date)"
echo "=========================================="
echo ""

# Rileva GPU
if command -v python3 &> /dev/null; then
    GPU_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
else
    GPU_AVAILABLE="False"
fi

echo "  Python:  $(python3 --version 2>&1)"
echo "  GPU:     $GPU_AVAILABLE"
echo ""

if [ "$CPU_ONLY" = true ]; then
    echo "  Modalita' CPU-only. Test GPU saltati."
    echo ""
fi

# ==========================================================================
# FASE 1 - Sanity check strutturale (CPU)
# ==========================================================================

print_header "FASE 1: Sanity check strutturale (CPU)"

run_test_group \
    "Level 1 - Strutturale" \
    "tests/level_1_structural/" \
    true \
    "" || exit 1

# ==========================================================================
# FASE 2a - GramDetAttention (CPU, test di correttezza)
# ==========================================================================

print_header "FASE 2a: GramDetAttention - Correttezza (CPU)"

run_test_group \
    "GramDet - Pair indices + Sarrus + Forward/Backward + Score" \
    "tests/test_gram_det_attention.py" \
    true \
    "-k \"not requires_gpu\"" || exit 1

# ==========================================================================
# FASE 2b - Forward + Backward (GPU)
# ==========================================================================

print_header "FASE 2b: Forward + Backward (GPU - richiede Triton)"

if [ "$CPU_ONLY" = false ] && [ "$GPU_AVAILABLE" = "True" ]; then
    run_test_group \
        "Level 2 - Forward/Backward: gradienti, shape, frozen" \
        "tests/level_2_forward/" \
        false \
        "" || true
else
    print_skip "Level 2 - Forward/Backward (GPU assente o CPU-only)"
    SKIPPED=$((SKIPPED + 1))
fi

# ==========================================================================
# FASE 2c - Sanity check numerico (GPU)
# ==========================================================================

print_header "FASE 2c: Sanity check numerico (GPU)"

if [ "$CPU_ONLY" = false ] && [ "$GPU_AVAILABLE" = "True" ]; then
    run_test_group \
        "Level 3 - Numerico: cos-sim, K2 zero-out, alpha" \
        "tests/level_3_numerical/" \
        false \
        "" || true
else
    print_skip "Level 3 - Sanity check numerico (GPU assente o CPU-only)"
    SKIPPED=$((SKIPPED + 1))
fi

# ==========================================================================
# FASE 2d - GramDetAttention GPU speed benchmark
# ==========================================================================

print_header "FASE 2d: GramDetAttention - Speed benchmark (GPU)"

if [ "$CPU_ONLY" = false ] && [ "$GPU_AVAILABLE" = "True" ]; then
    run_test_group \
        "GramDet - Speed: N=256 W=8 B=4 < 3s + speedup" \
        "tests/test_gram_det_attention.py" \
        false \
        "-k \"requires_gpu\"" || true
else
    print_skip "GramDet - Speed benchmark (GPU assente o CPU-only)"
    SKIPPED=$((SKIPPED + 1))
fi

# ==========================================================================
# Riepilogo finale
# ==========================================================================

echo ""
echo "=========================================="
echo "              RIEPILOGO FINALE"
echo "=========================================="
echo ""
echo "  Totali:  $TOTAL gruppi di test"
echo "  Passati: $PASSED"
echo "  Falliti: $FAILED"
echo "  Saltati: $SKIPPED"
echo ""

if [ "$FAILED" -gt 0 ]; then
    echo "  ATTENZIONE: $FAILED gruppo/i di test hanno fallito."
    echo ""
    exit 1
else
    echo "  TUTTI I TEST SONO PASSATI."
    echo ""
    exit 0
fi