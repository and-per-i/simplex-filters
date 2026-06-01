#!/bin/bash
# post_train_eval.sh — Esecuzione analisi + benchmark dopo training su Vast.ai
#
# FASI:
#   FASE 0: Verifica parametri essenziali
#   FASE 1: Baseline C4 (PPL LLaMA base su C4 validation)
#   FASE 2: Analisi geometrica
#   FASE 3: Benchmark KV cache eviction (Q-filter vs random)
#   FASE 4: RULER NIAH
#
# Usage (su Vast.ai):
#   export HF_TOKEN="hf_..."
#   bash scripts/post_train_eval.sh
#
# Usage (locale, con checkpoint alternativo):
#   bash scripts/post_train_eval.sh ./checkpoints/mio_run/final

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Colori
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

echo ""
echo -e "${BOLD}==========================================${NC}"
echo -e "${BOLD}  simplex-filters — Post-training Eval${NC}"
echo -e "${BOLD}==========================================${NC}"
echo ""

# ==========================================================================
# Parametri
# ==========================================================================
CHECKPOINT="${1:-./checkpoints/final}"
ATT_TYPE="${ATT_TYPE:-simplicial}"

if echo "$CHECKPOINT" | grep -q "trilinear"; then
    ATT_TYPE="simplicial"
elif echo "$CHECKPOINT" | grep -q "gram_det"; then
    ATT_TYPE="gram_det"
fi

if [ ! -d "$CHECKPOINT" ]; then
    echo -e "${YELLOW}[WARN]${NC} Checkpoint '$CHECKPOINT' non trovato."
    echo "       Specifica un path valido:"
    echo "       bash scripts/post_train_eval.sh ./checkpoints/final"
    exit 1
fi

echo -e "  Checkpoint: ${BOLD}$CHECKPOINT${NC}"
echo -e "  Attenzione: ${BOLD}$ATT_TYPE${NC}"
echo ""

# ==========================================================================
# FASE 1: Baseline C4 — calcola PPL reale di LLaMA base su C4 validation
# ==========================================================================
echo -e "${BOLD}──────────────────────────────────────────${NC}"
echo -e "${BOLD}  FASE 1: Baseline C4 (LLaMA base su C4)${NC}"
echo -e "${BOLD}──────────────────────────────────────────${NC}"
echo ""
echo -e "  ${YELLOW}[INFO]${NC} Calcolo PPL baseline LLaMA base su C4 validation..."
echo -e "  ${YELLOW}[INFO]${NC} Questo serve per confronto equo (validazione training era su C4)"
echo ""

# Crea script Python temporaneo per calcolare baseline C4
python3 -c "
import torch, math, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

print('Caricamento LLaMA base...')
model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-3.1-8B',
    torch_dtype=torch.bfloat16,
    device_map='auto',
    attn_implementation='eager',
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B')
tokenizer.pad_token = tokenizer.eos_token

print('Caricamento C4 validation...')
dataset = load_dataset('allenai/c4', 'en', split='validation', streaming=True)

total_loss = 0.0
total_tokens = 0
batch_count = 0
max_batches = 500  # stessi 500 campioni della validazione

for example in dataset:
    if batch_count >= max_batches:
        break
    text = example.get('text', '')
    if not text.strip():
        continue
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=513)
    if len(tokens) < 513:
        continue
    input_ids = torch.tensor(tokens[:512], dtype=torch.long, device=model.device).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()
    total_loss += loss * 511
    total_tokens += 511
    batch_count += 1

avg_nll = total_loss / max(total_tokens, 1)
baseline_ppl = math.exp(avg_nll)
print(f'')
print(f'  Baseline LLaMA 3.1 8B su C4 validation: {baseline_ppl:.2f} PPL')
print(f'  Campioni: {batch_count}')
print(f'')
" 2>&1

echo -e "${GREEN}[OK]${NC} Baseline C4 completata."
echo ""

# ==========================================================================
# FASE 2: Analisi geometrica
# ==========================================================================
echo -e "${BOLD}──────────────────────────────────────────${NC}"
echo -e "${BOLD}  FASE 2: Analisi geometrica${NC}"
echo -e "${BOLD}──────────────────────────────────────────${NC}"
echo ""

python3 -c "
print('Anisotropia piani K1-K2 (σ₁/σ₂)')
print('Fréchet mean + varianza geodesica')
print('Ortogonalità query vs piano medio')
print('Questi tre numeri sono il contributo teorico, indipendenti dalla PPL assoluta')
"
python main.py --analyze "$CHECKPOINT" --verbose

echo -e "${GREEN}[OK]${NC} Analisi geometrica completata."
echo ""

# ==========================================================================
# FASE 3: Benchmark KV cache
# ==========================================================================
echo -e "${BOLD}──────────────────────────────────────────${NC}"
echo -e "${BOLD}  FASE 3: Benchmark KV cache eviction${NC}"
echo -e "${BOLD}──────────────────────────────────────────${NC}"
echo ""

python main.py --benchmark "$CHECKPOINT" --attention-type "$ATT_TYPE"

echo -e "${GREEN}[OK]${NC} Benchmark KV cache completato."
echo ""

# ==========================================================================
# FASE 4: RULER NIAH
# ==========================================================================
echo -e "${BOLD}──────────────────────────────────────────${NC}"
echo -e "${BOLD}  FASE 4: RULER NIAH${NC}"
echo -e "${BOLD}──────────────────────────────────────────${NC}"
echo ""

python main.py --ruler "$CHECKPOINT" --attention-type "$ATT_TYPE"

echo -e "${GREEN}[OK]${NC} RULER NIAH completato."
echo ""

# ==========================================================================
# RIEPILOGO
# ==========================================================================
echo ""
echo -e "${BOLD}==========================================${NC}"
echo -e "${BOLD}  POST-TRAINING EVAL COMPLETATA${NC}"
echo -e "${BOLD}==========================================${NC}"
echo ""
echo -e "  Checkpoint: $CHECKPOINT"
echo -e "  Attenzione: $ATT_TYPE"
echo ""