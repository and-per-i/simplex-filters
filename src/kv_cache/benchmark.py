"""
benchmark.py — Benchmark delle performance di eviction Q-filter.

Per ogni budget B in [100%, 50%, 30%, 10%]:
1. Carica checkpoint addestrato (trilineare o Gram Det)
2. Calcola analisi geometrica UNA SOLA VOLTA (U_mean, σ₁, σ₂)
3. Per ogni token i in Wikitext-2:
   a. Estrai k_j nella finestra w1=512
   b. Calcola Q-filter score ortogonale ‖k − P̄k‖ per ogni k_j
   c. Tieni solo top-B chiavi (score alto = chiave atipica = preservata)
4. Misura perplexity
5. Ripeti con random eviction

NOTA: La formula usata e' qfilter_score_orthogonal (‖k − P̄k‖),
validata con Spearman ρ=+0.61 (GramDet) e ρ=+0.50 (trilineare).
La formula anisotropa di proiezione (qfilter_score) non e' piu' usata
perche' anti-correlata (ρ=-0.27).

Output: 4 curve PPL vs B su WandB + stdout.
"""

import math
import torch
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from src.kv_cache.qfilter_score import qfilter_score_orthogonal, top_k_indices, random_indices


@dataclass
class BenchmarkResult:
    """Risultati del benchmark per un checkpoint."""
    model_name: str
    budget: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.3, 0.1])
    ppl_qfilter: Dict[float, float] = field(default_factory=dict)
    ppl_random: Dict[float, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"\nModel: {self.model_name}"]
        lines.append(f"{'Budget':>8} {'Q-filter':>10} {'Random':>10} {'Delta':>10}")
        lines.append("-" * 42)
        for b in self.budget:
            q = self.ppl_qfilter.get(b, float('nan'))
            r = self.ppl_random.get(b, float('nan'))
            d = r - q
            lines.append(f"{b*100:>6.0f}% {q:>10.2f} {r:>10.2f} {d:>+10.2f}")
        return "\n".join(lines)


def _set_model_eviction(model, simplicial_indices, U_mean, budget, strategy):
    """Imposta eviction_params sui layer simpliciali del modello."""
    for idx in simplicial_indices:
        attn = model.model.layers[idx].self_attn
        if hasattr(attn, "eviction_params"):
            attn.eviction_params = {
                "U_mean": U_mean,
                "budget": budget,
                "strategy": strategy,
            }


def _clear_model_eviction(model, simplicial_indices):
    """Rimuove eviction_params dai layer simpliciali."""
    for idx in simplicial_indices:
        attn = model.model.layers[idx].self_attn
        if hasattr(attn, "eviction_params"):
            attn.eviction_params = None


def benchmark_checkpoint(
    checkpoint_path: str,
    attention_type: str = "simplicial",
    model_name: str = "meta-llama/Llama-3.1-8B",
    budgets: List[float] = None,
    seq_length: int = 256,
    num_batches: int = 5,
    window_size: int = 512,
    device: str = "cuda",
    wandb_active: bool = False,
    llama_base_model=None,
    llama_base_ppl: Optional[float] = None,
    tokenizer=None,
) -> BenchmarkResult:
    """
    Benchmark di eviction su un checkpoint addestrato.

    Usa qfilter_score_orthogonal (‖k − P̄k‖) come proxy valido per
    preservare le chiavi atipiche (lontane dal piano medio).

    Args:
        checkpoint_path: path al checkpoint
        attention_type: "simplicial" o "gram_det"
        budgets: frazioni da testare
        seq_length, num_batches: parametri analisi
        window_size: finestra K1
        device: device
        wandb_active: logga su WandB
        llama_base_model: modello LLaMA base (opzionale, evita ricaricamento)
        llama_base_ppl: PPL LLaMA base pre-calcolata (opzionale)
        tokenizer: tokenizer (opzionale, evita ricaricamento)

    Returns:
        BenchmarkResult con PPL per ogni budget
    """
    if budgets is None:
        budgets = [1.0, 0.5, 0.3, 0.1]
    
    # Simplicial indices (default per backward compat)
    simplicial_indices = [16, 20, 24, 28]

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.geometry.analyzer import analyze_checkpoint

    # 1. Carica modello ibrido
    print(f"\nCaricamento checkpoint: {checkpoint_path}")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
    )
    model.eval()

    # Tokenizer (riusa se passato, altrimenti carica — usa model_name passato o default)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Calcola analisi geometrica UNA SOLA VOLTA
    print("Analisi geometrica...")
    geo_results = analyze_checkpoint(
        checkpoint_path=checkpoint_path,
        attention_type=attention_type,
        num_analysis_batches=num_batches,
        seq_length=seq_length,
        device=device,
        verbose=True,
    )

    # Prendi il primo layer (16) per ora — tutti i layer simpliciali
    # hanno parametri geometrici simili
    layer_idx = 16
    if layer_idx in geo_results:
        layer_data = geo_results[layer_idx]
        U_mean = layer_data["U_mean"].to(device)
    else:
        print(f"Layer {layer_idx} non trovato, uso valori di default")
        U_mean = torch.eye(128, device=device)[:, :2]

    result = BenchmarkResult(model_name=os.path.basename(checkpoint_path))

    # 3. Prepara batch di validazione
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test", streaming=True)

    # 4a. Logga baseline LLaMA base (una volta, all'inizio)
    if llama_base_ppl is not None and wandb_active:
        import wandb
        wandb.log({"eval/llama_base_perplexity": llama_base_ppl})

    # 4b. Loop sui budget B
    for budget in budgets:
        print(f"\n  Budget: {budget*100:.0f}%")

        # Q-filter eviction (ortogonale — ‖k − P̄k‖)
        _set_model_eviction(model, simplicial_indices, U_mean, budget, "qfilter")
        ppl_qf = _eval_ppl_with_eviction(
            model, tokenizer, dataset, U_mean,
            budget=budget, strategy="qfilter",
            seq_length=seq_length, num_batches=num_batches, device=device,
        )
        _clear_model_eviction(model, simplicial_indices)
        result.ppl_qfilter[budget] = ppl_qf
        print(f"    Q-filter (ortogonale) PPL: {ppl_qf:.2f}")

        # Random eviction baseline
        _set_model_eviction(model, simplicial_indices, U_mean, budget, "random")
        ppl_rand = _eval_ppl_with_eviction(
            model, tokenizer, dataset, U_mean,
            budget=budget, strategy="random",
            seq_length=seq_length, num_batches=num_batches, device=device,
        )
        _clear_model_eviction(model, simplicial_indices)
        result.ppl_random[budget] = ppl_rand
        print(f"    Random PPL:   {ppl_rand:.2f}")

        if wandb_active:
            import wandb
            wandb.log({
                f"ppl_qfilter_{budget:.2f}": ppl_qf,
                f"ppl_random_{budget:.2f}": ppl_rand,
            })

    # 4c. Logga hybrid_no_eviction (B=100%) e perplexity_gap
    if 1.0 in result.ppl_qfilter and wandb_active:
        import wandb
        ppl_hybrid = result.ppl_qfilter[1.0]
        wandb.log({"eval/hybrid_no_eviction_perplexity": ppl_hybrid})

        if llama_base_ppl is not None:
            gap = ppl_hybrid - llama_base_ppl
            wandb.log({"eval/perplexity_gap": gap})
            passed = gap < 0.5
            wandb.log({"eval/gate_passed": 1.0 if passed else 0.0})
            print(f"\n  🔍 LLaMA base PPL: {llama_base_ppl:.2f}")
            print(f"  🔍 Ibrido (B=100%) PPL: {ppl_hybrid:.2f}")
            print(f"  🔍 Gap: {gap:+.2f}  — {'✅ PASS' if passed else '❌ FAIL'} (soglia: 0.5)")

    print(result.summary())
    return result


def _eval_ppl_with_eviction(
    model, tokenizer, dataset,
    U_mean,
    budget: float,
    strategy: str,
    seq_length: int = 256,
    num_batches: int = 5,
    device: str = "cuda",
) -> float:
    """Calcola PPL con una strategia di eviction."""
    import torch.nn.functional as F

    total_loss = 0.0
    total_tokens = 0
    batch_count = 0

    for example in dataset:
        if batch_count >= num_batches:
            break

        text = example.get("text", "")
        if not text.strip():
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True,
                                   max_length=seq_length)
        if len(tokens) < seq_length + 1:
            continue

        input_ids = torch.tensor(tokens[:seq_length], dtype=torch.long,
                                 device=device).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.item()

        total_loss += loss * (seq_length - 1)
        total_tokens += (seq_length - 1)
        batch_count += 1

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss)
    return ppl