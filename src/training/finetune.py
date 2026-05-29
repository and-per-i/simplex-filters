"""
Finetuning del modello ibrido LLaMA + 2-Simplicial su C4/The Pile.

Strategia:
- Modello frozen tranne K1/V1/K2/V2 dei 4 layer simpliciali
- 3 gruppi di learning rate
- Ottimizzatore AdamW custom passato al Trainer
"""

import os
import sys
import math
from typing import Optional, List

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
)
from datasets import load_dataset

# Aggiungi src al path per import relativi
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modeling.convert_to_hybrid import convert_llama_to_hybrid, freeze_parameters


class ConstantLengthDataset(IterableDataset):
    """
    Dataset iterabile che produce blocchi di lunghezza costante.
    Usato per training su dataset di testo libero (C4, The Pile).
    """
    
    def __init__(self, tokenizer, dataset, seq_length=512, num_samples=None):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.seq_length = seq_length
        self.num_samples = num_samples
        self.buffer = []
    
    def __iter__(self):
        for example in self.dataset:
            text = example["text"]
            tokens = self.tokenizer.encode(text)
            self.buffer.extend(tokens)
            
            while len(self.buffer) >= self.seq_length + 1:
                chunk = self.buffer[:self.seq_length + 1]
                self.buffer = self.buffer[self.seq_length:]
                
                input_ids = torch.tensor(chunk[:-1]).unsqueeze(0)
                labels = torch.tensor(chunk[1:]).unsqueeze(0)
                
                yield {
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": torch.ones_like(input_ids),
                }


def create_optimizer(
    model,
    simplicial_indices: List[int],
    lr_k2v2: float = 2e-4,
    lr_k1v1: float = 2e-5,
    weight_decay: float = 0.01,
):
    """
    Crea l'ottimizzatore AdamW con 3 gruppi di parametri.
    
    Args:
        model: modello ibrido
        simplicial_indices: indici dei layer simpliciali
        lr_k2v2: learning rate per K2/V2 (nuovi parametri)
        lr_k1v1: learning rate per K1/V1 (parametri pre-trained, raffinati)
        weight_decay: weight decay
    
    Returns:
        optimizer: AdamW configurato
    """
    param_groups = freeze_parameters(
        model, simplicial_indices,
        lr_k1v1=lr_k1v1,
        lr_k2v2=lr_k2v2,
    )
    
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    
    return optimizer


def run_finetune(
    model_name: str = "meta-llama/Llama-3.1-8B",
    output_dir: str = "./finetune-output",
    simplicial_indices: List[int] = [16, 20, 24, 28],
    alpha: float = 0.01,
    w1: int = 32,
    w2: int = 256,
    lr_k2v2: float = 2e-4,
    lr_k1v1: float = 2e-5,
    max_steps: int = 3000,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    seq_length: int = 512,
    warmup_steps: int = 100,
    logging_steps: int = 10,
    save_steps: int = 500,
    bf16: bool = True,
    dataset_name: str = "c4",
    dataset_config: str = "en",
    dataset_split: str = "train",
    max_train_samples: Optional[int] = None,
):
    """
    Esegue il finetuning del modello ibrido.
    
    Args:
        model_name: nome del modello su HuggingFace
        output_dir: directory di output per checkpoint
        simplicial_indices: quali layer convertire in SimplicialAttention
        alpha: coefficiente di perturbazione per K2/V2
        w1, w2: finestre dell'attenzione 2-simpliciale
        lr_k2v2: learning rate per K2/V2
        lr_k1v1: learning rate per K1/V1
        max_steps: numero di step di training
        per_device_batch_size: batch size per device
        gradient_accumulation_steps: gradient accumulation
        seq_length: lunghezza della sequenza
        warmup_steps: warmup del learning rate
        logging_steps: frequenza logging
        save_steps: frequenza salvataggio
        bf16: usa bfloat16
        dataset_name: nome del dataset (c4, the_pile, etc.)
        dataset_config: configurazione del dataset
        dataset_split: split del dataset
        max_train_samples: numero massimo di campioni di training
    """
    set_seed(42)
    
    # 1. Carica tokenizer e modello
    print(f"\n[finetune] Caricamento modello: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
        device_map="auto",
        attn_implementation="eager",  # evita flash_attn per compatibilità
    )
    model.train()
    
    # 2. Converti in ibrido
    print(f"\n[finetune] Conversione in modello ibrido")
    model, converted = convert_llama_to_hybrid(
        model,
        simplicial_indices=simplicial_indices,
        alpha=alpha,
        w1=w1,
        w2=w2,
    )
    
    # 3. Crea ottimizzatore
    optimizer = create_optimizer(
        model,
        simplicial_indices=simplicial_indices,
        lr_k2v2=lr_k2v2,
        lr_k1v1=lr_k1v1,
    )
    
    # 4. Carica dataset
    print(f"\n[finetune] Caricamento dataset: {dataset_name}")
    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=dataset_split,
        streaming=True,
    )
    
    if max_train_samples:
        dataset = dataset.take(max_train_samples)
    
    train_dataset = ConstantLengthDataset(
        tokenizer=tokenizer,
        dataset=dataset,
        seq_length=seq_length,
    )
    
    # 5. Configura training
    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        bf16=bf16,
        prediction_loss_only=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        save_total_limit=2,
        logging_dir=f"{output_dir}/logs",
        report_to="none",  # no wandb/tensorboard
        ddp_find_unused_parameters=False,
    )
    
    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        optimizers=(optimizer, None),  # scheduler=None (solo warmup lineare)
        tokenizer=tokenizer,
    )
    
    # 7. Training
    print(f"\n[finetune] Avvio training ({max_steps} steps)")
    print(f"  Dataset: {dataset_name}")
    print(f"  Batch size (eff.): {per_device_batch_size * gradient_accumulation_steps}")
    print(f"  Seq length: {seq_length}")
    print(f"  LR K2/V2: {lr_k2v2}, LR K1/V1: {lr_k1v1}")
    print(f"  Parametri trainable: ~134M (1.7%)")
    
    trainer.train()
    
    # 8. Salva modello finale
    print(f"\n[finetune] Salvataggio modello in {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"[finetune] Training completato!")
    
    return model, trainer


# Script eseguibile
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Finetuning ibrido LLaMA + 2-Simplicial")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--output-dir", type=str, default="./finetune-output")
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--w1", type=int, default=32)
    parser.add_argument("--w2", type=int, default=256)
    parser.add_argument("--lr-k2v2", type=float, default=2e-4)
    parser.add_argument("--lr-k1v1", type=float, default=2e-5)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--dataset", type=str, default="c4")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-bf16", action="store_true")
    
    args = parser.parse_args()
    
    run_finetune(
        model_name=args.model,
        output_dir=args.output_dir,
        alpha=args.alpha,
        w1=args.w1,
        w2=args.w2,
        lr_k2v2=args.lr_k2v2,
        lr_k1v1=args.lr_k1v1,
        max_steps=args.max_steps,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        seq_length=args.seq_length,
        dataset_name=args.dataset,
        bf16=not args.no_bf16,
    )