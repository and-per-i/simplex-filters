"""
Data loading utilities per finetuning ibrido LLaMA + 2-Simplicial.

Fornisce:
- C4 streaming per training
- Wikitext-2 per validation
- Chunking a sequenze di lunghezza fissa
"""

import math
import torch
from typing import Optional, Iterator
from datasets import load_dataset


class ConstantLengthDataset:
    """
    Dataset iterabile che produce blocchi di lunghezza costante.
    Legge testo da un dataset HuggingFace in streaming, tokenizza,
    e restituisce sequenze di lunghezza fissa (seq_length).

    Args:
        tokenizer: tokenizer HuggingFace
        dataset: dataset HuggingFace in streaming
        seq_length: lunghezza della sequenza
    """

    def __init__(self, tokenizer, dataset, seq_length=512):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.seq_length = seq_length
        self.buffer = []

    def __iter__(self):
        self.buffer = []
        for example in self.dataset:
            text = example.get("text", "")
            if not text.strip():
                continue
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            self.buffer.extend(tokens)

            while len(self.buffer) >= self.seq_length + 1:
                chunk = self.buffer[:self.seq_length + 1]
                self.buffer = self.buffer[self.seq_length:]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long).unsqueeze(0)
                labels = torch.tensor(chunk[1:], dtype=torch.long).unsqueeze(0)
                attention_mask = torch.ones_like(input_ids)

                yield {
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": attention_mask,
                }


def make_c4_train_loader(
    tokenizer,
    seq_length=512,
    max_samples: Optional[int] = None,
):
    """
    Crea un iteratore per training su C4 inglese in streaming.

    Args:
        tokenizer: tokenizer LLaMA
        seq_length: lunghezza sequenza
        max_samples: numero massimo di campioni (None = illimitato)

    Returns:
        iterator su dict con input_ids, labels, attention_mask
    """
    dataset = load_dataset("c4", "en", split="train", streaming=True)

    if max_samples is not None:
        dataset = dataset.take(max_samples)

    return ConstantLengthDataset(tokenizer, dataset, seq_length=seq_length)


def make_wikitext_val_loader(
    tokenizer,
    seq_length=512,
    stride=256,
    max_samples: Optional[int] = None,
):
    """
    Crea un iteratore per validation su Wikitext-2.
    Restituisce intere sequenze (senza sliding window — lo gestisce la loss).

    Args:
        tokenizer: tokenizer LLaMA
        seq_length: lunghezza sequenza
        stride: stride per sliding window (non usato, mantenuto per interfaccia)
        max_samples: numero massimo di campioni

    Returns:
        lista di dict con input_ids, labels, attention_mask
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", streaming=True)

    if max_samples is not None:
        dataset = dataset.take(max_samples)

    return ConstantLengthDataset(tokenizer, dataset, seq_length=seq_length)


def prepare_validation_batch(
    tokenizer,
    seq_length=512,
    num_samples=500,
    device="cuda",
):
    """
    Prepara un batch fisso di validazione da Wikitext-2.
    Usato per valutazione periodica durante il training.

    Args:
        tokenizer: tokenizer
        seq_length: lunghezza sequenza
        num_samples: numero di campioni
        device: device

    Returns:
        dict con input_ids, labels, attention_mask come tensori [N, S]
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", streaming=True)
    dataset = dataset.take(num_samples)

    all_input_ids = []
    all_labels = []

    for example in dataset:
        text = example.get("text", "")
        if not text.strip():
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=seq_length + 1)
        if len(tokens) < seq_length + 1:
            continue
        all_input_ids.append(tokens[:seq_length])
        all_labels.append(tokens[1:seq_length + 1])

        if len(all_input_ids) >= num_samples:
            break

    if not all_input_ids:
        # Fallback: crea batch fittizio
        all_input_ids = [[0] * seq_length]
        all_labels = [[0] * seq_length]

    return {
        "input_ids": torch.tensor(all_input_ids[:num_samples], dtype=torch.long, device=device),
        "labels": torch.tensor(all_labels[:num_samples], dtype=torch.long, device=device),
        "attention_mask": torch.ones(num_samples, seq_length, dtype=torch.long, device=device),
    }