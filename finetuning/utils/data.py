"""
Data loading utilities per finetuning ibrido LLaMA + 2-Simplicial.

Fornisce:
- C4 streaming per training
- Wikitext-2 per validation
- Chunking a sequenze di lunghezza fissa

I dataset possono essere caricati da HuggingFace (streaming) o da disco locale
(per ambienti con proxy restrittivi come Vast.ai). La directory locale è ./data/.
"""

import math
import torch
import os
from typing import Optional, Iterator
from datasets import load_dataset, load_from_disk

# Directory locale per dataset pre-scaricati (su Vast.ai: esegui prima lo script di download)
LOCAL_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

WIKITEST_PATH = os.path.join(LOCAL_DATA_DIR, "wikitext_test")
C4_PATH = os.path.join(LOCAL_DATA_DIR, "c4_validation")
C4_TRAIN_PATH = os.path.join(LOCAL_DATA_DIR, "c4_train")


def _load_dataset_or_fallback(
    hf_path: str,
    hf_config: str,
    hf_split: str,
    local_path: str,
    streaming: bool = True,
):
    """
    Prova a caricare un dataset da HuggingFace in streaming.
    Se fallisce (proxy Vast), carica da disco locale se disponibile.
    
    Args:
        hf_path: nome del dataset HF (es. "Salesforce/wikitext")
        hf_config: config del dataset (es. "wikitext-2-raw-v1")
        hf_split: split (es. "test", "validation")
        local_path: path locale dove il dataset è stato salvato
        streaming: se usare streaming per HF
        
    Returns:
        dataset
    """
    try:
        return load_dataset(hf_path, hf_config, split=hf_split, streaming=streaming)
    except Exception as e:
        print(f"  [WARN] Caricamento da HF fallito ({e})")
        if os.path.exists(local_path):
            print(f"  [INFO] Carico da disco: {local_path}")
            ds = load_from_disk(local_path)
            if not streaming:
                # load_from_disk restituisce un Dataset non iterabile se salvato con save_to_disk
                # Converto in iterabile se necessario
                pass
            return ds
        raise


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
    Se il dataset locale C4_TRAIN_PATH esiste (scaricato con download_c4.py), lo usa direttamente.
    Fallback: carica da disco C4_validation, poi Wikitext-2 se HF non raggiungibile.
    """
    # Priorità 1: dataset C4 locale completo
    if os.path.exists(C4_TRAIN_PATH):
        print(f"  [INFO] C4 training locale: {C4_TRAIN_PATH}")
        import datasets
        full_ds = datasets.load_from_disk(C4_TRAIN_PATH)
        if max_samples is not None:
            full_ds = full_ds.select(range(min(max_samples, len(full_ds))))
        return ConstantLengthDataset(tokenizer, full_ds, seq_length=seq_length)

    # Priorità 2: HF streaming o C4_validation fallback
    try:
        dataset = _load_dataset_or_fallback(
            "allenai/c4", "en", "train",
            C4_PATH, streaming=True,
        )
    except Exception:
        print(f"  [WARN] C4 non disponibile, uso Wikitext-2 train come fallback")
        dataset = _load_dataset_or_fallback(
            "Salesforce/wikitext", "wikitext-2-raw-v1", "train",
            WIKITEST_PATH, streaming=True,
        )

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
    """
    dataset = _load_dataset_or_fallback(
        "Salesforce/wikitext", "wikitext-2-raw-v1", "test",
        WIKITEST_PATH, streaming=True,
    )

    if max_samples is not None:
        dataset = dataset.take(max_samples)

    return ConstantLengthDataset(tokenizer, dataset, seq_length=seq_length)


def _concat_dataset_into_token_stream(tokenizer, dataset, max_tokens=2_000_000):
    """
    Concatena tutti i record di un dataset in un unico stream di token.
    Stesso approccio di scripts/baseline_ppl.py.
    """
    all_tokens = []
    for example in dataset:
        text = example.get("text", "")
        if not text.strip():
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) < 2:
            continue
        all_tokens.extend(tokens)
        if len(all_tokens) >= max_tokens:
            break
    return all_tokens


def prepare_c4_validation_batch(
    tokenizer,
    seq_length=512,
    num_samples=500,
    device="cuda",
):
    """
    Prepara un batch fisso di validazione concatenando tutti i record
    in un unico stream di token e chunkando in num_samples sequenze.
    Stesso approccio di scripts/baseline_ppl.py.
    """
    # Carica dataset (disco locale o HF)
    if os.path.exists(WIKITEST_PATH):
        print(f"  [INFO] Validation da disco: {WIKITEST_PATH}")
        import datasets
        dataset = datasets.load_from_disk(WIKITEST_PATH)
    else:
        print(f"  [INFO] Validation da HF: Wikitext-2 test")
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test", streaming=True)

    # Concatena tutti i record in un unico stream di token
    all_tokens = _concat_dataset_into_token_stream(tokenizer, dataset)
    total_tokens = len(all_tokens)
    print(f"  Token totali nello stream: {total_tokens:,}")

    # Calcola quanti chunk possiamo ottenere
    needed_tokens = num_samples * (seq_length + 1)
    if total_tokens < needed_tokens:
        print(f"  [WARN] Solo {total_tokens} token, necessari {needed_tokens} per {num_samples} campioni")
        actual_samples = max(1, total_tokens // (seq_length + 1))
    else:
        actual_samples = num_samples

    all_input_ids = []
    all_labels = []

    for i in range(actual_samples):
        start = i * (seq_length + 1)
        chunk = all_tokens[start : start + seq_length + 1]
        if len(chunk) < seq_length + 1:
            chunk = chunk + [0] * (seq_length + 1 - len(chunk))
        all_input_ids.append(chunk[:seq_length])
        all_labels.append(chunk[1:seq_length + 1])

    # Padding per arrivare a num_samples
    while len(all_input_ids) < num_samples:
        all_input_ids.append([0] * seq_length)
        all_labels.append([0] * seq_length)

    print(f"  Batch creato: {len(all_input_ids)} campioni")
    return {
        "input_ids": torch.tensor(all_input_ids[:num_samples], dtype=torch.long, device=device),
        "labels": torch.tensor(all_labels[:num_samples], dtype=torch.long, device=device),
        "attention_mask": torch.ones(num_samples, seq_length, dtype=torch.long, device=device),
    }


def prepare_validation_batch(
    tokenizer,
    seq_length=512,
    num_samples=500,
    device="cuda",
):
    """
    Prepara un batch fisso di validazione da Wikitext-2.
    Usato per benchmark finale DOPO il training.
    """
    dataset = _load_dataset_or_fallback(
        "Salesforce/wikitext", "wikitext-2-raw-v1", "test",
        WIKITEST_PATH, streaming=True,
    )
    dataset = dataset.shuffle(seed=42, buffer_size=10000).take(num_samples * 2)

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
        all_input_ids = [[0] * seq_length]
        all_labels = [[0] * seq_length]

    return {
        "input_ids": torch.tensor(all_input_ids[:num_samples], dtype=torch.long, device=device),
        "labels": torch.tensor(all_labels[:num_samples], dtype=torch.long, device=device),
        "attention_mask": torch.ones(num_samples, seq_length, dtype=torch.long, device=device),
    }