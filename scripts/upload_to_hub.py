#!/usr/bin/env python3
"""
upload_to_hub.py — Carica checkpoint su HuggingFace Hub.

Usage:
    python scripts/upload_to_hub.py ./trilinear/checkpoint-10000 and-per-i/simplex-trilinear-1b
    python scripts/upload_to_hub.py ./trilinear/final and-per-i/simplex-trilinear-1b-final
    python scripts/upload_to_hub.py --private ./checkpoint and-per-i/repo-name
"""

import argparse
import os
import sys
import glob
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def upload_checkpoint(
    checkpoint_path: str,
    repo_name: str,
    private: bool = False,
    commit_message: str = "Upload checkpoint",
):
    """
    Carica un checkpoint su HuggingFace Hub usando upload_large_folder
    (preserva la struttura con model.safetensors + config.json).
    """
    from huggingface_hub import HfApi, login
    from safetensors.torch import load_file as safetensors_load

    abs_path = os.path.abspath(checkpoint_path)
    if not os.path.exists(abs_path):
        print(f"❌ Path non trovato: {abs_path}")
        return False

    # Login
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        print("⚠️  HF_TOKEN non impostata. Provo login da cache...")

    api = HfApi()

    # Crea repo se non esiste
    try:
        api.create_repo(
            repo_id=repo_name,
            private=private,
            exist_ok=True,
        )
        print(f"  ✓ Repo {repo_name} pronto")
    except Exception as e:
        print(f"  ⚠️  Creazione repo: {e}")

    # Carica tutti i file nella directory
    uploaded = 0
    for fpath in sorted(glob.glob(os.path.join(abs_path, "*"))):
        fname = os.path.basename(fpath)
        if os.path.isfile(fpath):
            print(f"  Uploading {fname}...", end=" ", flush=True)
            api.upload_file(
                path_or_fileobj=fpath,
                path_in_repo=fname,
                repo_id=repo_name,
                commit_message=commit_message,
            )
            print("OK")
            uploaded += 1

    print(f"\n  ✅ Upload completato: {uploaded} file → {repo_name}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Carica checkpoint su HuggingFace Hub"
    )
    parser.add_argument("checkpoint_path", type=str,
                        help="Path alla directory del checkpoint (es. ./trilinear/checkpoint-10000)")
    parser.add_argument("repo_name", type=str,
                        help="Nome repo su HF (es. and-per-i/simplex-trilinear-1b)")
    parser.add_argument("--private", action="store_true",
                        help="Repo privato (default: pubblico)")
    parser.add_argument("--message", type=str, default="Upload checkpoint",
                        help="Commit message")

    args = parser.parse_args()

    success = upload_checkpoint(
        args.checkpoint_path,
        args.repo_name,
        private=args.private,
        commit_message=args.message,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())