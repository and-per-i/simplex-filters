#!/usr/bin/env python3
"""
upload_to_hub.py — Carica checkpoint su HuggingFace Hub.

Usa upload_folder per gestire correttamente i file LFS (.safetensors).

Usage:
    HF_TOKEN=hf_xxx python scripts/upload_to_hub.py ./trilinear/checkpoint-10000 and-per-i/simplex-trilinear-1b
    HF_TOKEN=hf_xxx python scripts/upload_to_hub.py --private ./trilinear/final and-per-i/simplex-trilinear-1b-final
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def upload_checkpoint(
    checkpoint_path: str,
    repo_name: str,
    private: bool = False,
    commit_message: str = "Upload checkpoint",
):
    """
    Carica l'intera directory del checkpoint su HuggingFace Hub.
    Usa upload_folder che gestisce correttamente LFS per i .safetensors.
    """
    from huggingface_hub import HfApi, login, create_repo, upload_folder

    abs_path = os.path.abspath(checkpoint_path)
    if not os.path.exists(abs_path):
        print(f"❌ Path non trovato: {abs_path}")
        return False

    # Login
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token, add_to_git_credential=True)
    else:
        print("⚠️  HF_TOKEN non impostata. Provo login da cache...")

    # Crea repo e upload in un colpo solo
    repo_url = create_repo(
        repo_id=repo_name,
        private=private,
        exist_ok=True,
    )
    print(f"  ✓ Repo pronto: {repo_url}")

    # Upload l'intera cartella
    print(f"\n  Uploading {abs_path} → {repo_name}...")
    result = upload_folder(
        repo_id=repo_name,
        folder_path=abs_path,
        commit_message=commit_message,
        ignore_patterns=[".gitkeep", ".DS_Store"],
    )
    print(f"  ✓ Upload completato: {result}")
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