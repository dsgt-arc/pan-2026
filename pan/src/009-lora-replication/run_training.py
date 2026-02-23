"""
Training script for experiment 009.
Same functionality as training.ipynb, just as a runnable Python file.

Usage:
    python run_training.py
    python run_training.py --max-samples 100   # quick smoke test
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
from dotenv import load_dotenv
import wandb
from huggingface_hub import login as hf_login, HfApi

# Load API keys from root .env
load_dotenv(Path(__file__).resolve().parent.parent.parent.parent / ".env")

from config import (
    TRAIN_FILE, VAL_FILE, ARTIFACTS_DIR,
    WANDB_PROJECT, WANDB_ENTITY, WANDB_RUN_NAME, HF_REPO,
)
from data import load_jsonl, get_tokenizer, tokenize_dataset
from model import create_lora_model
from train import get_training_args, run_training


def main():
    parser = argparse.ArgumentParser(description="009: LoRA training")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples for smoke tests")
    parser.add_argument("--skip-upload", action="store_true",
                        help="Skip uploading to HuggingFace")
    args = parser.parse_args()

    # ── Auth ───────────────────────────────────────────────────
    wandb.login(key=os.environ["WANDB_API_KEY"])
    hf_login(token=os.environ["HF_TOKEN"])
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    os.environ["WANDB_ENTITY"] = WANDB_ENTITY

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Data ───────────────────────────────────────────────────
    tokenizer = get_tokenizer()

    train_df = load_jsonl(str(TRAIN_FILE))
    val_df = load_jsonl(str(VAL_FILE))

    if args.max_samples:
        train_df = train_df.head(args.max_samples)
        val_df = val_df.head(args.max_samples)

    print(f"Train: {len(train_df)} samples")
    print(f"Val:   {len(val_df)} samples")

    train_dataset = tokenize_dataset(train_df, tokenizer)
    val_dataset = tokenize_dataset(val_df, tokenizer)

    # ── Model ──────────────────────────────────────────────────
    model = create_lora_model(device=device)

    # ── Train ──────────────────────────────────────────────────
    output_dir = str(ARTIFACTS_DIR / "checkpoints")
    training_args = get_training_args(output_dir, use_cpu=(device == "cpu"))

    trainer, eval_results = run_training(
        model, tokenizer, train_dataset, val_dataset, training_args
    )

    # ── Save ───────────────────────────────────────────────────
    adapter_dir = ARTIFACTS_DIR / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"Adapter saved to {adapter_dir}")

    with open(ARTIFACTS_DIR / "training_val_metrics.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    # ── Upload ─────────────────────────────────────────────────
    if not args.skip_upload:
        api = HfApi()
        api.create_repo(HF_REPO, exist_ok=True)
        api.upload_folder(
            folder_path=str(adapter_dir),
            repo_id=HF_REPO,
            repo_type="model",
        )
        print(f"Adapter uploaded to https://huggingface.co/{HF_REPO}")

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
