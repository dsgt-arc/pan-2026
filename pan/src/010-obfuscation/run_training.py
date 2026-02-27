"""
Training script for experiment 010-obfuscation.
Applies homoglyph + synonym augmentation to training data before LoRA fine-tuning.

Usage:
    python run_training.py
    python run_training.py --max-samples 100   # quick smoke test
    python run_training.py --skip-upload        # skip HuggingFace upload
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
    MODEL_NAME, MAX_LENGTH, NUM_LABELS,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    LEARNING_RATE, NUM_EPOCHS, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE,
    GRADIENT_ACCUMULATION, WARMUP_STEPS, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, EVAL_STEPS, SAVE_STEPS, LOGGING_STEPS,
    HOMOGLYPH_FRAC, SYNONYM_FRAC,
    HOMOGLYPH_PROB, ZWJ_PROB, SYNONYM_PROB,
)
from data import load_jsonl, get_tokenizer, tokenize_dataset
from model import create_lora_model
from train import get_training_args, run_training
from augment import augment_dataframe


def main():
    parser = argparse.ArgumentParser(description="010: LoRA training with obfuscation augmentation")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples for smoke tests")
    parser.add_argument("--skip-upload", action="store_true",
                        help="Skip uploading to HuggingFace")
    parser.add_argument("--homoglyph-frac", type=float, default=HOMOGLYPH_FRAC,
                        help=f"Fraction of AI texts for homoglyph augmentation (default: {HOMOGLYPH_FRAC})")
    parser.add_argument("--synonym-frac", type=float, default=SYNONYM_FRAC,
                        help=f"Fraction of AI texts for synonym replacement (default: {SYNONYM_FRAC})")
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

    # ── Augmentation ──────────────────────────────────────────
    print("\n=== Applying obfuscation augmentation ===")
    n_ai_before = int((train_df["label"] == 1).sum())
    train_df = augment_dataframe(
        train_df,
        homoglyph_frac=args.homoglyph_frac,
        synonym_frac=args.synonym_frac,
        h_prob=HOMOGLYPH_PROB,
        zwj_prob=ZWJ_PROB,
        syn_prob=SYNONYM_PROB,
    )

    # Save augmentation config for reproducibility
    aug_config = {
        "homoglyph_frac": args.homoglyph_frac,
        "synonym_frac": args.synonym_frac,
        "homoglyph_prob": HOMOGLYPH_PROB,
        "zwj_prob": ZWJ_PROB,
        "synonym_prob": SYNONYM_PROB,
        "train_samples": len(train_df),
        "ai_samples": n_ai_before,
        "n_homoglyph_augmented": int(n_ai_before * args.homoglyph_frac),
        "n_synonym_augmented": int(n_ai_before * args.synonym_frac),
    }
    with open(ARTIFACTS_DIR / "augmentation_config.json", "w") as f:
        json.dump(aug_config, f, indent=2)

    # ── Tokenize ──────────────────────────────────────────────
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

    # ── Log full config to W&B ─────────────────────────────────
    wandb.config.update({
        # Model
        "model_name": MODEL_NAME,
        "num_labels": NUM_LABELS,
        "max_length": MAX_LENGTH,
        # LoRA
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "lora_target_modules": LORA_TARGET_MODULES,
        # Training
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "effective_batch_size": TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION,
        "warmup_steps": WARMUP_STEPS,
        "weight_decay": WEIGHT_DECAY,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        # Augmentation
        **aug_config,
        # Data
        "train_samples_total": len(train_df),
        "val_samples": len(val_df),
        "device": device,
    }, allow_val_change=True)

    # Log final validation metrics as W&B summary
    for k, v in eval_results.items():
        if isinstance(v, (int, float)):
            wandb.run.summary[f"final_val/{k}"] = v

    # ── Save metrics locally ────────────────────────────────────
    with open(ARTIFACTS_DIR / "training_val_metrics.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    # ── Upload adapter to HuggingFace ─────────────────────────
    if not args.skip_upload:
        # Save adapter to temp dir for upload
        adapter_dir = ARTIFACTS_DIR / "lora_adapter"
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

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
