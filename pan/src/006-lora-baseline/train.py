"""
Training script for LoRA fine-tuning of Qwen2.5-1.5B on AI text detection.
"""

import os
import json
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from config import (
    MODEL_NAME,
    TRAIN_FILE,
    VAL_FILE,
    ARTIFACTS_DIR,
    LEARNING_RATE,
    NUM_EPOCHS,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    GRADIENT_ACCUMULATION,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from dataset import prepare_datasets
from model import create_lora_model


def compute_metrics(eval_pred):
    """Metrics callback for the Trainer."""
    logits, labels = eval_pred
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    predictions = np.argmax(logits, axis=-1)

    try:
        roc_auc = roc_auc_score(labels, probs[:, 1])
    except Exception:
        roc_auc = 0.5

    f1 = f1_score(labels, predictions)
    accuracy = (predictions == labels).mean()

    return {"roc_auc": roc_auc, "f1": f1, "accuracy": accuracy}


def train(max_samples: int = None):
    """
    Run LoRA fine-tuning of Qwen2.5-1.5B.

    Args:
        max_samples: If set, limit training/val to this many rows (for testing).
    """
    # ── Paths ──────────────────────────────────────────────────
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = ARTIFACTS_DIR / "checkpoints"

    # ── Data ───────────────────────────────────────────────────
    train_dataset, val_dataset, tokenizer = prepare_datasets(
        str(TRAIN_FILE), str(VAL_FILE)
    )

    if max_samples:
        train_dataset = train_dataset.select(range(min(max_samples, len(train_dataset))))
        val_dataset = val_dataset.select(range(min(max_samples, len(val_dataset))))
        print(f"Using {len(train_dataset)} train / {len(val_dataset)} val samples")

    # ── Model ──────────────────────────────────────────────────
    device = "cpu"
    peft_model = create_lora_model(device=device)

    # ── Training args ──────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        use_cpu=True,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        logging_dir=str(ARTIFACTS_DIR / "logs"),
        logging_steps=100,
        report_to="none",
        dataloader_num_workers=0,
    )

    # ── Trainer ────────────────────────────────────────────────
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ── Train ──────────────────────────────────────────────────
    print("\nStarting LoRA fine-tuning of Qwen2.5-1.5B ...")
    trainer.train()

    # ── Save adapter ───────────────────────────────────────────
    adapter_dir = ARTIFACTS_DIR / "lora_adapter"
    peft_model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"\nLoRA adapter saved to {adapter_dir}")

    # ── Final eval ─────────────────────────────────────────────
    print("\nRunning final evaluation on validation set ...")
    eval_results = trainer.evaluate()
    print("Validation results:")
    for k, v in eval_results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    with open(ARTIFACTS_DIR / "training_val_metrics.json", "w") as f:
        json.dump(eval_results, f, indent=4)

    return eval_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LoRA fine-tuning for 006")
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit data for quick tests",
    )
    args = parser.parse_args()
    train(max_samples=args.max_samples)
