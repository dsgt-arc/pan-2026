"""
Training utilities for LoRA fine-tuning with W&B logging.
- compute_metrics: all 6 PAN metrics + accuracy
- GPUStatsCallback: logs GPU memory at each logging step
- get_training_args: builds TrainingArguments with W&B enabled
- run_training: sets up Trainer and runs training
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score, fbeta_score
from transformers import TrainingArguments, Trainer, TrainerCallback, EarlyStoppingCallback

from config import (
    ARTIFACTS_DIR,
    LEARNING_RATE,
    NUM_EPOCHS,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    GRADIENT_ACCUMULATION,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    EVAL_STEPS,
    SAVE_STEPS,
    LOGGING_STEPS,
    WANDB_PROJECT,
    WANDB_RUN_NAME,
)


def compute_metrics(eval_pred):
    """
    Metrics callback for the Trainer.
    Computes all 6 PAN metrics: ROC-AUC, Brier, F1, C@1, F0.5u, + accuracy.
    """
    logits, labels = eval_pred

    # Softmax to get probabilities
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    pos_probs = probs[:, 1]
    preds = np.argmax(logits, axis=-1)

    # ROC-AUC
    try:
        roc_auc = roc_auc_score(labels, pos_probs)
    except Exception:
        roc_auc = 0.5

    # Brier score
    brier = brier_score_loss(labels, pos_probs)

    # F1
    f1 = f1_score(labels, preds)

    # C@1: (1/n)(nc + nu * nc/n) where nu = unanswered (prob == 0.5)
    n = len(labels)
    unanswered = pos_probs == 0.5
    nc = int(((preds == labels) & ~unanswered).sum())
    nu = int(unanswered.sum())
    c_at_1 = (nc + nu * nc / n) / n if n > 0 else 0

    # F0.5u: F0.5 with unanswered mapped to negative class
    modified_preds = preds.copy()
    modified_preds[unanswered] = 0
    f05u = fbeta_score(labels, modified_preds, beta=0.5)

    # Accuracy
    accuracy = (preds == labels).mean()

    return {
        "roc_auc": roc_auc,
        "brier": brier,
        "f1": f1,
        "c_at_1": c_at_1,
        "f05u": f05u,
        "accuracy": accuracy,
    }


class GPUStatsCallback(TrainerCallback):
    """Logs GPU memory usage at each logging step."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and torch.cuda.is_available():
            logs["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            logs["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1e9


def get_training_args(output_dir: str, use_cpu: bool = True):
    """Build TrainingArguments with W&B logging enabled."""
    return TrainingArguments(
        output_dir=output_dir,
        use_cpu=use_cpu,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_EPOCHS,
        warmup_steps=WARMUP_STEPS,
        # Evaluation & saving
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        # Logging — W&B
        logging_dir=str(ARTIFACTS_DIR / "logs"),
        logging_steps=LOGGING_STEPS,
        report_to="wandb",
        run_name=WANDB_RUN_NAME,
        dataloader_num_workers=0,
        # Mixed precision
        bf16=not use_cpu,
        disable_tqdm=True,
    )


def run_training(model, tokenizer, train_dataset, val_dataset, training_args):
    """
    Set up and run the HuggingFace Trainer with W&B logging,
    GPU monitoring, and early stopping on ROC-AUC.
    Returns (trainer, eval_results).
    """
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE),
            GPUStatsCallback(),
        ],
    )

    print("Starting LoRA fine-tuning ...")
    trainer.train()

    # Final evaluation
    print("Running final validation ...")
    eval_results = trainer.evaluate()
    for k, v in eval_results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    return trainer, eval_results
