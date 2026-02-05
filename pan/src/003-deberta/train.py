"""
CPU-optimized training script for DeBERTa-v3 fine-tuning.
"""

import os
import json
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

from dataset import prepare_datasets, MODEL_NAME, MAX_LENGTH

def compute_metrics(eval_pred):
    """Compute metrics for evaluation during training."""
    logits, labels = eval_pred
    # Softmax to get probabilities
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    predictions = np.argmax(logits, axis=-1)
    
    # ROC-AUC (use probability of positive class)
    try:
        roc_auc = roc_auc_score(labels, probs[:, 1])
    except:
        roc_auc = 0.5
    
    # F1
    f1 = f1_score(labels, predictions)
    
    # Accuracy
    accuracy = (predictions == labels).mean()
    
    return {
        'roc_auc': roc_auc,
        'f1': f1,
        'accuracy': accuracy
    }


def main():
    # 1. Paths
    base_data_path = "/Users/tanish/Desktop/CLEF2026/pan-2026/harshul/src/pan-data"
    train_file = os.path.join(base_data_path, "train.jsonl")
    val_file = os.path.join(base_data_path, "val.jsonl")
    
    experiment_dir = "/Users/tanish/Desktop/CLEF2026/pan-2026/harshul/experiments/003-deberta"
    artifacts_dir = os.path.join(experiment_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    output_dir = os.path.join(artifacts_dir, "checkpoints")
    
    # 2. Load Data
    train_dataset, val_dataset, tokenizer = prepare_datasets(train_file, val_file)
    
    # 3. Load Model
    print(f"\nLoading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        problem_type="single_label_classification"
    )
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # 4. Training Arguments (CPU-optimized)
    training_args = TrainingArguments(
        output_dir=output_dir,
        
        # CPU settings
        no_cuda=True,
        
        # Batch size (small for CPU memory)
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,  # Effective batch size = 16
        
        # Learning rate
        learning_rate=2e-5,
        weight_decay=0.01,
        
        # Epochs
        num_train_epochs=3,
        warmup_steps=500,
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        
        # Logging
        logging_dir=os.path.join(artifacts_dir, "logs"),
        logging_steps=100,
        report_to="none",  # Disable wandb/tensorboard
        
        # Memory optimization
        dataloader_num_workers=0,  # Avoid multiprocessing issues on some systems
    )
    
    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # 6. Train
    print("\nStarting training (CPU mode)...")
    print("This may take several hours. Consider running overnight.")
    trainer.train()
    
    # 7. Save final model
    final_model_path = os.path.join(artifacts_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    
    # 8. Final Evaluation
    print("\nRunning final evaluation...")
    eval_results = trainer.evaluate()
    print("Validation Results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Save metrics
    with open(os.path.join(artifacts_dir, "val_metrics.json"), 'w') as f:
        json.dump(eval_results, f, indent=4)


if __name__ == "__main__":
    main()
