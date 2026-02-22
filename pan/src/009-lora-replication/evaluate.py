"""
Evaluation utilities for the LoRA model.
- predict_single: chunked inference for one text
- evaluate_dataset: evaluate a DataFrame, return metrics + probabilities
- print_comparison: comparison table vs. baselines

Reuses metrics from src/003-deberta/metrics.py and chunking from
src/003-deberta/chunking.py.
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from config import CHUNK_SIZE, CHUNK_STRIDE, ARTIFACTS_DIR, SRC_003

# Import shared utilities from 003-deberta
sys.path.insert(0, str(SRC_003))
from metrics import get_all_metrics  # noqa: E402
from chunking import chunk_text_by_tokens, aggregate_predictions  # noqa: E402


def predict_single(model, tokenizer, text, chunk_size=CHUNK_SIZE,
                    stride=CHUNK_STRIDE, device="cpu"):
    """
    Predict P(AI-generated) for a single text.
    Uses sliding-window chunking for long documents.
    """
    encoding = tokenizer(text, truncation=False, return_tensors="pt",
                         add_special_tokens=True)
    input_ids = encoding["input_ids"][0].tolist()
    attention_mask = encoding["attention_mask"][0].tolist()

    chunked_ids, chunked_masks = chunk_text_by_tokens(
        input_ids, attention_mask,
        chunk_size=chunk_size, stride=stride,
        pad_token_id=tokenizer.pad_token_id,
    )

    chunk_probs = []
    with torch.no_grad():
        for ids, mask in zip(chunked_ids, chunked_masks):
            inputs = {
                "input_ids": torch.tensor([ids]).to(device),
                "attention_mask": torch.tensor([mask]).to(device),
            }
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            chunk_probs.append(probs[0, 1].item())

    return aggregate_predictions(chunk_probs, method="mean")


def evaluate_dataset(model, tokenizer, df, device="cpu", desc="Evaluating"):
    """
    Evaluate the model on a DataFrame with 'text' and 'label' columns.
    Returns (metrics_dict, probabilities_array).
    """
    y_true = df["label"].tolist()
    y_probs = []

    for text in tqdm(df["text"], desc=desc):
        prob = predict_single(model, tokenizer, text, device=device)
        y_probs.append(prob)

    y_probs = np.array(y_probs)
    y_pred = (y_probs >= 0.5).astype(int)
    metrics = get_all_metrics(y_true, y_pred, y_probs)
    return metrics, y_probs


def print_comparison(results):
    """Print ROC-AUC comparison table against baselines."""
    baselines = {
        "LR (001)":         {"PAN2025": 0.9953, "HC3": 0.5517, "RAID": 0.5727},
        "LightGBM (002)":   {"PAN2025": 0.9980, "HC3": 0.3356, "RAID": 0.3398},
        "DeBERTa FT (003)": {"PAN2025": 0.9948, "HC3": 0.5939, "RAID": 0.5879},
        "Qwen LoRA (006)":  {"PAN2025": 0.9999, "HC3": 0.9982, "RAID": 0.8152},
    }

    datasets = ["PAN2025", "HC3", "RAID"]
    header = f"{'Model':<22}" + "".join(f"{d:>14}" for d in datasets)
    print("\n" + "=" * len(header))
    print("ROC-AUC Comparison")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for name, scores in baselines.items():
        row = f"{name:<22}" + "".join(f"{scores.get(d, 0):>14.4f}" for d in datasets)
        print(row)

    # Current experiment results
    key_map = {"PAN2025 Validation": "PAN2025", "HC3 Wiki": "HC3", "RAID": "RAID"}
    current = {key_map.get(k, k): v.get("ROC-AUC", 0) for k, v in results.items()}
    row = f"{'Qwen LoRA (009)':<22}" + "".join(f"{current.get(d, 0):>14.4f}" for d in datasets)
    print(row)
    print("=" * len(header))
