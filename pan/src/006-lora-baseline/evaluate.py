"""
Evaluation script for the LoRA fine-tuned Qwen2.5-1.5B model.
Handles long documents via chunking + aggregation (same strategy as 003).
Computes all 5 standard PAN metrics.
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from peft import PeftModel
from tqdm import tqdm

# Reuse metrics from 003
from config import (
    MODEL_NAME,
    NUM_LABELS,
    ARTIFACTS_DIR,
    VAL_FILE,
    HC3_FILE,
    SRC_003,
    CHUNK_SIZE,
    CHUNK_STRIDE,
)

# Add 003 source to path so we can import metrics & chunking
sys.path.insert(0, str(SRC_003))
from metrics import get_all_metrics  # noqa: E402
from chunking import chunk_text_by_tokens, aggregate_predictions  # noqa: E402
from data_loader import load_jsonl  # noqa: E402


def load_lora_model(adapter_path: str, device: str = "cpu"):
    """
    Loads the base Qwen2.5-1.5B model + LoRA adapter for inference.
    """
    print(f"Loading base model: {MODEL_NAME}")
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.pad_token_id = config.eos_token_id
    config.num_labels = NUM_LABELS
    config.problem_type = "single_label_classification"

    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        config=config,
        torch_dtype="auto",
    )

    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()  # Merge for faster inference
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    # Ensure padding configuration matches training
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


def predict_with_chunking(
    model,
    tokenizer,
    text: str,
    max_length: int = CHUNK_SIZE,
    stride: int = CHUNK_STRIDE,
    device: str = "cpu",
) -> float:
    """
    Predict P(AI-generated) for a single text, with chunking for long docs.
    """
    encoding = tokenizer(
        text,
        truncation=False,
        return_tensors="pt",
        add_special_tokens=True,
    )

    input_ids = encoding["input_ids"][0].tolist()
    attention_mask = encoding["attention_mask"][0].tolist()

    chunked_ids, chunked_masks = chunk_text_by_tokens(
        input_ids,
        attention_mask,
        chunk_size=max_length,
        stride=stride,
        pad_token_id=tokenizer.pad_token_id,
    )

    chunk_probs = []
    with torch.no_grad():
        for chunk_ids, chunk_mask in zip(chunked_ids, chunked_masks):
            inputs = {
                "input_ids": torch.tensor([chunk_ids]).to(device),
                "attention_mask": torch.tensor([chunk_mask]).to(device),
            }
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            chunk_probs.append(probs[0, 1].item())

    return aggregate_predictions(chunk_probs, method="mean")


def evaluate_dataset(
    model, tokenizer, data_path: str, device: str = "cpu", max_samples: int = None,
):
    """
    Evaluate the model on a JSONL dataset.

    Returns:
        dict of metrics
    """
    df = load_jsonl(data_path)
    if max_samples:
        df = df.head(max_samples)

    y_true = df["label"].tolist()
    y_probs = []

    print(f"Evaluating {len(df)} samples from {Path(data_path).name} ...")
    for text in tqdm(df["text"], desc="Predicting"):
        prob = predict_with_chunking(model, tokenizer, text, device=device)
        y_probs.append(prob)

    y_probs = np.array(y_probs)
    y_pred = (y_probs >= 0.5).astype(int)

    metrics = get_all_metrics(y_true, y_pred, y_probs)
    return metrics, y_probs


def evaluate_all(adapter_path: str = None, max_samples: int = None):
    """
    Evaluate on PAN2025 validation + HC3 (OOD) and save results.
    """
    if adapter_path is None:
        adapter_path = str(ARTIFACTS_DIR / "lora_adapter")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    device = "cpu"
    model, tokenizer = load_lora_model(adapter_path, device=device)

    all_results = {}

    datasets = [
        ("PAN2025 Validation", str(VAL_FILE), "pan2025_val_results.json"),
        ("HC3 Wiki (OOD)", str(HC3_FILE), "hc3_results.json"),
    ]

    for name, path, out_file in datasets:
        if not os.path.exists(path):
            print(f"Warning: {name}: file not found at {path}, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")

        metrics, _ = evaluate_dataset(
            model, tokenizer, path, device=device, max_samples=max_samples,
        )

        print(f"\n{name} Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        with open(ARTIFACTS_DIR / out_file, "w") as f:
            json.dump(metrics, f, indent=4)

        all_results[name] = metrics

    # Combined
    with open(ARTIFACTS_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=4)

    # Comparison table
    _print_comparison(all_results)

    return all_results


def _print_comparison(results: dict):
    """Print a comparison table against baselines."""
    baselines = {
        "LR (001)":        {"PAN2025": 0.9953, "HC3": 0.5520},
        "LightGBM (002)":  {"PAN2025": 0.9980, "HC3": 0.3356},
        "DeBERTa FT (003)":{"PAN2025": 0.9948, "HC3": 0.5939},
        "Ollama (005)":    {"PAN2025": 0.4606, "HC3": 0.4162},
    }

    print("\n" + "=" * 60)
    print("ROC-AUC Comparison")
    print("=" * 60)
    print(f"{'Model':<22} {'PAN2025 Val':>14} {'HC3 (OOD)':>14}")
    print("-" * 50)

    for name, scores in baselines.items():
        print(f"{name:<22} {scores['PAN2025']:>14.4f} {scores['HC3']:>14.4f}")

    pan = results.get("PAN2025 Validation", {}).get("ROC-AUC", 0)
    hc3 = results.get("HC3 Wiki (OOD)", {}).get("ROC-AUC", 0)
    print(f"{'Qwen LoRA (006)':<22} {pan:>14.4f} {hc3:>14.4f}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate LoRA model")
    parser.add_argument("--adapter-path", default=None, help="Path to saved LoRA adapter")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    evaluate_all(adapter_path=args.adapter_path, max_samples=args.max_samples)
