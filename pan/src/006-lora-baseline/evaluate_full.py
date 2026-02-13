"""
Full evaluation of the Qwen2.5-1.5B LoRA model on PAN2025 val + all OOD datasets.
Pulls the adapter from HuggingFace and runs comprehensive evaluation.
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from peft import PeftModel
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PAN_ROOT = PROJECT_ROOT / "pan"
DATA_DIR = PAN_ROOT / "src" / "pan-data"
EXTERNAL_DIR = DATA_DIR / "external"
ARTIFACTS_DIR = PAN_ROOT / "experiments" / "006-lora-baseline" / "artifacts"
SRC_003 = PAN_ROOT / "src" / "003-deberta"

# Add 003 source to path for metrics & chunking
sys.path.insert(0, str(SRC_003))
from metrics import get_all_metrics
from chunking import chunk_text_by_tokens, aggregate_predictions
from data_loader import load_jsonl

# ── Model config ───────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
ADAPTER_REPO = "hersheys-baklava/qwen-lora-pan2026"
NUM_LABELS = 2
CHUNK_SIZE = 512
CHUNK_STRIDE = 256

# ── Datasets ───────────────────────────────────────────────────
DATASETS = [
    ("PAN2025 Validation", str(DATA_DIR / "val.jsonl"), "pan2025_val"),
    ("HC3 Wiki", str(EXTERNAL_DIR / "hc3_wiki_processed.jsonl"), "hc3_wiki"),
    ("RAID", str(EXTERNAL_DIR / "raid.jsonl"), "raid"),
    ("TuringBench", str(EXTERNAL_DIR / "turingbench.jsonl"), "turingbench"),
    ("M4", str(EXTERNAL_DIR / "m4.jsonl"), "m4"),
    ("Ghostbuster", str(EXTERNAL_DIR / "ghostbuster.jsonl"), "ghostbuster"),
]


def load_lora_model(adapter_path: str, device: str = "cpu"):
    """Loads the base Qwen2.5-1.5B model + LoRA adapter for inference."""
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
    model = model.merge_and_unload()
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


def predict_with_chunking(model, tokenizer, text, device="cpu"):
    """Predict P(AI-generated) for a single text, with chunking for long docs."""
    encoding = tokenizer(
        text,
        truncation=False,
        return_tensors="pt",
        add_special_tokens=True,
    )

    input_ids = encoding["input_ids"][0].tolist()
    attention_mask = encoding["attention_mask"][0].tolist()

    chunked_ids, chunked_masks = chunk_text_by_tokens(
        input_ids, attention_mask,
        chunk_size=CHUNK_SIZE, stride=CHUNK_STRIDE,
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


def evaluate_dataset(model, tokenizer, data_path, device="cpu", max_samples=None):
    """Evaluate the model on a JSONL dataset. Returns metrics dict and raw probs."""
    df = load_jsonl(data_path)
    if max_samples:
        df = df.head(max_samples)

    y_true = df["label"].tolist()
    y_probs = []

    dataset_name = Path(data_path).stem
    print(f"  Evaluating {len(df)} samples ...")
    for text in tqdm(df["text"], desc=f"  {dataset_name}", leave=True):
        prob = predict_with_chunking(model, tokenizer, text, device=device)
        y_probs.append(prob)

    y_probs = np.array(y_probs)
    y_pred = (y_probs >= 0.5).astype(int)

    metrics = get_all_metrics(y_true, y_pred, y_probs)
    return metrics, y_probs


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Full evaluation of Qwen LoRA model")
    parser.add_argument("--adapter-path", default=ADAPTER_REPO,
                        help="HuggingFace repo or local path to adapter")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples per dataset (for testing)")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda/mps)")
    args = parser.parse_args()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("006-lora-baseline: Full Evaluation")
    print("=" * 60)
    print(f"Adapter: {args.adapter_path}")
    print(f"Device: {args.device}")
    print(f"Started: {datetime.now().isoformat()}")
    print()

    # ── Load model ─────────────────────────────────────────────
    model, tokenizer = load_lora_model(args.adapter_path, device=args.device)

    # ── Evaluate all datasets ──────────────────────────────────
    all_results = {}

    for name, path, key in DATASETS:
        if not os.path.exists(path):
            print(f"\nWarning: {name} not found at {path}, skipping.")
            continue

        print(f"\n{'─'*60}")
        print(f"Dataset: {name}")
        print(f"{'─'*60}")

        metrics, probs = evaluate_dataset(
            model, tokenizer, path, device=args.device,
            max_samples=args.max_samples,
        )

        print(f"\n  Results:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")

        # Save per-dataset results
        with open(ARTIFACTS_DIR / f"{key}_results.json", "w") as f:
            json.dump(metrics, f, indent=4)

        # Save raw predictions
        with open(ARTIFACTS_DIR / f"{key}_predictions.json", "w") as f:
            json.dump({"probabilities": probs.tolist()}, f)

        all_results[name] = metrics

    # ── Save combined results ──────────────────────────────────
    with open(ARTIFACTS_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=4)

    # ── Print comparison table ─────────────────────────────────
    print_comparison_table(all_results)

    print(f"\nCompleted: {datetime.now().isoformat()}")
    print(f"Artifacts saved to: {ARTIFACTS_DIR}")


def print_comparison_table(results):
    """Print a comprehensive comparison table against all baselines."""

    # Previous experiment results from 004 results.md
    baselines = {
        "LR (001)":         {"PAN2025": 0.9953, "HC3": 0.5517, "RAID": 0.5727,
                             "TuringBench": 0.5449, "M4": 0.5409, "Ghostbuster": 0.5227},
        "LightGBM (002)":   {"PAN2025": 0.9980, "HC3": 0.3356, "RAID": 0.3398,
                             "TuringBench": 0.3209, "M4": 0.3170, "Ghostbuster": 0.3012},
        "DeBERTa FT (003)": {"PAN2025": 0.9948, "HC3": 0.5939, "RAID": 0.5879,
                             "TuringBench": 0.5660, "M4": 0.5591, "Ghostbuster": 0.5889},
    }

    # Map result keys to short names
    key_map = {
        "PAN2025 Validation": "PAN2025",
        "HC3 Wiki": "HC3",
        "RAID": "RAID",
        "TuringBench": "TuringBench",
        "M4": "M4",
        "Ghostbuster": "Ghostbuster",
    }

    # Collect Qwen LoRA results
    qwen_results = {}
    for name, metrics in results.items():
        short = key_map.get(name, name)
        qwen_results[short] = metrics.get("ROC-AUC", 0)

    print("\n" + "=" * 80)
    print("ROC-AUC Comparison: All Models × All Datasets")
    print("=" * 80)

    datasets = ["PAN2025", "HC3", "RAID", "TuringBench", "M4", "Ghostbuster"]
    header = f"{'Model':<20}" + "".join(f"{d:>14}" for d in datasets) + f"{'Avg OOD':>14}"
    print(header)
    print("-" * len(header))

    for model_name, scores in baselines.items():
        ood_vals = [scores.get(d, 0) for d in datasets[1:]]
        avg_ood = np.mean(ood_vals) if ood_vals else 0
        row = f"{model_name:<20}"
        for d in datasets:
            row += f"{scores.get(d, 0):>14.4f}"
        row += f"{avg_ood:>14.4f}"
        print(row)

    # Qwen LoRA row
    ood_vals = [qwen_results.get(d, 0) for d in datasets[1:]]
    avg_ood = np.mean(ood_vals) if ood_vals else 0
    row = f"{'Qwen LoRA (006)':<20}"
    for d in datasets:
        row += f"{qwen_results.get(d, 0):>14.4f}"
    row += f"{avg_ood:>14.4f}"
    print(row)
    print("=" * len(header))

    # Full metrics table
    print("\n\nFull Metrics (Qwen LoRA 006):")
    print(f"{'Dataset':<20} {'ROC-AUC':>10} {'Brier':>10} {'F1':>10} {'C@1':>10} {'F0.5u':>10}")
    print("-" * 70)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics.get('ROC-AUC',0):>10.4f} {metrics.get('Brier',0):>10.4f} "
              f"{metrics.get('F1',0):>10.4f} {metrics.get('C@1',0):>10.4f} {metrics.get('F0.5u',0):>10.4f}")


if __name__ == "__main__":
    main()
