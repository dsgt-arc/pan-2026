"""
Evaluation script for experiment 009.
Same functionality as evaluation.ipynb, just as a runnable Python file.

Usage:
    python run_evaluation.py
    python run_evaluation.py --adapter-path /path/to/adapter
    python run_evaluation.py --max-samples 100   # quick test
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
from dotenv import load_dotenv

# Load API keys from root .env (needed if loading from HF)
load_dotenv(Path(__file__).resolve().parent.parent.parent.parent / ".env")

from config import VAL_FILE, HC3_FILE, RAID_FILE, ARTIFACTS_DIR
from data import load_jsonl, get_tokenizer
from model import load_trained_model
from evaluate import evaluate_dataset, print_comparison


def main():
    parser = argparse.ArgumentParser(description="009: LoRA evaluation")
    parser.add_argument("--adapter-path", default=None,
                        help="Path to saved adapter (default: artifacts/lora_adapter)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples per dataset")
    args = parser.parse_args()

    adapter_path = args.adapter_path or str(ARTIFACTS_DIR / "lora_adapter")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load model ─────────────────────────────────────────────
    print(f"Loading adapter from: {adapter_path}")
    model = load_trained_model(adapter_path, device=device)
    tokenizer = get_tokenizer()

    # ── Evaluate ───────────────────────────────────────────────
    datasets = [
        ("PAN2025 Validation", str(VAL_FILE), "pan2025_val"),
        ("HC3 Wiki",           str(HC3_FILE),  "hc3"),
        ("RAID",               str(RAID_FILE), "raid"),
    ]

    all_results = {}

    for name, path, key in datasets:
        if not os.path.exists(path):
            print(f"Skipping {name}: not found at {path}")
            continue

        print(f"\n{'='*50}")
        print(f"Evaluating: {name}")
        print(f"{'='*50}")

        df = load_jsonl(path)
        if args.max_samples:
            df = df.head(args.max_samples)

        metrics, probs = evaluate_dataset(model, tokenizer, df,
                                          device=device, desc=name)

        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        with open(ARTIFACTS_DIR / f"{key}_results.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(ARTIFACTS_DIR / f"{key}_predictions.json", "w") as f:
            json.dump({"probabilities": probs.tolist()}, f)

        all_results[name] = metrics

    # ── Save + print ───────────────────────────────────────────
    with open(ARTIFACTS_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print_comparison(all_results)

    # Full metrics table
    print(f"\n{'Dataset':<22} {'ROC-AUC':>10} {'Brier':>10} {'F1':>10} {'C@1':>10} {'F0.5u':>10}")
    print("-" * 72)
    for name, metrics in all_results.items():
        print(f"{name:<22} {metrics.get('ROC-AUC',0):>10.4f} {metrics.get('Brier',0):>10.4f} "
              f"{metrics.get('F1',0):>10.4f} {metrics.get('C@1',0):>10.4f} {metrics.get('F0.5u',0):>10.4f}")

    print(f"\nResults saved to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
