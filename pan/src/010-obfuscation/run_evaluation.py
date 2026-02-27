"""
Evaluation script for experiment 010-obfuscation.
Evaluates on PAN2025 val and all available OOD datasets.
Logs all metrics to W&B for dashboarding.

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
import wandb

# Load API keys from root .env (needed if loading from HF)
load_dotenv(Path(__file__).resolve().parent.parent.parent.parent / ".env")

from config import (
    VAL_FILE, HC3_FILE, RAID_FILE,
    MAGE_FILE, OPENGPTTEXT_FILE,
    EXTERNAL_DIR, ARTIFACTS_DIR,
    WANDB_PROJECT, WANDB_ENTITY, WANDB_RUN_NAME, HF_REPO,
)
from data import load_jsonl, get_tokenizer
from model import load_trained_model
from evaluate import evaluate_dataset, print_comparison


def main():
    parser = argparse.ArgumentParser(description="010: LoRA evaluation on all datasets")
    parser.add_argument("--adapter-path", default=None,
                        help="Path or HF repo for adapter (default: HF_REPO)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples per dataset")
    args = parser.parse_args()

    adapter_path = args.adapter_path or HF_REPO
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── W&B init for evaluation logging ────────────────────────
    wandb.login(key=os.environ["WANDB_API_KEY"])
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=f"{WANDB_RUN_NAME}-eval",
        job_type="evaluation",
        config={
            "adapter_path": adapter_path,
            "max_samples": args.max_samples,
            "device": device,
        },
    )

    # ── Load model ─────────────────────────────────────────────
    print(f"Loading adapter from: {adapter_path}")
    model = load_trained_model(adapter_path, device=device)
    tokenizer = get_tokenizer()

    # ── Dataset registry ───────────────────────────────────────
    # All datasets to evaluate on — PAN2025 val + all OOD
    datasets = [
        ("PAN2025 Validation", str(VAL_FILE), "pan2025_val"),
        ("HC3 Wiki",           str(HC3_FILE),  "hc3"),
        ("RAID",               str(RAID_FILE), "raid"),
        ("MAGE",               str(MAGE_FILE), "mage"),
        ("OpenGPTText",        str(OPENGPTTEXT_FILE), "opengpttext"),
    ]

    all_results = {}

    for name, path, key in datasets:
        if not os.path.exists(path):
            print(f"\nSkipping {name}: not found at {path}")
            continue

        print(f"\n{'='*50}")
        print(f"Evaluating: {name}")
        print(f"{'='*50}")

        df = load_jsonl(path)
        if args.max_samples:
            df = df.head(args.max_samples)

        print(f"  Samples: {len(df)} (label distribution: {df['label'].value_counts().to_dict()})")

        metrics, probs = evaluate_dataset(model, tokenizer, df,
                                          device=device, desc=name)

        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        # Log per-dataset metrics to W&B
        wandb.log({
            f"eval/{key}/ROC-AUC": metrics.get("ROC-AUC", 0),
            f"eval/{key}/Brier": metrics.get("Brier", 0),
            f"eval/{key}/F1": metrics.get("F1", 0),
            f"eval/{key}/C@1": metrics.get("C@1", 0),
            f"eval/{key}/F0.5u": metrics.get("F0.5u", 0),
            f"eval/{key}/n_samples": len(df),
        })

        # Also log to W&B summary for the dashboard overview
        for metric_name, metric_val in metrics.items():
            wandb.run.summary[f"{key}/{metric_name}"] = metric_val

        with open(ARTIFACTS_DIR / f"{key}_results.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(ARTIFACTS_DIR / f"{key}_predictions.json", "w") as f:
            json.dump({"probabilities": probs.tolist()}, f)

        all_results[name] = metrics

    # ── Compute aggregate OOD metrics ─────────────────────────
    ood_keys = [k for k in all_results if k != "PAN2025 Validation"]
    if ood_keys:
        avg_ood_auc = sum(all_results[k].get("ROC-AUC", 0) for k in ood_keys) / len(ood_keys)
        wandb.run.summary["eval/avg_ood_roc_auc"] = avg_ood_auc
        wandb.run.summary["eval/n_ood_datasets"] = len(ood_keys)
        print(f"\nAvg OOD ROC-AUC: {avg_ood_auc:.4f} (across {len(ood_keys)} datasets)")

    # ── Save combined results + print ─────────────────────────
    with open(ARTIFACTS_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Log the full results JSON as a W&B artifact
    artifact = wandb.Artifact(
        name="evaluation-results",
        type="evaluation",
        description="All evaluation metrics for experiment 010",
    )
    artifact.add_file(str(ARTIFACTS_DIR / "all_results.json"))
    wandb.log_artifact(artifact)

    print_comparison(all_results)

    # Full metrics table
    print(f"\n{'Dataset':<22} {'ROC-AUC':>10} {'Brier':>10} {'F1':>10} {'C@1':>10} {'F0.5u':>10}")
    print("-" * 72)
    for name, metrics in all_results.items():
        print(f"{name:<22} {metrics.get('ROC-AUC',0):>10.4f} {metrics.get('Brier',0):>10.4f} "
              f"{metrics.get('F1',0):>10.4f} {metrics.get('C@1',0):>10.4f} {metrics.get('F0.5u',0):>10.4f}")

    print(f"\nResults saved to {ARTIFACTS_DIR}")

    wandb.finish()


if __name__ == "__main__":
    main()
