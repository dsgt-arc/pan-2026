"""
Main runner for experiment 006-lora-baseline.
Orchestrates training → evaluation → comparison.

Uses Qwen2.5-1.5B with LoRA adapters for AI text detection.
"""

import argparse
from datetime import datetime
from pathlib import Path

from config import ARTIFACTS_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Run experiment 006: LoRA fine-tuning baseline (Qwen2.5-1.5B)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit samples for quick smoke tests",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and only run evaluation (requires existing adapter)",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Path to pre-trained LoRA adapter (defaults to artifacts/lora_adapter)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("EXPERIMENT 006: LoRA Fine-Tuning Baseline (Qwen2.5-1.5B)")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Max samples: {args.max_samples or 'all'}")
    print()

    # ── Phase 1: Training ──────────────────────────────────────
    if not args.skip_train:
        print("\nPhase 1: Training")
        print("-" * 40)
        from train import train

        train(max_samples=args.max_samples)
    else:
        print("\nSkipping training (--skip-train)")

    # ── Phase 2: Evaluation ────────────────────────────────────
    print("\nPhase 2: Evaluation")
    print("-" * 40)
    from evaluate import evaluate_all

    adapter_path = args.adapter_path or str(ARTIFACTS_DIR / "lora_adapter")
    results = evaluate_all(adapter_path=adapter_path, max_samples=args.max_samples)

    # ── Done ───────────────────────────────────────────────────
    print(f"\nExperiment completed: {datetime.now().isoformat()}")
    print(f"   Artifacts: {ARTIFACTS_DIR}")

    return results


if __name__ == "__main__":
    main()
