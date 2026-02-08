"""
Evaluation script for zero-shot Ollama inference results.
Computes all PAN2025 metrics on predictions.
"""

import json
import argparse
import sys
from pathlib import Path
import numpy as np

# Add parent directory to import metrics
sys.path.insert(0, str(Path(__file__).parent.parent / "003-deberta"))
from metrics import get_all_metrics


def load_predictions(pred_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load predictions from JSON file.
    
    Returns:
        Tuple of (y_true, y_pred, y_prob)
    """
    with open(pred_path, 'r') as f:
        results = json.load(f)
    
    y_true = np.array([r["label"] for r in results])
    y_pred = np.array([r["prediction"] for r in results])
    y_prob = np.array([r["probability"] for r in results])
    
    return y_true, y_pred, y_prob


def evaluate(pred_path: str, output_path: str = None):
    """
    Compute all metrics on predictions.
    
    Args:
        pred_path: Path to predictions JSON
        output_path: Optional path to save metrics JSON
    """
    y_true, y_pred, y_prob = load_predictions(pred_path)
    
    # Compute metrics
    metrics = get_all_metrics(y_true, y_pred, y_prob)
    
    # Add summary stats
    n_total = len(y_true)
    n_correct = np.sum(y_true == y_pred)
    n_ai_pred = np.sum(y_pred == 1)
    n_human_pred = np.sum(y_pred == 0)
    
    results = {
        "metrics": metrics,
        "summary": {
            "total_samples": n_total,
            "correct": int(n_correct),
            "accuracy": float(n_correct / n_total),
            "ai_predictions": int(n_ai_pred),
            "human_predictions": int(n_human_pred),
        }
    }
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"\nSamples: {n_total}")
    print(f"Accuracy: {n_correct/n_total:.4f}")
    print(f"AI predictions: {n_ai_pred} ({n_ai_pred/n_total*100:.1f}%)")
    print(f"Human predictions: {n_human_pred} ({n_human_pred/n_total*100:.1f}%)")
    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    print("="*50 + "\n")
    
    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate zero-shot predictions")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSON")
    parser.add_argument("--output", default=None, help="Path to save metrics JSON")
    
    args = parser.parse_args()
    evaluate(args.predictions, args.output)


if __name__ == "__main__":
    main()
