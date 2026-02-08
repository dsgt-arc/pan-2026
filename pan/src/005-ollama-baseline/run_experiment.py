"""
Main script to run the complete zero-shot Ollama experiment.
Runs inference on both PAN2025 validation and HC3 datasets.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

from infer import run_inference
from evaluate import evaluate


# Default paths
DATA_DIR = Path(__file__).parent.parent / "pan-data"
ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "experiments" / "005-ollama-baseline" / "artifacts"


def run_full_experiment(
    model: str = "qwen2:7b",
    prompt_type: str = "default",
    max_samples: int = None,
):
    """
    Run the complete experiment on both datasets.
    """
    print("="*60)
    print("EXPERIMENT 005: Zero-Shot LLM Detection via Ollama")
    print("="*60)
    print(f"\nModel: {model}")
    print(f"Prompt type: {prompt_type}")
    print(f"Max samples: {max_samples or 'all'}")
    print(f"Started: {datetime.now().isoformat()}")
    
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # Dataset configurations
    datasets = [
        {
            "name": "PAN2025 Validation",
            "path": DATA_DIR / "val.jsonl",
            "pred_file": "pan2025_val_predictions.json",
            "metrics_file": "pan2025_val_results.json",
        },
        {
            "name": "HC3 Wiki",
            "path": DATA_DIR / "external" / "hc3_wiki_processed.jsonl",
            "pred_file": "hc3_predictions.json",
            "metrics_file": "hc3_results.json",
        },
    ]
    
    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Processing: {ds['name']}")
        print("="*60)
        
        pred_path = ARTIFACTS_DIR / ds["pred_file"]
        metrics_path = ARTIFACTS_DIR / ds["metrics_file"]
        
        # Run inference
        run_inference(
            data_path=str(ds["path"]),
            output_path=str(pred_path),
            model=model,
            prompt_type=prompt_type,
            max_samples=max_samples,
        )
        
        # Evaluate
        results = evaluate(str(pred_path), str(metrics_path))
        all_results[ds["name"]] = results
    
    # Save combined results
    combined_path = ARTIFACTS_DIR / "all_results.json"
    with open(combined_path, 'w') as f:
        json.dump({
            "model": model,
            "prompt_type": prompt_type,
            "timestamp": datetime.now().isoformat(),
            "results": all_results,
        }, f, indent=2)
    
    # Print summary comparison
    print("\n" + "="*60)
    print("SUMMARY: Zero-Shot vs Trained Baselines")
    print("="*60)
    
    baselines = {
        "LR (001)": {"PAN2025 Validation": 0.9953, "HC3 Wiki": 0.5520},
        "LightGBM (002)": {"PAN2025 Validation": 0.9980, "HC3 Wiki": 0.3356},
        "DeBERTa (003)": {"PAN2025 Validation": 0.9948, "HC3 Wiki": 0.5939},
    }
    
    print("\nROC-AUC Comparison:")
    print(f"{'Model':<20} {'PAN2025 Val':>15} {'HC3 (OOD)':>15}")
    print("-"*50)
    
    for name, scores in baselines.items():
        print(f"{name:<20} {scores['PAN2025 Validation']:>15.4f} {scores['HC3 Wiki']:>15.4f}")
    
    # Add our results
    ollama_pan = all_results.get("PAN2025 Validation", {}).get("metrics", {}).get("ROC-AUC", 0)
    ollama_hc3 = all_results.get("HC3 Wiki", {}).get("metrics", {}).get("ROC-AUC", 0)
    print(f"{'Ollama (005)':<20} {ollama_pan:>15.4f} {ollama_hc3:>15.4f}")
    print("="*60)
    
    print(f"\nExperiment completed: {datetime.now().isoformat()}")
    print(f"Results saved to: {ARTIFACTS_DIR}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run full Ollama zero-shot experiment")
    parser.add_argument("--model", default="qwen2:7b", help="Ollama model name")
    parser.add_argument("--prompt-type", default="default", choices=["default", "confidence", "cot"])
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for testing")
    
    args = parser.parse_args()
    
    run_full_experiment(
        model=args.model,
        prompt_type=args.prompt_type,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
