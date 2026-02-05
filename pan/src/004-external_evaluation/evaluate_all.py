"""
Main evaluation script for all models on all external datasets.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Suppress warnings and fix tokenizer fork issue
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add paths
SRC_DIR = Path("/Users/tanish/Desktop/CLEF2026/pan-2026/harshul/src")
sys.path.insert(0, str(SRC_DIR / "003-deberta"))
sys.path.insert(0, str(SRC_DIR / "004-external_evaluation"))

from model_loaders import load_all_models
from metrics import get_all_metrics

# Paths
DATA_DIR = SRC_DIR / "pan-data/external"
ARTIFACTS_DIR = Path("/Users/tanish/Desktop/CLEF2026/pan-2026/harshul/experiments/004-external_evaluation/artifacts")

# Config
MAX_SAMPLES = None  # Full evaluation
SKIP_DEBERTA = False

# Datasets to evaluate
DATASETS = [
    "hc3_wiki_processed",
    "raid",
    "turingbench", 
    "m4",
    "ghostbuster"
]


def load_dataset(name: str, max_samples: int = None) -> tuple:
    """Load a dataset and return (texts, labels)."""
    path = DATA_DIR / f"{name}.jsonl"
    
    texts = []
    labels = []
    
    with open(path) as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            item = json.loads(line)
            texts.append(item["text"])
            labels.append(item["label"])
    
    return texts, np.array(labels)


def evaluate_model_on_dataset(model, texts: list, labels: np.ndarray) -> dict:
    """Evaluate a model on a dataset."""
    all_probs = []
    
    # DeBERTa processes one at a time - show per-sample progress
    if model.name == "DeBERTa":
        pbar = tqdm(texts, desc=f"  {model.name}", leave=True, dynamic_ncols=True)
        for text in pbar:
            probs = model.predict_proba([text])
            all_probs.extend(probs)
            sys.stdout.flush()  # Force progress bar update
    else:
        # LR and LightGBM can batch - show batch progress
        batch_size = 32
        for i in tqdm(range(0, len(texts), batch_size), desc=f"  {model.name}", leave=True):
            batch = texts[i:i+batch_size]
            probs = model.predict_proba(batch)
            all_probs.extend(probs)
    
    y_probs = np.array(all_probs)
    y_pred = (y_probs >= 0.5).astype(int)
    
    # Compute metrics
    metrics = get_all_metrics(labels, y_pred, y_probs)
    return metrics


def run_full_evaluation():
    """Run evaluation on all models and datasets."""
    print("=" * 60)
    print("External Evaluation Benchmark")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create artifacts directory
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load models (optionally skip DeBERTa for speed)
    models = load_all_models(skip_deberta=SKIP_DEBERTA)
    
    if not models:
        print("No models loaded. Exiting.")
        return
    
    # Results storage
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "models": list(models.keys()),
            "datasets": DATASETS,
            "metrics": ["ROC-AUC", "Brier", "F1", "C@1", "F0.5u"]
        },
        "results": {}
    }
    
    # Evaluate each model on each dataset
    for dataset_name in DATASETS:
        print(f"\n📊 Dataset: {dataset_name}")
        
        try:
            texts, labels = load_dataset(dataset_name, max_samples=MAX_SAMPLES)
            print(f"   Samples: {len(texts)}, Label distribution: {np.mean(labels):.2%} AI")
        except FileNotFoundError:
            print(f"   ⚠️  Dataset not found, skipping")
            continue
        
        results["results"][dataset_name] = {}
        
        for model_name, model in models.items():
            try:
                metrics = evaluate_model_on_dataset(model, texts, labels)
                results["results"][dataset_name][model_name] = metrics
                
                # Display ALL metrics
                print(f"   {model_name}:")
                for metric_name, value in metrics.items():
                    print(f"      {metric_name}: {value:.4f}")
                    
            except Exception as e:
                print(f"   {model_name}: ❌ Error - {e}")
                results["results"][dataset_name][model_name] = {"error": str(e)}
    
    # Save results
    output_path = ARTIFACTS_DIR / "all_metrics.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {output_path}")
    
    # Generate summary table
    generate_summary_table(results)
    
    return results


def generate_summary_table(results: dict):
    """Generate markdown summary table."""
    print("\n" + "=" * 60)
    print("Summary: ROC-AUC by Model and Dataset")
    print("=" * 60)
    
    # Create ROC-AUC table manually (avoid tabulate dependency)
    models = list(next(iter(results["results"].values())).keys())
    
    # Header
    header = "| Dataset | " + " | ".join(models) + " |"
    separator = "|" + "---|" * (len(models) + 1)
    print(header)
    print(separator)
    
    # Rows
    rows = []
    for dataset, model_results in results["results"].items():
        row_vals = [dataset]
        for model in models:
            if model in model_results and "error" not in model_results[model]:
                row_vals.append(f"{model_results[model]['ROC-AUC']:.4f}")
            else:
                row_vals.append("ERR")
        row = "| " + " | ".join(row_vals) + " |"
        print(row)
        rows.append(row)
    
    # Save as markdown with ALL metrics
    table_path = ARTIFACTS_DIR / "comparison_table.md"
    with open(table_path, 'w') as f:
        f.write("# External Evaluation Results\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## ROC-AUC Summary\n\n")
        f.write(header + "\n")
        f.write(separator + "\n")
        for row in rows:
            f.write(row + "\n")
        
        f.write("\n\n## Full Metrics by Dataset\n\n")
        
        for dataset, model_results in results["results"].items():
            f.write(f"### {dataset}\n\n")
            f.write("| Model | ROC-AUC | Brier | F1 | C@1 | F0.5u |\n")
            f.write("|-------|---------|-------|-----|-----|-------|\n")
            
            for model, metrics in model_results.items():
                if "error" not in metrics:
                    f.write(f"| {model} | {metrics['ROC-AUC']:.4f} | {metrics['Brier']:.4f} | {metrics['F1']:.4f} | {metrics['C@1']:.4f} | {metrics['F0.5u']:.4f} |\n")
                else:
                    f.write(f"| {model} | ERR | ERR | ERR | ERR | ERR |\n")
            f.write("\n")
    
    print(f"\n📝 Full results saved to {table_path}")


if __name__ == "__main__":
    run_full_evaluation()
