"""
External evaluation script for DeBERTa model on HC3 dataset.
"""

import os
import json
from evaluate import evaluate_dataset
from load_model import get_model

def main():
    # Paths
    experiment_dir = "/Users/tanish/Desktop/CLEF2026/pan-2026/harshul/experiments/003-deberta"
    artifacts_dir = os.path.join(experiment_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    ext_file = "/Users/tanish/Desktop/CLEF2026/pan-2026/harshul/src/pan-data/external/hc3_wiki_processed.jsonl"
    
    if not os.path.exists(ext_file):
        print(f"Error: External dataset not found at {ext_file}.")
        return
    
    # Load model (uses local cache, downloads from HuggingFace if needed)
    device = 'cpu'
    model, tokenizer = get_model(device=device)
    
    # Evaluate
    metrics = evaluate_dataset(model, tokenizer, ext_file, device=device)
    
    print("\nExternal Dataset (HC3 Wiki) Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Save
    output_file = os.path.join(artifacts_dir, "external_metrics.json")
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to {output_file}")


if __name__ == "__main__":
    main()
