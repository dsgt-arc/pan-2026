"""
Generate visualization plots from evaluation results.
Run this after evaluate.py completes to create confusion matrix and ROC curve.
"""

import os
import json
import numpy as np
from tqdm import tqdm

from data_loader import load_jsonl
from load_model import get_model
from evaluate import predict_with_chunking
from metrics import plot_confusion_matrix, plot_roc_curve


def generate_plots(data_path: str, output_dir: str, dataset_name: str = "PAN2025"):
    """
    Generates confusion matrix and ROC curve plots.
    
    Args:
        data_path: Path to JSONL data file.
        output_dir: Directory to save plots.
        dataset_name: Name for plot titles.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    device = 'cpu'
    model, tokenizer = get_model(device=device)
    
    # Load data
    df = load_jsonl(data_path)
    y_true = df['label'].tolist()
    
    # Generate predictions
    print(f"Generating predictions for {len(df)} samples...")
    y_probs = []
    for text in tqdm(df['text'], desc="Predicting"):
        prob = predict_with_chunking(model, tokenizer, text, device=device)
        y_probs.append(prob)
    
    y_probs = np.array(y_probs)
    y_pred = (y_probs >= 0.5).astype(int)
    
    # Generate plots
    cm_path = os.path.join(output_dir, f"confusion_matrix_{dataset_name.lower()}.png")
    plot_confusion_matrix(
        y_true, y_pred, 
        save_path=cm_path,
        title=f"DeBERTa Confusion Matrix - {dataset_name}"
    )
    
    roc_path = os.path.join(output_dir, f"roc_curve_{dataset_name.lower()}.png")
    plot_roc_curve(
        y_true, y_probs,
        save_path=roc_path,
        title=f"DeBERTa ROC Curve - {dataset_name}"
    )
    
    print(f"\nPlots saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate evaluation plots")
    parser.add_argument("--dataset", choices=["pan2025", "hc3"], default="pan2025",
                        help="Which dataset to generate plots for")
    args = parser.parse_args()
    
    experiment_dir = "/Users/tanish/Desktop/CLEF2026/pan-2026/harshul/experiments/003-deberta"
    artifacts_dir = os.path.join(experiment_dir, "artifacts")
    
    if args.dataset == "pan2025":
        data_path = "/Users/tanish/Desktop/CLEF2026/pan-2026/harshul/src/pan-data/val.jsonl"
        generate_plots(data_path, artifacts_dir, "PAN2025")
    else:
        data_path = "/Users/tanish/Desktop/CLEF2026/pan-2026/harshul/src/pan-data/external/hc3_wiki_processed.jsonl"
        generate_plots(data_path, artifacts_dir, "HC3")
