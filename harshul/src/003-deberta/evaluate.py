"""
Evaluation script for DeBERTa model with long document support.
Uses chunking + aggregation for documents > 512 tokens.
"""

import os
import json
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

from data_loader import load_jsonl
from chunking import chunk_text_by_tokens, aggregate_predictions
from metrics import get_all_metrics
from load_model import get_model

def predict_with_chunking(
    model, 
    tokenizer, 
    text: str, 
    max_length: int = 512,
    stride: int = 256,
    device: str = 'cpu'
) -> float:
    """
    Predicts the probability that a text is AI-generated.
    Handles long documents via chunking.
    
    Args:
        model: Fine-tuned DeBERTa model.
        tokenizer: DeBERTa tokenizer.
        text: Input text.
        max_length: Maximum chunk size.
        stride: Overlap between chunks.
        device: 'cpu' or 'cuda'.
        
    Returns:
        Probability that the text is AI-generated.
    """
    # Tokenize
    encoding = tokenizer(
        text,
        truncation=False,  # Don't truncate, we'll chunk manually
        return_tensors='pt',
        add_special_tokens=True
    )
    
    input_ids = encoding['input_ids'][0].tolist()
    attention_mask = encoding['attention_mask'][0].tolist()
    
    # Chunk if necessary
    chunked_ids, chunked_masks = chunk_text_by_tokens(
        input_ids, 
        attention_mask, 
        chunk_size=max_length, 
        stride=stride,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Predict each chunk
    chunk_probs = []
    model.eval()
    with torch.no_grad():
        for chunk_ids, chunk_mask in zip(chunked_ids, chunked_masks):
            inputs = {
                'input_ids': torch.tensor([chunk_ids]).to(device),
                'attention_mask': torch.tensor([chunk_mask]).to(device)
            }
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            chunk_probs.append(probs[0, 1].item())  # Probability of class 1 (AI)
    
    # Aggregate
    return aggregate_predictions(chunk_probs, method='mean')


def evaluate_dataset(model, tokenizer, data_path: str, device: str = 'cpu'):
    """
    Evaluates the model on a dataset.
    
    Args:
        model: Fine-tuned DeBERTa model.
        tokenizer: DeBERTa tokenizer.
        data_path: Path to JSONL file.
        device: 'cpu' or 'cuda'.
        
    Returns:
        Dictionary of metrics.
    """
    df = load_jsonl(data_path)
    
    y_true = df['label'].tolist()
    y_probs = []
    
    print(f"Evaluating {len(df)} samples...")
    for text in tqdm(df['text'], desc="Predicting"):
        prob = predict_with_chunking(model, tokenizer, text, device=device)
        y_probs.append(prob)
    
    y_probs = np.array(y_probs)
    y_pred = (y_probs >= 0.5).astype(int)
    
    metrics = get_all_metrics(y_true, y_pred, y_probs)
    return metrics


def main():
    # Paths
    experiment_dir = "/Users/tanish/Desktop/CLEF2026/pan-2026/harshul/experiments/003-deberta"
    artifacts_dir = os.path.join(experiment_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    val_file = "/Users/tanish/Desktop/CLEF2026/pan-2026/harshul/src/pan-data/val.jsonl"
    
    if not os.path.exists(val_file):
        print(f"Error: Validation data not found at {val_file}.")
        return
    
    # Load model (uses local cache, downloads from HuggingFace if needed)
    device = 'cpu'
    model, tokenizer = get_model(device=device)
    
    # Evaluate
    metrics = evaluate_dataset(model, tokenizer, val_file, device=device)
    
    print("\nValidation Metrics (PAN2025):")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Save
    output_file = os.path.join(artifacts_dir, "val_metrics.json")
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to {output_file}")


if __name__ == "__main__":
    main()
