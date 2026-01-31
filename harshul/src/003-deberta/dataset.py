"""
Dataset utilities for DeBERTa fine-tuning.
Handles tokenization and HuggingFace Dataset wrapping.
"""

import os
import json
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 512

def load_jsonl(path: str) -> pd.DataFrame:
    """Loads a JSONL file into a pandas DataFrame."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)


def create_hf_dataset(df: pd.DataFrame, tokenizer, max_length: int = MAX_LENGTH) -> Dataset:
    """
    Converts a DataFrame to a HuggingFace Dataset with tokenization.
    
    Args:
        df: DataFrame with 'text' and 'label' columns.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length.
        
    Returns:
        HuggingFace Dataset ready for training.
    """
    dataset = Dataset.from_pandas(df[['text', 'label']])
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors=None  # Return lists for Dataset
        )
    
    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    tokenized = tokenized.rename_column('label', 'labels')
    
    return tokenized


def get_tokenizer():
    """Returns the DeBERTa-v3-base tokenizer."""
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def prepare_datasets(train_path: str, val_path: str, max_length: int = MAX_LENGTH):
    """
    Loads and tokenizes training and validation datasets.
    
    Args:
        train_path: Path to training JSONL file.
        val_path: Path to validation JSONL file.
        max_length: Maximum sequence length.
        
    Returns:
        Tuple of (train_dataset, val_dataset, tokenizer)
    """
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = get_tokenizer()
    
    print(f"Loading training data from {train_path}...")
    train_df = load_jsonl(train_path)
    print(f"Loaded {len(train_df)} training samples.")
    
    print(f"Loading validation data from {val_path}...")
    val_df = load_jsonl(val_path)
    print(f"Loaded {len(val_df)} validation samples.")
    
    print("Tokenizing training data...")
    train_dataset = create_hf_dataset(train_df, tokenizer, max_length)
    
    print("Tokenizing validation data...")
    val_dataset = create_hf_dataset(val_df, tokenizer, max_length)
    
    return train_dataset, val_dataset, tokenizer


if __name__ == "__main__":
    # Quick test
    base_path = "/Users/tanish/Desktop/CLEF2026/pan-2026/harshul/src/pan-data"
    train_path = os.path.join(base_path, "train.jsonl")
    val_path = os.path.join(base_path, "val.jsonl")
    
    train_ds, val_ds, tokenizer = prepare_datasets(train_path, val_path)
    
    print(f"\nTrain dataset: {train_ds}")
    print(f"Val dataset: {val_ds}")
    print(f"\nSample tokenized input: {train_ds[0]}")
