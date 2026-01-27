import pandas as pd
import json
import os

def load_jsonl(path: str) -> pd.DataFrame:
    """
    Reads a .jsonl file and returns a pandas DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    return df

def get_data_splits(train_path: str, val_path: str):
    """
    Loads train and validation data splits.
    """
    print(f"Loading training data from {train_path}...")
    train_df = load_jsonl(train_path)
    print(f"Loaded {len(train_df)} training samples.")
    
    print(f"Loading validation data from {val_path}...")
    val_df = load_jsonl(val_path)
    print(f"Loaded {len(val_df)} validation samples.")
    
    # Class distribution
    print("\nTraining Class Distribution:")
    print(train_df['label'].value_counts(normalize=True))
    
    return train_df, val_df

if __name__ == "__main__":
    # Example usage/sanity check
    base_path = "/Users/tanish/Desktop/CLEF2026/pan-2026/harshul/src/pan-data"
    train_file = os.path.join(base_path, "train.jsonl")
    val_file = os.path.join(base_path, "val.jsonl")
    
    train, val = get_data_splits(train_file, val_file)
    print("\nColumns:", train.columns.tolist())
    print("\nFirst sample text head:", train['text'].iloc[0][:100])
