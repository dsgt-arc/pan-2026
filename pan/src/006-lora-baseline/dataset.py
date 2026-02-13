"""
Dataset utilities for LoRA fine-tuning of a causal LM (Qwen2.5-1.5B).
Loads PAN JSONL data and wraps it in HuggingFace Datasets with
left-padded tokenization appropriate for causal language models.
"""

import json
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

from config import MODEL_NAME, MAX_LENGTH


def load_jsonl(path: str) -> pd.DataFrame:
    """Reads a JSONL file into a pandas DataFrame."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return pd.DataFrame(data)


def create_hf_dataset(
    df: pd.DataFrame,
    tokenizer,
    max_length: int = MAX_LENGTH,
) -> Dataset:
    """
    Converts a DataFrame (must have 'text' and 'label' columns)
    to a tokenised HuggingFace Dataset.

    For causal LMs used in classification, we pad on the LEFT so the
    last non-padding token (used by the classification head) is always
    at the end of the sequence.
    """
    dataset = Dataset.from_pandas(df[["text", "label"]])

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )

    tokenised = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenised = tokenised.rename_column("label", "labels")
    return tokenised


def get_tokenizer():
    """Returns the Qwen2.5-1.5B tokenizer, configured for classification."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Causal LMs often lack a pad token; use eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left-padding ensures the classification head reads the last
    # real token, not a pad token
    tokenizer.padding_side = "left"
    return tokenizer


def prepare_datasets(train_path: str, val_path: str, max_length: int = MAX_LENGTH):
    """
    Loads and tokenises training + validation datasets.

    Returns:
        (train_dataset, val_dataset, tokenizer)
    """
    tokenizer = get_tokenizer()

    print(f"Loading training data from {train_path} ...")
    train_df = load_jsonl(train_path)
    print(f"  → {len(train_df)} training samples")

    print(f"Loading validation data from {val_path} ...")
    val_df = load_jsonl(val_path)
    print(f"  → {len(val_df)} validation samples")

    print("Tokenising training data ...")
    train_dataset = create_hf_dataset(train_df, tokenizer, max_length)

    print("Tokenising validation data ...")
    val_dataset = create_hf_dataset(val_df, tokenizer, max_length)

    return train_dataset, val_dataset, tokenizer
