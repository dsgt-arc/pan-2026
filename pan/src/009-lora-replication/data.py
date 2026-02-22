"""
Data loading and tokenization for LoRA fine-tuning.
Loads PAN JSONL data and converts to HuggingFace Datasets
with left-padded tokenization (required for causal LM classification).
"""

import json
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

from config import MODEL_NAME, MAX_LENGTH


def load_jsonl(path: str) -> pd.DataFrame:
    """Read a JSONL file into a DataFrame."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return pd.DataFrame(records)


def get_tokenizer(model_name: str = MODEL_NAME):
    """
    Load the tokenizer with left-padding configured.
    Left-padding ensures the classification head reads the last real token,
    not a pad token (important for causal LM classification).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def tokenize_dataset(
    df: pd.DataFrame,
    tokenizer,
    max_length: int = MAX_LENGTH,
) -> Dataset:
    """
    Convert a DataFrame with 'text' and 'label' columns
    into a tokenized HuggingFace Dataset.
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

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized = tokenized.rename_column("label", "labels")
    return tokenized
