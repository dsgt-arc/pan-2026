#!/usr/bin/env python3
"""
PAN 2026 TIRA Submission — Experiment 010 (Obfuscation-Robust LoRA).

Loads the Qwen2.5-1.5B + LoRA adapter, processes each test case in isolation,
and writes predictions as JSONL with confidence scores.

Usage (TIRA invocation):
    python run_submission.py --input $inputDataset/dataset.jsonl --output $outputDir

The merged model is expected at /model inside the Docker container (baked in
at build time). For local testing, pass --model-path to override.
"""

import sys
import json
from pathlib import Path
from typing import List, Tuple

import click
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from peft import PeftModel, AutoPeftModelForSequenceClassification

# ── Constants ──────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
NUM_LABELS = 2
MAX_LENGTH = 512
CHUNK_SIZE = 512
CHUNK_STRIDE = 256  # 50 % overlap
DEFAULT_MODEL_PATH = "/model"


# ── Chunking (inlined from 003-deberta/chunking.py) ───────────────
def chunk_text_by_tokens(
    input_ids: List[int],
    attention_mask: List[int],
    chunk_size: int = CHUNK_SIZE,
    stride: int = CHUNK_STRIDE,
    pad_token_id: int = 0,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Split tokenised input into overlapping chunks."""
    if len(input_ids) <= chunk_size:
        return [input_ids], [attention_mask]

    chunked_ids, chunked_masks = [], []
    start = 0
    while start < len(input_ids):
        end = min(start + chunk_size, len(input_ids))
        c_ids = input_ids[start:end]
        c_mask = attention_mask[start:end]
        if len(c_ids) < chunk_size:
            pad_len = chunk_size - len(c_ids)
            c_ids = c_ids + [pad_token_id] * pad_len
            c_mask = c_mask + [0] * pad_len
        chunked_ids.append(c_ids)
        chunked_masks.append(c_mask)
        start += stride
        if start + stride >= len(input_ids):
            break
    return chunked_ids, chunked_masks


# ── Model loading ─────────────────────────────────────────────────
def load_model(model_path: str, device: str = "cpu"):
    """
    Load the model for inference.

    Tries loading as a pre-merged model first (Docker scenario).
    Falls back to PEFT adapter merge (local testing with HF repo).
    """
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, torch_dtype="auto",
        )
        print(f"[submission] loaded pre-merged model from {model_path}",
              file=sys.stderr)
    except Exception:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(MODEL_NAME)
        config.pad_token_id = config.eos_token_id
        config.num_labels = NUM_LABELS
        config.problem_type = "single_label_classification"

        base_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, config=config, torch_dtype="auto",
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        print(f"[submission] loaded adapter from {model_path} and merged",
              file=sys.stderr)

    model.to(device)
    model.eval()
    return model


def get_tokenizer(model_path: str = DEFAULT_MODEL_PATH):
    """Load tokenizer — tries the merged model dir first, then the base model."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


# ── Single-text inference ─────────────────────────────────────────
def predict_single(model, tokenizer, text: str, device: str = "cpu") -> float:
    """Return P(AI-generated) for one text using chunked inference."""
    encoding = tokenizer(text, truncation=False, return_tensors="pt",
                         add_special_tokens=True)
    input_ids = encoding["input_ids"][0].tolist()
    attention_mask = encoding["attention_mask"][0].tolist()

    chunked_ids, chunked_masks = chunk_text_by_tokens(
        input_ids, attention_mask,
        chunk_size=CHUNK_SIZE, stride=CHUNK_STRIDE,
        pad_token_id=tokenizer.pad_token_id,
    )

    chunk_probs = []
    with torch.no_grad():
        for ids, mask in zip(chunked_ids, chunked_masks):
            inputs = {
                "input_ids": torch.tensor([ids]).to(device),
                "attention_mask": torch.tensor([mask]).to(device),
            }
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            chunk_probs.append(probs[0, 1].item())  # P(label=1) = P(AI)

    return float(np.mean(chunk_probs)) if chunk_probs else 0.5


# ── Main ──────────────────────────────────────────────────────────
@click.command()
@click.option("--input", "input_file", type=click.Path(exists=True), required=True,
              help="Absolute path to the input JSONL file (dataset.jsonl).")
@click.option("--output", "output_dir", type=click.Path(), required=True,
              help="Absolute path to the output directory.")
@click.option("--model-path", type=str, default=DEFAULT_MODEL_PATH,
              help="Path to merged model or HF adapter repo (default: /model).")
def main(input_file, output_dir, model_path):
    input_path = Path(input_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[submission] device={device}", file=sys.stderr)

    # Load model once
    print(f"[submission] loading model from {model_path}", file=sys.stderr)
    model = load_model(model_path, device=device)
    tokenizer = get_tokenizer(model_path)

    # Read input
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"[submission] {len(records)} test cases", file=sys.stderr)

    # Process each case independently
    output_path = output_dir / "predictions.jsonl"
    with open(output_path, "w", encoding="utf-8") as out:
        for rec in records:
            text_id = rec["id"]
            text = rec["text"]
            score = predict_single(model, tokenizer, text, device=device)
            out.write(json.dumps({"id": text_id, "label": round(score, 4)})
                      + "\n")
            print(f"  {text_id}: {score:.4f}", file=sys.stderr)

    print(f"[submission] wrote {len(records)} predictions to {output_path}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
