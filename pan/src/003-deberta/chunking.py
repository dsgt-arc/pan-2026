"""
Chunking utilities for handling long documents with DeBERTa.
Implements sliding window approach to split documents > 512 tokens.
"""

from typing import List, Tuple
import numpy as np

def chunk_text_by_tokens(
    input_ids: List[int],
    attention_mask: List[int],
    chunk_size: int = 512,
    stride: int = 256,
    pad_token_id: int = 0
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Splits tokenized input into overlapping chunks.
    
    Args:
        input_ids: Token IDs from tokenizer.
        attention_mask: Attention mask from tokenizer.
        chunk_size: Maximum tokens per chunk (default 512).
        stride: Step size between chunks (default 256 = 50% overlap).
        pad_token_id: Token ID to use for padding short chunks.
        
    Returns:
        Tuple of (chunked_input_ids, chunked_attention_masks)
    """
    # Remove special tokens for chunking, we'll add them back
    # Assuming [CLS] at start and [SEP] at end
    # Actually, for simplicity, let's work with the full sequence and just chunk
    
    if len(input_ids) <= chunk_size:
        # No chunking needed
        return [input_ids], [attention_mask]
    
    chunked_ids = []
    chunked_masks = []
    
    start = 0
    while start < len(input_ids):
        end = min(start + chunk_size, len(input_ids))
        
        chunk_ids = input_ids[start:end]
        chunk_mask = attention_mask[start:end]
        
        # Pad if necessary (for the last chunk)
        if len(chunk_ids) < chunk_size:
            padding_length = chunk_size - len(chunk_ids)
            chunk_ids = chunk_ids + [pad_token_id] * padding_length
            chunk_mask = chunk_mask + [0] * padding_length
        
        chunked_ids.append(chunk_ids)
        chunked_masks.append(chunk_mask)
        
        start += stride
        
        # Avoid creating a chunk that's mostly padding
        if start + stride >= len(input_ids):
            break
    
    return chunked_ids, chunked_masks


def aggregate_predictions(
    chunk_probs: List[float],
    method: str = 'mean'
) -> float:
    """
    Aggregates predictions from multiple chunks into a single probability.
    
    Args:
        chunk_probs: List of probabilities from each chunk.
        method: Aggregation method ('mean' or 'max').
        
    Returns:
        Aggregated probability.
    """
    if not chunk_probs:
        return 0.5  # Default to uncertain
    
    if method == 'mean':
        return float(np.mean(chunk_probs))
    elif method == 'max':
        return float(np.max(chunk_probs))
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


if __name__ == "__main__":
    # Quick test
    print("Testing chunking with 1000 tokens...")
    fake_ids = list(range(1000))
    fake_mask = [1] * 1000
    
    chunks_ids, chunks_mask = chunk_text_by_tokens(fake_ids, fake_mask)
    print(f"Number of chunks: {len(chunks_ids)}")
    for i, chunk in enumerate(chunks_ids):
        print(f"  Chunk {i}: tokens {chunk[0]} to {chunk[-1]}, length {len(chunk)}")
    
    print("\nTesting aggregation...")
    probs = [0.7, 0.8, 0.6, 0.9]
    print(f"  Mean: {aggregate_predictions(probs, 'mean'):.4f}")
    print(f"  Max:  {aggregate_predictions(probs, 'max'):.4f}")
