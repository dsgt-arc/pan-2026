"""
Simple test script to debug DeBERTa inference.
"""

import sys
from pathlib import Path
import time

# Add path to load_model
SRC_DIR = Path("/Users/tanish/Desktop/CLEF2026/pan-2026/harshul/src")
sys.path.insert(0, str(SRC_DIR / "003-deberta"))

print("=== DeBERTa Test Script ===")

# 1. Load model
print("\n1. Loading model...")
start = time.time()
from load_model import get_model
model, tokenizer = get_model(device='cpu')
print(f"   Model loaded in {time.time() - start:.2f}s")

# 2. Simple inference test
print("\n2. Testing inference on single text...")
test_text = "This is a simple test sentence to check if the model works."

start = time.time()
from evaluate import predict_with_chunking
prob = predict_with_chunking(model, tokenizer, test_text, device='cpu')
print(f"   Prediction: {prob:.4f} (took {time.time() - start:.2f}s)")

# 3. Batch test
print("\n3. Testing batch of 5 texts...")
texts = [
    "Hello world, this is test one.",
    "Machine learning is fascinating.",
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is changing everything.",
    "This is the fifth test sentence."
]

start = time.time()
for i, text in enumerate(texts):
    prob = predict_with_chunking(model, tokenizer, text, device='cpu')
    print(f"   Text {i+1}: prob={prob:.4f}")
print(f"   Total time: {time.time() - start:.2f}s")

print("\n=== Done ===")
