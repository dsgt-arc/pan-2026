"""
Debug script to test DeBERTa evaluation.
"""

import sys
import warnings
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Add paths
SRC_DIR = Path("/Users/tanish/Desktop/CLEF2026/pan-2026/harshul/src")
sys.path.insert(0, str(SRC_DIR / "003-deberta"))

print("=== DeBERTa Debug Script ===")

# Load model
print("\n1. Loading DeBERTa model...")
from load_model import get_model
model, tokenizer = get_model(device='cpu')
print("   Model loaded!")

# Load some test data
print("\n2. Loading 10 test samples...")
import json
data_path = SRC_DIR / "pan-data/external/hc3_wiki_processed.jsonl"
texts = []
labels = []
with open(data_path) as f:
    for i, line in enumerate(f):
        if i >= 10:
            break
        item = json.loads(line)
        texts.append(item["text"])
        labels.append(item["label"])
print(f"   Loaded {len(texts)} samples")

# Test prediction function
print("\n3. Testing predict_with_chunking on first sample...")
from evaluate import predict_with_chunking
prob = predict_with_chunking(model, tokenizer, texts[0], device='cpu')
print(f"   Result: {prob:.4f}")

# Test in loop with tqdm
print("\n4. Testing loop with tqdm (10 samples)...")
probs = []
for text in tqdm(texts, desc="Processing"):
    p = predict_with_chunking(model, tokenizer, text, device='cpu')
    probs.append(p)
    
print(f"   Predictions: {probs}")

# Now test the model wrapper
print("\n5. Testing DeBERTaModel wrapper...")
sys.path.insert(0, str(SRC_DIR / "004-external_evaluation"))
from model_loaders import DeBERTaModel

deberta = DeBERTaModel()
print(f"   Wrapper created: {deberta.name}")

print("\n6. Testing wrapper predict_proba on one sample...")
result = deberta.predict_proba([texts[0]])
print(f"   Result: {result}")

print("\n7. Testing wrapper predict_proba in loop with tqdm...")
all_probs = []
for text in tqdm(texts, desc="DeBERTa"):
    probs = deberta.predict_proba([text])
    all_probs.extend(probs)
print(f"   Predictions: {all_probs}")

print("\n=== DEBUG COMPLETE ===")
