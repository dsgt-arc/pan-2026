"""
Minimal test to reproduce DeBERTa issue in evaluate_all context.
"""

import sys
import warnings
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

SRC_DIR = Path("/Users/tanish/Desktop/CLEF2026/pan-2026/harshul/src")
sys.path.insert(0, str(SRC_DIR / "003-deberta"))
sys.path.insert(0, str(SRC_DIR / "004-external_evaluation"))

print("=== Minimal DeBERTa Test ===")

# Load model using wrapper (same as evaluate_all.py)
print("\n1. Loading DeBERTaModel wrapper...")
from model_loaders import DeBERTaModel
model = DeBERTaModel()
print(f"   Created: {model.name}")

# Load test texts
print("\n2. Loading 10 test samples...")
import json
data_path = SRC_DIR / "pan-data/external/hc3_wiki_processed.jsonl"
texts = []
with open(data_path) as f:
    for i, line in enumerate(f):
        if i >= 10:
            break
        texts.append(json.loads(line)["text"])
print(f"   Loaded {len(texts)} samples")

# Test with tqdm exactly like evaluate_all.py
print("\n3. Testing with tqdm loop (exactly like evaluate_all.py)...")
all_probs = []
pbar = tqdm(texts, desc="  DeBERTa", leave=True, dynamic_ncols=True)
for text in pbar:
    probs = model.predict_proba([text])
    all_probs.extend(probs)
    sys.stdout.flush()

print(f"\n4. Results: {all_probs}")
print("\n=== DONE ===")
