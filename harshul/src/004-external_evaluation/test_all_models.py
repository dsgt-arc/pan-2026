"""
Test loading all models together like evaluate_all.py does.
"""

import sys
import warnings
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

SRC_DIR = Path("/Users/tanish/Desktop/CLEF2026/pan-2026/harshul/src")
sys.path.insert(0, str(SRC_DIR / "003-deberta"))
sys.path.insert(0, str(SRC_DIR / "004-external_evaluation"))

print("=== Test All Models Together ===")

# Load ALL models (same as evaluate_all.py)
print("\n1. Loading all models with load_all_models()...")
from model_loaders import load_all_models
models = load_all_models(skip_deberta=False)
print(f"   Loaded: {list(models.keys())}")

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

# Test each model
for model_name, model in models.items():
    print(f"\n3. Testing {model_name}...")
    all_probs = []
    
    if model.name == "DeBERTa":
        pbar = tqdm(texts, desc=f"  {model.name}", leave=True, dynamic_ncols=True)
        for text in pbar:
            probs = model.predict_proba([text])
            all_probs.extend(probs)
            sys.stdout.flush()
    else:
        probs = model.predict_proba(texts)
        all_probs.extend(probs)
    
    print(f"   Got {len(all_probs)} predictions")

print("\n=== DONE ===")
