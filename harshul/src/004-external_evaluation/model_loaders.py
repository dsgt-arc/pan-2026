"""
Unified model loading for all three trained models.
"""

import os
import sys
import joblib
import numpy as np
from pathlib import Path

# Add src directories to path
SRC_DIR = Path("/Users/tanish/Desktop/CLEF2026/pan-2026/harshul/src")
sys.path.insert(0, str(SRC_DIR / "001-logistic-regression"))
sys.path.insert(0, str(SRC_DIR / "003-deberta"))

# Artifact paths
EXPERIMENTS_DIR = Path("/Users/tanish/Desktop/CLEF2026/pan-2026/harshul/experiments")
LR_MODEL_PATH = EXPERIMENTS_DIR / "001-logistic-regression/artifacts/model.joblib"
LR_VECTORIZER_PATH = EXPERIMENTS_DIR / "001-logistic-regression/artifacts/vectorizer.joblib"
LGBM_MODEL_PATH = EXPERIMENTS_DIR / "002-lightgbm/artifacts/model.joblib"
LGBM_VECTORIZER_PATH = EXPERIMENTS_DIR / "002-lightgbm/artifacts/vectorizer.joblib"


class LogisticRegressionModel:
    """Wrapper for Logistic Regression model."""
    
    def __init__(self):
        self.model = joblib.load(LR_MODEL_PATH)
        self.vectorizer = joblib.load(LR_VECTORIZER_PATH)
        self.name = "LogisticRegression"
    
    def predict_proba(self, texts: list) -> np.ndarray:
        """Return probability of AI class (label=1)."""
        X = self.vectorizer.transform(texts)
        probs = self.model.predict_proba(X)[:, 1]
        return probs


class LightGBMModel:
    """Wrapper for LightGBM model with combined TF-IDF + stylometric features."""
    
    def __init__(self):
        self.model = joblib.load(LGBM_MODEL_PATH)
        self.vectorizer = joblib.load(LGBM_VECTORIZER_PATH)
        self.name = "LightGBM"
        
        # Import stylometric feature extraction
        sys.path.insert(0, str(SRC_DIR / "002-lightgbm"))
        from stylometric_features import get_stylometric_df
        from preprocessing import clean_text
        self.get_stylometric_df = get_stylometric_df
        self.clean_text = clean_text
    
    def predict_proba(self, texts: list) -> np.ndarray:
        """Return probability of AI class (label=1) using combined features."""
        import pandas as pd
        from scipy.sparse import hstack
        
        # Create dataframe
        df = pd.DataFrame({"text": texts})
        df["cleaned_text"] = df["text"].apply(self.clean_text)
        
        # TF-IDF features
        X_tfidf = self.vectorizer.transform(df["cleaned_text"])
        
        # Stylometric features
        df_stylo = self.get_stylometric_df(df, text_column="text")
        
        # Combine
        X_combined = hstack([X_tfidf, df_stylo.values])
        
        probs = self.model.predict_proba(X_combined)[:, 1]
        return probs


class DeBERTaModel:
    """Wrapper for DeBERTa model with chunking support."""
    
    def __init__(self, device: str = 'cpu'):
        from load_model import get_model
        self.model, self.tokenizer = get_model(device=device)
        self.device = device
        self.name = "DeBERTa"
    
    def predict_proba(self, texts: list) -> np.ndarray:
        """Return probability of AI class (label=1) with chunking."""
        from evaluate import predict_with_chunking
        
        probs = []
        for text in texts:
            prob = predict_with_chunking(
                self.model, self.tokenizer, text, device=self.device
            )
            probs.append(prob)
        return np.array(probs)


def load_model(model_name: str):
    """Load a model by name."""
    if model_name.lower() in ["lr", "logistic", "logisticregression"]:
        return LogisticRegressionModel()
    elif model_name.lower() in ["lgbm", "lightgbm"]:
        return LightGBMModel()
    elif model_name.lower() in ["deberta", "transformer"]:
        return DeBERTaModel()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def load_all_models(skip_deberta: bool = False):
    """Load all three models."""
    models = {}
    
    print("Loading models...")
    
    # DeBERTa FIRST (before LightGBM imports textstat which may interfere)
    if not skip_deberta:
        try:
            models["DeBERTa"] = DeBERTaModel()
            print("  ✓ DeBERTa loaded")
        except Exception as e:
            print(f"  ✗ DeBERTa failed: {e}")
    else:
        print("  ⏭ DeBERTa skipped (skip_deberta=True)")
    
    # LR
    try:
        models["LR"] = LogisticRegressionModel()
        print("  ✓ LogisticRegression loaded")
    except Exception as e:
        print(f"  ✗ LogisticRegression failed: {e}")
    
    # LightGBM
    try:
        models["LightGBM"] = LightGBMModel()
        print("  ✓ LightGBM loaded")
    except Exception as e:
        print(f"  ✗ LightGBM failed: {e}")
    
    return models


if __name__ == "__main__":
    # Test loading all models
    models = load_all_models()
    print(f"\nLoaded {len(models)} models: {list(models.keys())}")
