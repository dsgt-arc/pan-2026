import os
import joblib
import pandas as pd
import json
from data_loader import load_jsonl
from preprocessing import preprocess_dataframe
from metrics import get_all_metrics

def main():
    # 1. Paths
    experiment_dir = "/Users/tanish/Desktop/CLEF2026/pan-2026/harshul/experiments/001-logistic-regression"
    artifacts_dir = os.path.join(experiment_dir, "artifacts")
    
    model_path = os.path.join(artifacts_dir, "model.joblib")
    vectorizer_path = os.path.join(artifacts_dir, "vectorizer.joblib")
    
    val_file = "/Users/tanish/Desktop/CLEF2026/pan-2026/harshul/src/pan-data/val.jsonl"
    
    # Check if artifacts exist
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("Error: Model or vectorizer not found in artifacts. Please run train.py first.")
        return
    
    # 2. Load Model and Vectorizer
    print("Loading model and vectorizer...")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    # 3. Load and Preprocess Validation Data
    print(f"Loading validation data from {val_file}...")
    val_df = load_jsonl(val_file)
    val_df = preprocess_dataframe(val_df)
    
    # 4. Feature Extraction
    print("Extracting features...")
    X_val = vectorizer.transform(val_df['cleaned_text'])
    y_val = val_df['label']
    
    # 5. Predict
    print("Generating predictions...")
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    
    # 6. Evaluate
    metrics = get_all_metrics(y_val, y_pred, y_prob)
    
    print("\nValidation Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # 7. Save Detailed Results (Optional)
    results_path = os.path.join(artifacts_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nDetailed metrics saved to {results_path}")

if __name__ == "__main__":
    main()
