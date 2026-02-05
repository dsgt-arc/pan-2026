import os
import joblib
import pandas as pd
import json
from data_loader import load_jsonl
from preprocessing import preprocess_dataframe
from features import create_combined_features
from metrics import get_all_metrics

def main():
    # 1. Paths
    experiment_dir = "/Users/tanish/Desktop/CLEF2026/pan-2026/harshul/experiments/002-lightgbm"
    artifacts_dir = os.path.join(experiment_dir, "artifacts")
    
    model_path = os.path.join(artifacts_dir, "model.joblib")
    vectorizer_path = os.path.join(artifacts_dir, "vectorizer.joblib")
    
    external_file = "/Users/tanish/Desktop/CLEF2026/pan-2026/harshul/src/pan-data/external/hc3_wiki_processed.jsonl"
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("Error: Model or vectorizer not found. Please run train.py first.")
        return
    
    if not os.path.exists(external_file):
        print(f"Error: External data not found. Run experiment 001 scripts or fetch_hc3.py.")
        return

    # 2. Load Model and Vectorizer
    print("Loading model and vectorizer...")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    # 3. Load and Preprocess External Data
    print(f"Loading external data from {external_file}...")
    df = load_jsonl(external_file)
    df = preprocess_dataframe(df)
    
    # 4. Feature Extraction
    print("Extracting combined features...")
    X, _, _ = create_combined_features(
        df, 
        train=False, 
        vectorizer=vectorizer
    )
    y_true = df['label']
    
    # 5. Predict
    print("Generating predictions...")
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    # 6. Evaluate
    metrics = get_all_metrics(y_true, y_pred, y_prob)
    
    print("\nExternal Dataset (HC3 Wiki) Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # 7. Save Results
    results_path = os.path.join(artifacts_dir, "external_metrics.json")
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nExternal metrics saved to {results_path}")

if __name__ == "__main__":
    main()
