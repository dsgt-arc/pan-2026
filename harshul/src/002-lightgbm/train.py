import os
import joblib
import json
import pandas as pd
from lightgbm import LGBMClassifier
from data_loader import get_data_splits
from preprocessing import preprocess_dataframe
from features import create_combined_features
from metrics import get_all_metrics

def main():
    # 1. Paths
    base_data_path = "/Users/tanish/Desktop/CLEF2026/pan-2026/harshul/src/pan-data"
    train_file = os.path.join(base_data_path, "train.jsonl")
    val_file = os.path.join(base_data_path, "val.jsonl")
    
    experiment_dir = "/Users/tanish/Desktop/CLEF2026/pan-2026/harshul/experiments/002-lightgbm"
    artifacts_dir = os.path.join(experiment_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    model_save_path = os.path.join(artifacts_dir, "model.joblib")
    vectorizer_save_path = os.path.join(artifacts_dir, "vectorizer.joblib")
    feature_names_path = os.path.join(artifacts_dir, "feature_names.json")
    
    # 2. Load Data
    train_df, val_df = get_data_splits(train_file, val_file)
    
    # 3. Preprocess
    train_df = preprocess_dataframe(train_df)
    val_df = preprocess_dataframe(val_df)
    
    # 4. Feature Extraction (Combined TF-IDF + Stylometric)
    X_train, vectorizer, feature_names = create_combined_features(
        train_df, 
        train=True, 
        save_path=vectorizer_save_path
    )
    
    # Transform Validation Data
    X_val, _, _ = create_combined_features(
        val_df, 
        train=False, 
        vectorizer=vectorizer
    )
    
    y_train = train_df['label']
    y_val = val_df['label']
    
    # Save feature names
    with open(feature_names_path, 'w') as f:
        json.dump(feature_names, f)
    
    # 5. Train Model
    print("\nTraining LightGBM model...")
    model = LGBMClassifier(
        n_estimators=100,
        num_leaves=31,
        learning_rate=0.1,
        random_state=42,
        importance_type='gain' # Better for understanding contribution
    )
    model.fit(X_train, y_train)
    
    # 6. Save Model
    joblib.dump(model, model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # 7. Evaluate on Validation Set
    print("\nEvaluating on validation set...")
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    
    metrics = get_all_metrics(y_val, y_pred, y_prob)
    
    print("\nValidation Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # 8. Save Metrics to Artifacts
    metrics_path = os.path.join(artifacts_dir, "val_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
