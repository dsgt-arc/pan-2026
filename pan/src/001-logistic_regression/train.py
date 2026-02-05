import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from data_loader import get_data_splits
from preprocessing import preprocess_dataframe
from features import create_tfidf_features
from metrics import get_all_metrics

def main():
    # 1. Paths
    base_data_path = "/Users/tanish/Desktop/CLEF2026/pan-2026/harshul/src/pan-data"
    train_file = os.path.join(base_data_path, "train.jsonl")
    val_file = os.path.join(base_data_path, "val.jsonl")
    
    experiment_dir = "/Users/tanish/Desktop/CLEF2026/pan-2026/harshul/experiments/001-logistic-regression"
    artifacts_dir = os.path.join(experiment_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    model_save_path = os.path.join(artifacts_dir, "model.joblib")
    vectorizer_save_path = os.path.join(artifacts_dir, "vectorizer.joblib")
    
    # 2. Load Data
    train_df, val_df = get_data_splits(train_file, val_file)
    
    # 3. Preprocess
    train_df = preprocess_dataframe(train_df)
    val_df = preprocess_dataframe(val_df)
    
    # 4. Feature Extraction
    X_train, X_val, vectorizer = create_tfidf_features(
        train_df['cleaned_text'], 
        val_df['cleaned_text'],
        save_path=vectorizer_save_path
    )
    
    y_train = train_df['label']
    y_val = val_df['label']
    
    # 5. Train Model
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
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
    pd.Series(metrics).to_json(metrics_path)
    print(f"\nMetrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
