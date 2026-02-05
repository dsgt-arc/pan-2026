import os
import joblib
import json
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
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
    
    # 2. Load and Preprocess Data
    train_df, val_df = get_data_splits(train_file, val_file)
    train_df = preprocess_dataframe(train_df)
    val_df = preprocess_dataframe(val_df)
    
    # 3. Feature Extraction
    X_train, vectorizer, feature_names = create_combined_features(
        train_df, 
        train=True
    )
    y_train = train_df['label']
    
    # 4. Hyperparameter Search
    print("\nStarting hyperparameter tuning...")
    param_dist = {
        'n_estimators': [50, 100, 200, 300],
        'num_leaves': [15, 31, 63, 127],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [-1, 5, 10, 15],
        'min_child_samples': [5, 10, 20, 50],
        'reg_alpha': [0, 0.1, 1.0],
        'reg_lambda': [0, 0.1, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'subsample': [0.6, 0.8, 1.0]
    }
    
    lgbm = LGBMClassifier(random_state=42, importance_type='gain', n_jobs=-1, verbosity=-1)
    
    search = RandomizedSearchCV(
        lgbm, 
        param_distributions=param_dist, 
        n_iter=20, # Limited for speed, increase for better results
        scoring='roc_auc', 
        cv=3, # 3-fold for speed
        random_state=42,
        verbose=1
    )
    
    search.fit(X_train, y_train)
    
    print(f"\nBest Score (ROC-AUC): {search.best_score_:.4f}")
    print(f"Best Parameters: {search.best_params_}")
    
    # 5. Save Search Results
    results_df = pd.DataFrame(search.cv_results_)
    results_df.to_csv(os.path.join(artifacts_dir, "hyperparameter_search.csv"), index=False)
    
    with open(os.path.join(artifacts_dir, "best_params.json"), 'w') as f:
        json.dump(search.best_params_, f, indent=4)
        
    # 6. Evaluate Best Model
    X_val, _, _ = create_combined_features(val_df, train=False, vectorizer=vectorizer)
    y_val = val_df['label']
    
    y_pred = search.predict(X_val)
    y_prob = search.predict_proba(X_val)[:, 1]
    
    metrics = get_all_metrics(y_val, y_pred, y_prob)
    print("\nBest Model Validation Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
        
    with open(os.path.join(artifacts_dir, "best_val_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()
