import os
import joblib
import json
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from data_loader import get_data_splits, load_jsonl
from preprocessing import preprocess_dataframe
from stylometric_features import get_stylometric_df
from metrics import get_all_metrics

def evaluate_set(X_train, y_train, X_val, y_val, X_ext, y_ext, name):
    print(f"\nEvaluating: {name}")
    model = LGBMClassifier(random_state=42, n_jobs=-1, verbosity=-1)
    model.fit(X_train, y_train)
    
    val_probs = model.predict_proba(X_val)[:, 1]
    val_preds = model.predict(X_val)
    val_metrics = get_all_metrics(y_val, val_preds, val_probs)
    
    ext_probs = model.predict_proba(X_ext)[:, 1]
    ext_preds = model.predict(X_ext)
    ext_metrics = get_all_metrics(y_ext, ext_preds, ext_probs)
    
    return {
        'name': name,
        'val_roc_auc': val_metrics['ROC-AUC'],
        'ext_roc_auc': ext_metrics['ROC-AUC'],
        'val_f1': val_metrics['F1'],
        'ext_f1': ext_metrics['F1']
    }

def main():
    # 1. Paths
    base_data_path = "/Users/tanish/Desktop/CLEF2026/pan-2026/harshul/src/pan-data"
    train_file = os.path.join(base_data_path, "train.jsonl")
    val_file = os.path.join(base_data_path, "val.jsonl")
    ext_file = "/Users/tanish/Desktop/CLEF2026/pan-2026/harshul/src/pan-data/external/hc3_wiki_processed.jsonl"
    
    experiment_dir = "/Users/tanish/Desktop/CLEF2026/pan-2026/harshul/experiments/002-lightgbm"
    artifacts_dir = os.path.join(experiment_dir, "artifacts")
    
    # 2. Load and Preprocess
    train_df, val_df = get_data_splits(train_file, val_file)
    ext_df = load_jsonl(ext_file)
    
    train_df = preprocess_dataframe(train_df)
    val_df = preprocess_dataframe(val_df)
    ext_df = preprocess_dataframe(ext_df)
    
    y_train = train_df['label']
    y_val = val_df['label']
    y_ext = ext_df['label']
    
    # 3. Stylometric Features (precomputed)
    print("Computing stylometric features...")
    stylo_train = get_stylometric_df(train_df).values
    stylo_val = get_stylometric_df(val_df).values
    stylo_ext = get_stylometric_df(ext_df).values
    
    results = []
    
    # Experiment 1: Only TF-IDF (10k features)
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
    X_train_tfidf = vec.fit_transform(train_df['cleaned_text'])
    X_val_tfidf = vec.transform(val_df['cleaned_text'])
    X_ext_tfidf = vec.transform(ext_df['cleaned_text'])
    
    results.append(evaluate_set(X_train_tfidf, y_train, X_val_tfidf, y_val, X_ext_tfidf, y_ext, "TF-IDF Only (10k)"))
    
    # Experiment 2: Only Stylometric
    results.append(evaluate_set(stylo_train, y_train, stylo_val, y_val, stylo_ext, y_ext, "Stylo Only"))
    
    # Experiment 3: Both (Combined)
    X_train_comb = hstack([X_train_tfidf, stylo_train])
    X_val_comb = hstack([X_val_tfidf, stylo_val])
    X_ext_comb = hstack([X_ext_tfidf, stylo_ext])
    
    results.append(evaluate_set(X_train_comb, y_train, X_val_comb, y_val, X_ext_comb, y_ext, "Combined (TF-IDF 10k + Stylo)"))
    
    # Experiment 4: Smaller TF-IDF (5k) + Stylo
    vec_small = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    X_train_5k = vec_small.fit_transform(train_df['cleaned_text'])
    X_val_5k = vec_small.transform(val_df['cleaned_text'])
    X_ext_5k = vec_small.transform(ext_df['cleaned_text'])
    
    X_train_5k_comb = hstack([X_train_5k, stylo_train])
    X_val_5k_comb = hstack([X_val_5k, stylo_val])
    X_ext_5k_comb = hstack([X_ext_5k, stylo_ext])
    
    results.append(evaluate_set(X_train_5k_comb, y_train, X_val_5k_comb, y_val, X_ext_5k_comb, y_ext, "Combined (TF-IDF 5k + Stylo)"))

    # 4. Save results
    res_df = pd.DataFrame(results)
    print("\nFeature Ablation Results:")
    print(res_df)
    res_df.to_csv(os.path.join(artifacts_dir, "feature_ablation.csv"), index=False)

if __name__ == "__main__":
    main()
