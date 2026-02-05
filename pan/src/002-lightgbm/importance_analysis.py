import joblib
import json
import os
import pandas as pd
import numpy as np

def main():
    experiment_dir = "/Users/tanish/Desktop/CLEF2026/pan-2026/harshul/experiments/002-lightgbm"
    artifacts_dir = os.path.join(experiment_dir, "artifacts")
    
    model_path = os.path.join(artifacts_dir, "model.joblib")
    feature_names_path = os.path.join(artifacts_dir, "feature_names.json")
    
    if not os.path.exists(model_path) or not os.path.exists(feature_names_path):
        print("Error: Model or feature names not found. Please run train.py first.")
        return
        
    model = joblib.load(model_path)
    with open(feature_names_path, 'r') as f:
        feature_names = json.load(f)
        
    importances = model.feature_importances_
    
    # Create DataFrame
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort by importance
    feat_imp = feat_imp.sort_values(by='importance', ascending=False)
    
    # Save top features
    top_20_path = os.path.join(artifacts_dir, "top_features.csv")
    feat_imp.head(20).to_csv(top_20_path, index=False)
    print(f"Top 20 features saved to {top_20_path}")
    
    # Print top 20
    print("\nTop 20 Features:")
    print(feat_imp.head(20))
    
    # Analyze Stylometric vs TF-IDF
    stylometric_set = {
        'word_count', 'char_count', 'sentence_count', 'avg_word_length', 
        'flesch_reading_ease', 'digit_count', 'uppercase_word_count', 
        'longest_sentence_length', 'repeated_token_ratio', 'punctuation_ending_ratio'
    }
    
    feat_imp['type'] = feat_imp['feature'].apply(lambda x: 'stylometric' if x in stylometric_set else 'tfidf')
    
    print("\nImportance by Feature Type:")
    print(feat_imp.groupby('type')['importance'].sum().sort_values(ascending=False))
    
    print("\nTop Stylometric Features:")
    print(feat_imp[feat_imp['type'] == 'stylometric'])

if __name__ == "__main__":
    main()
