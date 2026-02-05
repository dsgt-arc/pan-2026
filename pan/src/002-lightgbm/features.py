from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import joblib
import os
import pandas as pd

def create_combined_features(df, train=True, vectorizer=None, max_tfidf_features=10000, save_path=None):
    """
    Combines TF-IDF features with stylometric features.
    If train=True, fits the vectorizer.
    """
    print(f"Creating combined features (train={train})...")
    
    # 1. Stylometric Features (already computed in df for convenience, or compute here)
    # For this implementation, we assume they are already columns in df or we can extract them
    # Let's assume passed df has 'text' column and 'cleaned_text' column
    
    from stylometric_features import get_stylometric_df
    
    # Stylometric part
    df_stylo = get_stylometric_df(df, text_column='text')
    
    # 2. TF-IDF part
    if train:
        print(f"Fitting TF-IDF on training data (max_features={max_tfidf_features})...")
        vectorizer = TfidfVectorizer(
            max_features=max_tfidf_features,
            ngram_range=(1, 2),
            stop_words='english'
        )
        X_tfidf = vectorizer.fit_transform(df['cleaned_text'])
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(vectorizer, save_path)
            print(f"Vectorizer saved to {save_path}")
    else:
        if vectorizer is None:
            raise ValueError("Vectorizer must be provided for non-training mode.")
        X_tfidf = vectorizer.transform(df['cleaned_text'])
        
    # 3. Combine
    print("Combining TF-IDF and stylometric features...")
    X_combined = hstack([X_tfidf, df_stylo.values])
    
    # Feature names
    tfidf_names = vectorizer.get_feature_names_out().tolist()
    stylo_names = df_stylo.columns.tolist()
    feature_names = tfidf_names + stylo_names
    
    print(f"Final feature shape: {X_combined.shape}")
    
    return X_combined, vectorizer, feature_names

def load_vectorizer(path):
    return joblib.load(path)
