from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

def create_tfidf_features(train_texts, val_texts, max_features=10000, save_path=None):
    """
    Fits a TF-IDF vectorizer on training texts and transforms both train and val texts.
    """
    print(f"Creating TF-IDF features (max_features={max_features})...")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    
    print(f"Train features shape: {X_train.shape}")
    print(f"Val features shape:   {X_val.shape}")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(vectorizer, save_path)
        print(f"Vectorizer saved to {save_path}")
    
    return X_train, X_val, vectorizer

def load_vectorizer(path):
    """
    Loads a saved TF-IDF vectorizer.
    """
    return joblib.load(path)
