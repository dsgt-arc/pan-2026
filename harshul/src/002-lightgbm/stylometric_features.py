import re
import pandas as pd
import numpy as np
import textstat

def extract_stylometric_features(text: str) -> dict:
    """
    Extracts hand-crafted linguistic statistics from a given text.
    """
    if not isinstance(text, str) or not text.strip():
        return {
            'word_count': 0,
            'char_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0,
            'flesch_reading_ease': 0,
            'digit_count': 0,
            'uppercase_word_count': 0,
            'longest_sentence_length': 0,
            'repeated_token_ratio': 0,
            'punctuation_ending_ratio': 0
        }

    # Basic stats
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    
    # Sentences - split by . ! ?
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)
    
    # Average word length
    avg_word_length = char_count / word_count if word_count > 0 else 0
    
    # Flesch reading ease
    # Using textstat library
    try:
        flesch_score = textstat.flesch_reading_ease(text)
    except:
        flesch_score = 0
        
    # Digit count
    digit_count = sum(c.isdigit() for c in text)
    
    # Uppercase word count
    uppercase_word_count = sum(1 for w in words if w.isupper() and len(w) > 1)
    
    # Longest sentence length (in words)
    if sentences:
        longest_sentence_length = max(len(s.split()) for s in sentences)
    else:
        longest_sentence_length = 0
        
    # Repeated token ratio
    if word_count > 0:
        unique_words = set(w.lower() for w in words)
        repeated_token_ratio = (word_count - len(unique_words)) / word_count
    else:
        repeated_token_ratio = 0
        
    # Punctuation ending ratio
    # Frequency of . ! ? as last characters of segments
    # Actually requested: Fraction of sentences ending with . ! ? vs total sentences
    # Since we split by . ! ?, we can count how many of these chars were at the end of parts
    punct_endings = len(re.findall(r'[.!?](\s|$)', text))
    # We'll avoid dividing by zero
    punctuation_ending_ratio = punct_endings / sentence_count if sentence_count > 0 else 0
    
    return {
        'word_count': word_count,
        'char_count': char_count,
        'sentence_count': sentence_count,
        'avg_word_length': avg_word_length,
        'flesch_reading_ease': flesch_score,
        'digit_count': digit_count,
        'uppercase_word_count': uppercase_word_count,
        'longest_sentence_length': longest_sentence_length,
        'repeated_token_ratio': repeated_token_ratio,
        'punctuation_ending_ratio': punctuation_ending_ratio
    }

def get_stylometric_df(df, text_column='text'):
    """
    Applies feature extraction to a DataFrame and returns a dense DataFrame of features.
    """
    print(f"Extracting stylometric features from '{text_column}'...")
    features_list = df[text_column].apply(extract_stylometric_features).tolist()
    return pd.DataFrame(features_list)

if __name__ == "__main__":
    # Quick sanity check
    sample_text = "This is a TEST! It has 123 digits. This is a longer sentence intended to be the longest sentence in this small proof of concept."
    features = extract_stylometric_features(sample_text)
    for k, v in features.items():
        print(f"{k}: {v}")
