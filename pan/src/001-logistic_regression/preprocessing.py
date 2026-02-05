import re
import string

def clean_text(text: str) -> str:
    """
    Applies basic text cleaning: lowercasing, removing special characters/punctuation, 
    and stripping extra whitespace.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove punctuation and special characters
    # Keep alphanumeric characters and spaces
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    
    # 3. Strip extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def preprocess_dataframe(df, text_column='text'):
    """
    Applies clean_text to a specific column in a DataFrame.
    """
    print(f"Preprocessing '{text_column}' column...")
    df[f'cleaned_{text_column}'] = df[text_column].apply(clean_text)
    return df

if __name__ == "__main__":
    test_text = "Hello, World! This is a test... with 123 numbers & symbols."
    cleaned = clean_text(test_text)
    print(f"Original: {test_text}")
    print(f"Cleaned:  {cleaned}")
