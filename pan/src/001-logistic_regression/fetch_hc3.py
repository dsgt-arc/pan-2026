import requests
import json
import os
import pandas as pd

def download_file(url, local_path):
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Saved to {local_path}")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

def prepare_hc3_subset(input_path, output_path, max_samples=1000):
    """
    Transforms HC3 format (question, human_answers, chatgpt_answers) 
    to PAN format (text, label).
    """
    print(f"Preparing HC3 subset from {input_path}...")
    combined_data = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            
            entry = json.loads(line)
            # Add human answers
            for ans in entry.get('human_answers', []):
                combined_data.append({'text': ans, 'label': 0, 'source': 'hc3_human'})
            # Add chatgpt answers
            for ans in entry.get('chatgpt_answers', []):
                combined_data.append({'text': ans, 'label': 1, 'source': 'hc3_chatgpt'})
    
    df = pd.DataFrame(combined_data)
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save as JSONL
    df.to_json(output_path, orient='records', lines=True)
    print(f"Prepared {len(df)} samples and saved to {output_path}")

if __name__ == "__main__":
    # URL for wiki_csai.jsonl from Hello-SimpleAI/HC3
    url = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/wiki_csai.jsonl"
    
    data_dir = "/Users/tanish/Desktop/CLEF2026/pan-2026/harshul/src/pan-data/external"
    os.makedirs(data_dir, exist_ok=True)
    
    raw_path = os.path.join(data_dir, "hc3_wiki_raw.jsonl")
    processed_path = os.path.join(data_dir, "hc3_wiki_processed.jsonl")
    
    # Download if not exists
    if not os.path.exists(raw_path):
        download_file(url, raw_path)
    
    # Process
    prepare_hc3_subset(raw_path, processed_path)
