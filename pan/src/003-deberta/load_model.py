"""
Model loading utilities for DeBERTa PAN2025 model.
Pulls the fine-tuned model from Hugging Face Hub.
"""

import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Hugging Face model repository
HF_MODEL_ID = "hersheys-baklava/deberta-pan2025"

def load_model_from_hub(
    model_id: str = HF_MODEL_ID,
    cache_dir: str = None,
    device: str = 'cpu'
):
    """
    Loads the fine-tuned DeBERTa model from Hugging Face Hub.
    
    Args:
        model_id: Hugging Face model repository ID.
        cache_dir: Optional local directory to cache the model.
        device: Device to load model on ('cpu' or 'cuda').
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model from Hugging Face: {model_id}...")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        cache_dir=cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir
    )
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    return model, tokenizer


def load_model_local(
    model_path: str,
    device: str = 'cpu'
):
    """
    Loads the model from a local path.
    
    Args:
        model_path: Local path to the saved model.
        device: Device to load model on.
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    print(f"Loading model from local path: {model_path}...")
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    return model, tokenizer


def save_model_locally(
    model,
    tokenizer,
    save_path: str
):
    """
    Saves the model and tokenizer to a local directory.
    
    Args:
        model: The model to save.
        tokenizer: The tokenizer to save.
        save_path: Directory to save to.
    """
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")


# Default local cache path
DEFAULT_LOCAL_MODEL_PATH = "/Users/tanish/Desktop/CLEF2026/pan-2026/harshul/experiments/003-deberta/artifacts/model"


def get_model(
    local_path: str = DEFAULT_LOCAL_MODEL_PATH,
    device: str = 'cpu',
    force_download: bool = False
):
    """
    Gets the model, using local cache if available.
    Downloads from HuggingFace and saves locally if not cached.
    
    Args:
        local_path: Path to check for cached model / save downloaded model.
        device: Device to load model on ('cpu' or 'cuda').
        force_download: If True, re-download from HuggingFace even if local exists.
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Check if model exists locally
    model_config_path = os.path.join(local_path, "config.json")
    
    if os.path.exists(model_config_path) and not force_download:
        # Load from local cache
        print(f"Found cached model at {local_path}")
        return load_model_local(local_path, device=device)
    else:
        # Download from HuggingFace and save locally
        print(f"Model not found locally. Downloading from HuggingFace...")
        model, tokenizer = load_model_from_hub(HF_MODEL_ID, device=device)
        
        # Save locally for future use
        print(f"Caching model to {local_path}...")
        save_model_locally(model, tokenizer, local_path)
        
        return model, tokenizer


if __name__ == "__main__":
    # Test loading from HuggingFace
    print("Testing model loading from Hugging Face Hub...")
    model, tokenizer = load_model_from_hub()
    
    # Quick inference test
    test_text = "This is a test sentence to verify the model works."
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=512)
    
    import torch
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        print(f"\nTest prediction:")
        print(f"  P(Human): {probs[0, 0].item():.4f}")
        print(f"  P(AI): {probs[0, 1].item():.4f}")
    
    print("\nModel loading test complete!")
