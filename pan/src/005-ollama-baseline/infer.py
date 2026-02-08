"""
Zero-shot inference pipeline for AI text detection using Ollama.
"""

import json
import re
import argparse
from pathlib import Path
from typing import Optional
import ollama
from tqdm import tqdm

from prompts import get_prompt


def parse_response(response: str, prompt_type: str = "default") -> tuple[int, float]:
    """
    Parse LLM response to extract prediction and confidence.
    
    Args:
        response: Raw response from LLM
        prompt_type: The prompt type used
    
    Returns:
        Tuple of (prediction: 0=human, 1=AI, confidence: 0.0-1.0)
    """
    response_upper = response.upper().strip()
    
    if prompt_type == "confidence":
        # Parse structured response
        classification_match = re.search(r'CLASSIFICATION:\s*(HUMAN|AI)', response_upper)
        confidence_match = re.search(r'CONFIDENCE:\s*(\d+)', response_upper)
        
        if classification_match:
            pred = 1 if classification_match.group(1) == "AI" else 0
            conf = int(confidence_match.group(1)) / 100 if confidence_match else 0.5
            # Adjust confidence based on prediction
            prob = conf if pred == 1 else (1 - conf)
            return pred, prob
    
    elif prompt_type == "cot":
        # Parse chain-of-thought response (look for FINAL:)
        final_match = re.search(r'FINAL:\s*(HUMAN|AI)', response_upper)
        if final_match:
            pred = 1 if final_match.group(1) == "AI" else 0
            return pred, float(pred)  # No confidence info
    
    # Default: simple classification
    # Look for AI or HUMAN in the response
    if "AI" in response_upper and "HUMAN" not in response_upper:
        return 1, 1.0
    elif "HUMAN" in response_upper and "AI" not in response_upper:
        return 0, 0.0
    elif "AI" in response_upper:
        # Both present, check which comes first or is more prominent
        ai_pos = response_upper.find("AI")
        human_pos = response_upper.find("HUMAN")
        if ai_pos < human_pos:
            return 1, 0.7  # Less confident
        else:
            return 0, 0.3
    
    # Fallback: couldn't parse, return uncertain
    return -1, 0.5


def run_inference(
    data_path: str,
    output_path: str,
    model: str = "qwen2:7b",
    prompt_type: str = "default",
    max_samples: Optional[int] = None,
    temperature: float = 0.0,
):
    """
    Run zero-shot inference on a dataset.
    
    Args:
        data_path: Path to input JSONL file
        output_path: Path to save predictions
        model: Ollama model name
        prompt_type: Prompt template to use
        max_samples: Limit number of samples (for testing)
        temperature: LLM temperature (0 for determinism)
    """
    # Load data
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    if max_samples:
        data = data[:max_samples]
    
    print(f"Loaded {len(data)} samples from {data_path}")
    print(f"Using model: {model}, prompt_type: {prompt_type}")
    
    results = []
    failed = 0
    
    for item in tqdm(data, desc="Running inference"):
        text = item.get("text", "")
        label = item.get("label")
        
        prompt = get_prompt(text, prompt_type)
        
        try:
            response = ollama.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "num_predict": 50,  # Short response expected
                }
            )
            
            response_text = response.get("response", "")
            pred, prob = parse_response(response_text, prompt_type)
            
            if pred == -1:
                failed += 1
                pred = 1  # Default to AI if unparseable
                prob = 0.5
            
            results.append({
                "id": item.get("id", ""),
                "label": label,
                "prediction": pred,
                "probability": prob,
                "response": response_text.strip(),
            })
            
        except Exception as e:
            print(f"Error processing sample: {e}")
            failed += 1
            results.append({
                "id": item.get("id", ""),
                "label": label,
                "prediction": 1,  # Default
                "probability": 0.5,
                "response": f"ERROR: {str(e)}",
            })
    
    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print(f"Total samples: {len(results)}, Failed to parse: {failed}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Zero-shot AI text detection via Ollama")
    parser.add_argument("--data", required=True, help="Path to input JSONL file")
    parser.add_argument("--output", required=True, help="Path to save predictions JSON")
    parser.add_argument("--model", default="qwen2:7b", help="Ollama model name")
    parser.add_argument("--prompt-type", default="default", choices=["default", "confidence", "cot"])
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for testing")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    
    args = parser.parse_args()
    
    run_inference(
        data_path=args.data,
        output_path=args.output,
        model=args.model,
        prompt_type=args.prompt_type,
        max_samples=args.max_samples,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
