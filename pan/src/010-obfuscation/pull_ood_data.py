"""
Download and prepare OOD evaluation datasets from HuggingFace.

Properly sources real OOD data (not synthetic splits of HC3):
- MAGE: Multi-domain AI-generated text detection dataset
- OpenGPTText: GPT-generated Wikipedia intro paragraphs
- HC3: Human ChatGPT Comparison Corpus (re-verify)
- RAID: Real AI Detection benchmark (re-verify)

All datasets are saved as standardized JSONL with:
  {"text": "...", "label": 0/1, "source": "dataset_name"}
  label 0 = human, label 1 = AI-generated

Usage:
    python pull_ood_data.py                 # download all
    python pull_ood_data.py --dataset mage  # download one
    python pull_ood_data.py --verify        # verify existing data
"""

import json
import argparse
import random
from pathlib import Path

from config import EXTERNAL_DIR

random.seed(42)


def save_jsonl(records: list[dict], path: Path):
    """Save records as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records)} records to {path}")


def download_mage(max_samples: int = 1000):
    """
    Download MAGE dataset from HuggingFace.
    Source: yaful/MAGE — multi-domain AI-generated text detection.
    """
    from datasets import load_dataset

    print("\n=== Downloading MAGE dataset ===")
    output_path = EXTERNAL_DIR / "mage.jsonl"

    try:
        ds = load_dataset("yaful/MAGE", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"  Failed to download MAGE from yaful/MAGE: {e}")
        print("  Trying alternative: yaful/MAGE with 'train' split...")
        try:
            ds = load_dataset("yaful/MAGE", split="train", trust_remote_code=True)
        except Exception as e2:
            print(f"  Failed: {e2}")
            return False

    # Sample if too large
    if len(ds) > max_samples:
        indices = random.sample(range(len(ds)), max_samples)
        ds = ds.select(indices)

    records = []
    for row in ds:
        text = row.get("text", row.get("article", ""))
        label = row.get("label", 0)
        if isinstance(label, str):
            label = 1 if label.lower() in ("machine", "ai", "generated", "1") else 0
        if text and len(text.strip()) > 50:
            records.append({"text": text.strip(), "label": int(label), "source": "mage"})

    if records:
        save_jsonl(records, output_path)
        print(f"  MAGE: {sum(1 for r in records if r['label']==0)} human, "
              f"{sum(1 for r in records if r['label']==1)} AI")
        return True
    else:
        print("  No valid records found in MAGE")
        return False


def download_opengpttext(max_samples: int = 1000):
    """
    Download OpenGPTText dataset from HuggingFace.
    Source: aadityaubhat/GPT-wiki-intro — GPT-generated Wikipedia intros.
    """
    from datasets import load_dataset

    print("\n=== Downloading OpenGPTText dataset ===")
    output_path = EXTERNAL_DIR / "opengpttext.jsonl"

    try:
        ds = load_dataset("aadityaubhat/GPT-wiki-intro", split="train", trust_remote_code=True)
    except Exception as e:
        print(f"  Failed to download OpenGPTText: {e}")
        return False

    # This dataset has original Wikipedia text and GPT-generated text
    records = []
    indices = list(range(len(ds)))
    random.shuffle(indices)

    for idx in indices:
        if len(records) >= max_samples:
            break
        row = ds[idx]

        # Add human text (Wikipedia original)
        wiki_text = row.get("wiki_intro", "")
        if wiki_text and len(wiki_text.strip()) > 50:
            records.append({"text": wiki_text.strip(), "label": 0, "source": "opengpttext"})

        # Add AI text (GPT-generated)
        gpt_text = row.get("generated_intro", "")
        if gpt_text and len(gpt_text.strip()) > 50:
            records.append({"text": gpt_text.strip(), "label": 1, "source": "opengpttext"})

    # Balance classes
    human_records = [r for r in records if r["label"] == 0]
    ai_records = [r for r in records if r["label"] == 1]
    min_count = min(len(human_records), len(ai_records), max_samples // 2)
    records = human_records[:min_count] + ai_records[:min_count]
    random.shuffle(records)

    if records:
        save_jsonl(records, output_path)
        print(f"  OpenGPTText: {sum(1 for r in records if r['label']==0)} human, "
              f"{sum(1 for r in records if r['label']==1)} AI")
        return True
    else:
        print("  No valid records found in OpenGPTText")
        return False


def download_hc3(max_samples: int = 2000):
    """
    Re-download HC3 (Human ChatGPT Comparison) from HuggingFace.
    Source: Hello-SimpleAI/HC3 — English wiki subset.
    """
    from datasets import load_dataset

    print("\n=== Downloading HC3 dataset (re-verify) ===")
    output_path = EXTERNAL_DIR / "hc3_wiki_processed.jsonl"

    try:
        ds = load_dataset("Hello-SimpleAI/HC3", "all", split="train", trust_remote_code=True)
    except Exception as e:
        print(f"  Failed to download HC3: {e}")
        return False

    records = []
    for row in ds:
        # HC3 has human_answers and chatgpt_answers
        question = row.get("question", "")
        human_answers = row.get("human_answers", [])
        chatgpt_answers = row.get("chatgpt_answers", [])

        for answer in human_answers:
            if answer and len(answer.strip()) > 50:
                records.append({"text": answer.strip(), "label": 0, "source": "hc3"})

        for answer in chatgpt_answers:
            if answer and len(answer.strip()) > 50:
                records.append({"text": answer.strip(), "label": 1, "source": "hc3"})

    # Balance and limit
    human_records = [r for r in records if r["label"] == 0]
    ai_records = [r for r in records if r["label"] == 1]
    min_count = min(len(human_records), len(ai_records), max_samples // 2)
    records = human_records[:min_count] + ai_records[:min_count]
    random.shuffle(records)

    if records:
        save_jsonl(records, output_path)
        print(f"  HC3: {sum(1 for r in records if r['label']==0)} human, "
              f"{sum(1 for r in records if r['label']==1)} AI")
        return True
    else:
        print("  No valid records found in HC3")
        return False


def download_raid(max_samples: int = 2000):
    """
    Download RAID benchmark from HuggingFace.
    Source: liamdugan/raid — multi-generator AI detection benchmark.
    """
    from datasets import load_dataset

    print("\n=== Downloading RAID dataset (re-verify) ===")
    output_path = EXTERNAL_DIR / "raid.jsonl"

    try:
        ds = load_dataset("liamdugan/raid", split="train", trust_remote_code=True)
    except Exception as e:
        print(f"  Failed to download RAID: {e}")
        return False

    records = []
    for row in ds:
        text = row.get("generation", row.get("text", ""))
        model_name = row.get("model", "")
        # In RAID, model="human" means human-written
        label = 0 if model_name == "human" else 1
        if text and len(text.strip()) > 50:
            records.append({"text": text.strip(), "label": label, "source": "raid"})

    # Sample if too large
    if len(records) > max_samples:
        human_records = [r for r in records if r["label"] == 0]
        ai_records = [r for r in records if r["label"] == 1]
        min_count = min(len(human_records), len(ai_records), max_samples // 2)
        records = human_records[:min_count] + ai_records[:min_count]
        random.shuffle(records)

    if records:
        save_jsonl(records, output_path)
        print(f"  RAID: {sum(1 for r in records if r['label']==0)} human, "
              f"{sum(1 for r in records if r['label']==1)} AI")
        return True
    else:
        print("  No valid records found in RAID")
        return False


def verify_datasets():
    """
    Verify that all OOD datasets are properly sourced and not duplicates.
    Checks that datasets have distinct text content.
    """
    print("\n=== Verifying OOD datasets ===")

    dataset_files = {
        "hc3_wiki_processed": EXTERNAL_DIR / "hc3_wiki_processed.jsonl",
        "raid": EXTERNAL_DIR / "raid.jsonl",
        "mage": EXTERNAL_DIR / "mage.jsonl",
        "opengpttext": EXTERNAL_DIR / "opengpttext.jsonl",
    }

    # Load first 100 texts from each dataset to check for duplicates
    dataset_texts = {}
    for name, path in dataset_files.items():
        if not path.exists():
            print(f"  ✗ {name}: not found")
            continue

        texts = set()
        count = 0
        labels = {0: 0, 1: 0}
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                texts.add(rec["text"][:200])  # first 200 chars as fingerprint
                labels[rec["label"]] = labels.get(rec["label"], 0) + 1
                count += 1

        dataset_texts[name] = texts
        print(f"  ✓ {name}: {count} samples (human={labels[0]}, AI={labels[1]})")

    # Check pairwise overlap
    names = list(dataset_texts.keys())
    print("\n  Pairwise text overlap (first 200 chars):")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = dataset_texts[names[i]] & dataset_texts[names[j]]
            pct_i = len(overlap) / len(dataset_texts[names[i]]) * 100 if dataset_texts[names[i]] else 0
            pct_j = len(overlap) / len(dataset_texts[names[j]]) * 100 if dataset_texts[names[j]] else 0
            status = "⚠️ HIGH" if max(pct_i, pct_j) > 10 else "✓ OK"
            print(f"    {names[i]} ↔ {names[j]}: {len(overlap)} shared "
                  f"({pct_i:.1f}% / {pct_j:.1f}%) {status}")


def main():
    parser = argparse.ArgumentParser(description="Download OOD evaluation datasets")
    parser.add_argument("--dataset", choices=["mage", "opengpttext", "hc3", "raid", "all"],
                        default="all", help="Which dataset to download")
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing datasets for duplicates")
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="Max samples per dataset (default: 1000)")
    args = parser.parse_args()

    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

    if args.verify:
        verify_datasets()
        return

    download_fns = {
        "mage": lambda: download_mage(args.max_samples),
        "opengpttext": lambda: download_opengpttext(args.max_samples),
        "hc3": lambda: download_hc3(args.max_samples),
        "raid": lambda: download_raid(args.max_samples),
    }

    if args.dataset == "all":
        for name, fn in download_fns.items():
            fn()
    else:
        download_fns[args.dataset]()

    # Always verify after downloading
    verify_datasets()


if __name__ == "__main__":
    main()
