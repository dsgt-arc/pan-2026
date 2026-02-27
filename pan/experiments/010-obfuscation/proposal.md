# Proposal: Obfuscation Augmentation for Robust AI Text Detection

**Status**: in-progress
**Created**: 2026-02-26
**Author**: tanish

## Hypothesis

Training Qwen2.5-1.5B LoRA on PAN2025 data augmented with two obfuscation strategies — homoglyph replacement and synonym substitution — will improve robustness against obfuscation attacks while maintaining in-distribution performance. Specifically:

- **PAN2025 ROC-AUC ≥ 0.999** (no regression from experiment 009)
- **Avg OOD ROC-AUC ≥ 0.95** (improvement over 009's baseline measured on properly sourced OOD data)

## Background

Experiment 008 tested homoglyph augmentation alone (from mdok) but saw mixed results: marginal HC3 improvement offset by RAID regression. Experiment 009 replicated the baseline LoRA pipeline with near-perfect in-distribution results (0.9993 ROC-AUC) and strong OOD (0.9974 RAID).

This experiment combines two augmentation techniques:
1. **Homoglyph replacement** (from experiment 008): replacing characters with visually similar unicode look-alikes
2. **Synonym substitution**: replacing words with semantically equivalent synonyms to simulate paraphrasing attacks

Both techniques force the model to learn deeper semantic features rather than surface-level patterns. Additionally, previous OOD evaluation used synthetic test data (HC3 resampled as Ghostbuster/M4/TuringBench). This experiment properly downloads and evaluates on real OOD benchmarks.

**References**:
- [008 Homoglyph Results](file:///Users/tanish/Desktop/CLEF2026/pan-2026/pan/experiments/008-homoglyph/results.md)
- [009 LoRA Replication Results](file:///Users/tanish/Desktop/CLEF2026/pan-2026/pan/experiments/009-lora-replication/results.md)

## Method

### Approach

Fork the experiment 009 LoRA pipeline. Before tokenization, augment 10% of AI-labeled training texts with homoglyphs and a separate 10% with synonym replacement. Evaluate on PAN2025 val and all properly sourced OOD benchmarks.

### Setup

| Parameter | Value |
|-----------|-------|
| **Data** | PAN2025 train + val |
| **Base Model** | `Qwen/Qwen2.5-1.5B` |
| **Homoglyph Augmentation** | 10% of AI texts, 5% per-char swap, 5% ZWJ |
| **Synonym Augmentation** | 10% of AI texts, 20% per-word replacement |
| **LoRA** | r=16, α=32, dropout=0.1, targets=q_proj,v_proj |
| **Training** | 3 epochs, lr=2e-4, bf16, batch=8+grad_accum=2 |
| **Compute** | Google Colab GPU |
| **OOD Eval** | HC3, RAID, MAGE, OpenGPTText (properly sourced) |

### Procedure

1. Download and verify all OOD evaluation datasets
2. Apply homoglyph augmentation to 10% of AI-labeled train samples
3. Apply synonym replacement to separate 10% of AI-labeled train samples
4. Fine-tune Qwen2.5-1.5B with LoRA (same base config as 009)
5. Evaluate on PAN2025 val + all OOD datasets
6. Push adapter to HuggingFace
7. Compare against experiments 006, 008, and 009

### Variables

- **Independent**: Augmentation strategy (homoglyph + synonym vs. none)
- **Dependent**: ROC-AUC, Brier Score, F1, C@1, F0.5u
- **Controlled**: All other hyperparameters match 009

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| ROC-AUC | Primary ranking metric |
| Brier Score | Probability calibration |
| F1 Score | Balanced precision/recall at 0.5 |
| C@1 | PAN metric with abstention |
| F0.5u | PAN metric penalizing false negatives |

### Baseline (from 009)

| Dataset | ROC-AUC |
|---------|---------|
| PAN2025 Val | 0.9993 |
| HC3 Wiki | 0.9977 |
| RAID | 0.9974 |

### Success Criteria

- **Confirm if**: PAN2025 ROC-AUC ≥ 0.999 AND Avg OOD ROC-AUC ≥ 0.95
- **Reject if**: PAN2025 ROC-AUC < 0.99 OR Avg OOD ROC-AUC < 0.90

## Limitations

- Synonym replacement quality depends on WordNet coverage — domain-specific terms may lack synonyms
- 10%+10% augmentation rates are taken from 008 without tuning
- Limited to word-level synonym replacement; doesn't cover sentence-level paraphrasing
- OOD dataset sizes vary, which may affect metric stability