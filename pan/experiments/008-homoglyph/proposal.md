# 008: Homoglyph Data Augmentation for Robust AI Text Detection

**Status**: completed
**Created**: 2026-02-16
**Author**: harshul

## Hypothesis

Training Qwen2.5-1.5B LoRA on PAN2025 data augmented with unicode homoglyph replacements will improve robustness against obfuscation attacks while maintaining in-distribution performance. Specifically:

- **PAN2025 ROC-AUC ≥ 0.999** (no regression from experiment 006)
- **OOD Avg ROC-AUC ≥ 0.91** (improvement over 006's 0.9067)

## Background

The [mdok detector](https://github.com/kinit-sk/mdok) (Macko et al., 2025) achieved strong results at PAN 2025 by using a robust fine-tuning technique that includes **homoglyph augmentation** — replacing characters with visually similar unicode look-alikes. This forces the model to learn semantic features rather than relying on surface-level character patterns that could be easily obfuscated.

The technique works by:
1. Selecting 10% of AI-labeled training texts for augmentation
2. Replacing each character with a confusable unicode character with 5% probability
3. Inserting zero-width joiner (ZWJ) characters with 5% probability

This is a data augmentation strategy that can be applied to any fine-tuning pipeline. We apply it to our existing Qwen2.5-1.5B LoRA setup from experiment 006.

**Reference**: [Increasing the Robustness of Fine-tuned Multilingual Machine-Generated Text Detectors](https://arxiv.org/abs/2503.15128)

## Method

### Approach

Replicate the experiment 006 Qwen LoRA training pipeline with one modification: augment the training data with homoglyph replacements before tokenization. Use the `confusables` Python library for character replacement, matching mdok's exact probabilities.

### Setup

| Parameter | Value |
|-----------|-------|
| **Data** | PAN2025 train + val (Google Drive) |
| **Base Model** | `Qwen/Qwen2.5-1.5B` |
| **Augmentation** | 10% of AI texts → homoglyph + ZWJ |
| **H_PROBABILITY** | 0.05 (per-character homoglyph swap) |
| **ZWJ_PROBABILITY** | 0.05 (per-character ZWJ insertion) |
| **LoRA** | r=16, α=32, dropout=0.1, targets=q_proj,v_proj |
| **Training** | 3 epochs, lr=2e-4, bf16, batch=8+grad_accum=2 |
| **Compute** | Google Colab GPU (T4/L4) |
| **Output** | HuggingFace: `hersheys-baklava/qwen-lora-homoglyph` |

### Procedure

1. Download PAN2025 train and val data
2. Apply homoglyph augmentation to 10% of AI-labeled train samples
3. Tokenize augmented training data
4. Fine-tune Qwen2.5-1.5B with LoRA (same config as 006)
5. Evaluate on PAN2025 val, HC3, RAID
6. Push adapter to HuggingFace
7. Compare against experiment 006 baseline

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| ROC-AUC | Primary ranking metric |
| Brier | Calibration quality |
| F1 | Balanced precision/recall |
| C@1 | PAN metric with abstention |
| F0.5u | PAN metric penalizing FN |

### Baseline (Experiment 006)

| Dataset | ROC-AUC |
|---------|--------:|
| PAN2025 | 0.9999 |
| HC3 | 0.9982 |
| RAID | 0.8152 |
| Avg OOD | 0.9067 |

### Success Criteria

- PAN2025 ROC-AUC ≥ 0.999 (no degradation)
- Avg OOD ROC-AUC ≥ 0.91 (improvement)

## Limitations

- Homoglyph augmentation only targets one form of obfuscation; paraphrasing attacks are not addressed
- Limited by the `confusables` library's coverage of unicode look-alikes
- 10% augmentation rate is taken from mdok without tuning — may not be optimal for our smaller model