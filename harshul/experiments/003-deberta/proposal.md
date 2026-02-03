# Proposal: DeBERTa-v3 for Robust AI Text Detection

**Status**: proposed
**Created**: 2026-01-30
**Author**: harshul

## Hypothesis

A pre-trained Transformer model (**DeBERTa-v3**) will significantly improve out-of-distribution (OOD) generalization compared to TF-IDF based baselines. By learning deep semantic and contextual representations rather than shallow lexical "accents" or hand-crafted stylometrics, the model will achieve an **ROC-AUC > 0.70 on the HC3 external dataset** while maintaining high in-distribution performance.

## Background

Experiments [001-logistic-regression](../001-logistic-regression/results.md) and [002-lightgbm](../002-lightgbm/results.md) revealed a major "generalization gap":
- **In-distribution (PAN val)**: > 0.99 ROC-AUC.
- **Out-of-distribution (HC3)**: 0.33 - 0.55 ROC-AUC.

The previous models relied on bag-of-words (TF-IDF) and hand-crafted stylometrics, which captured dataset-specific artifacts (e.g., the specific length or vocabulary of the LLMs used for PAN) rather than a robust signal for AI-generated text. DeBERTa-v3 is state-of-the-art for sequence classification and uses disentangled attention and Electra-style pre-training, making it a strong candidate for robust detection.

## Method

### Approach

We will fine-tune `microsoft/deberta-v3-base` on the PAN2025 training data. Unlike previous experiments, we will minimize text cleaning to allow the model to leverage punctuation and casing patterns learned during pre-training.

### Setup

| Component | Details |
|-----------|---------|
| Data | PAN2025 (train/val), HC3 Wiki (external test) |
| Compute | GPU (NVIDIA A100 or similar suggested) |
| Dependencies | `transformers`, `torch`, `datasets`, `accelerate` |
| Code | `harshul/src/003-deberta/` |

### Procedure

1. **Data Prep**: Tokenize PAN2025 data using the DeBERTa-v3 tokenizer (512 max length).
2. **Fine-tuning**: Train using the HuggingFace `Trainer` API with standard hyperparameters (Learning rate: 2e-5, Batch size: 16-32, Epochs: 3).
3. **Internal Evaluation**: Compute PAN2025 metrics on the validation set.
4. **External Evaluation**: Run inference on the HC3 dataset to test OOD generalization.
5. **Robustness Check**: If overfitting occurs, experiment with weight decay or freezing early layers.

### Variables

- **Independent**: Model architecture (DeBERTa-v3-base Transformer).
- **Dependent**: ROC-AUC, Brier score, C@1, F1, F0.5u (PAN & HC3).
- **Controlled**: Dataset splits, max sequence length (512).

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| ROC-AUC | Primary metric for ranking models across both datasets. |
| Brier Score | Measures probability calibration quality. |
| C@1 | PAN-specific metric evaluating decision accuracy. |
| F1 / F0.5u | Balanced and precision-weighted classification scores. |

### Baseline

We compare against Experiment 001 (Logistic Regression) as the most resilient baseline so far:
- **Baseline OOD ROC-AUC (HC3)**: 0.5520
- **Baseline OOD F1 (HC3)**: 0.6079

### Success Criteria

- **Confirm if**: The OOD (external) ROC-AUC is **≥ 0.70**.
- **Reject if**: The OOD ROC-AUC remains **< 0.60**, suggesting the model is still primarily learning domain-specific artifacts.

## Limitations

- **Truncation**: Texts longer than 512 tokens will be truncated, potentially losing evidence.
- **Compute**: Training time will be orders of magnitude longer than baseline models.
- **Brittleness**: Transformers can sometimes be sensitive to prompt/domain changes in subtle ways.