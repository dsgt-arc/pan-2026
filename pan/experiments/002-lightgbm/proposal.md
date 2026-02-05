# Proposal: LightGBM Classifier for AI Text Detection

**Status**: proposed
**Created**: 2026-01-27
**Author**: harshul

## Hypothesis

A gradient-boosted tree model (LightGBM) will outperform the simple Logistic Regression baseline for AI text detection by capturing non-linear feature interactions, while maintaining computational efficiency suitable for rapid iteration.

## Background

Our previous experiment ([001-logistic-regression](../001-logistic-regression/results.md)) established a TF-IDF + Logistic Regression baseline achieving **0.995 ROC-AUC** on PAN2025 validation data, but only **0.552 ROC-AUC** on the external HC3 dataset. This significant generalization gap suggests the linear model is overfitting to dataset-specific artifacts.

**Why LightGBM?**
- **Non-linear decision boundaries**: Can capture complex feature interactions that linear models miss.
- **Feature importance**: Built-in tools to identify which features matter most.
- **Speed**: Highly optimized for large datasets; faster than most tree-based methods.
- **Regularization**: Native support for L1/L2 regularization and early stopping to reduce overfitting.

This experiment tests whether a more expressive model architecture improves generalization without sacrificing interpretability.

## Method

### Approach

Train a LightGBM binary classifier using a **hybrid feature set**:
1. **TF-IDF features**: Sparse bag-of-words representation (reused from experiment 001).
2. **Stylometric features**: Hand-crafted linguistic statistics that capture writing style differences between humans and AI.

This combined approach should allow the model to learn both lexical patterns (from TF-IDF) and structural/stylistic patterns (from stylometric features).

### Stylometric Features

| Feature | Description |
|---------|-------------|
| Word count | Total number of words in the text. |
| Character count | Total number of characters (including spaces). |
| Sentence count | Number of sentences (split by `.`, `!`, `?`). |
| Average word length | Mean number of characters per word. |
| Flesch reading ease | Readability score (higher = easier to read). |
| Digit count | Number of numeric digits in the text. |
| Uppercase word count | Words that are fully capitalized. |
| Longest sentence length | Word count of the longest sentence. |
| Repeated token ratio | Fraction of words that appear more than once. |
| Punctuation ending ratio | Fraction of sentences ending with `.`, `!`, or `?`. |

### Setup

| Component | Details |
|-----------|---------|
| Data | PAN2025 train/val splits + HC3 Wiki (external test) |
| Compute | Local CPU (LightGBM is CPU-optimized) |
| Dependencies | `lightgbm`, `scikit-learn`, `pandas`, `numpy` |
| Code | `harshul/src/002-lightgbm/` |

### Procedure

1. **Feature engineering**: Implement `stylometric_features.py` to compute all stylometric features.
2. **Combine features**: Concatenate TF-IDF sparse matrix with dense stylometric feature vectors.
3. **Train LightGBM**: Fit a `LGBMClassifier` with default hyperparameters initially.
4. **Evaluate**: Compute PAN2025 metrics on both PAN2025 validation and HC3 external datasets.
5. **Analyze features**: Extract and visualize feature importance (both TF-IDF and stylometric).
6. **(Optional) Tune hyperparameters**: Grid search over `num_leaves`, `learning_rate`, and `n_estimators`.

### Variables

- **Independent**: Model architecture (LightGBM vs. Logistic Regression), hyperparameters.
- **Dependent**: ROC-AUC, Brier, C@1, F1, F0.5u on both in-distribution and out-of-distribution data.
- **Controlled**: TF-IDF feature extraction pipeline, dataset splits, preprocessing steps.

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| ROC-AUC | Measures discriminative ability across all thresholds. |
| Brier Score | Measures probability calibration (lower is better). |
| F1 | Balanced precision/recall (macro-averaged). |
| C@1 | PAN metric rewarding correct predictions and penalizing incorrect ones. |
| F0.5u | Precision-weighted F-score for unanswered cases. |

### Baseline

We compare against [001-logistic-regression](../001-logistic-regression/results.md):

| Metric | LR (PAN2025 Val) | LR (HC3 External) |
|--------|-----------------|-------------------|
| ROC-AUC | 0.9953 | 0.5520 |
| F1 | 0.9772 | 0.6079 |

### Success Criteria

- **Confirm if**: LightGBM achieves **ROC-AUC ≥ 0.65** on HC3 (meaningful improvement over LR's 0.55) while maintaining **ROC-AUC ≥ 0.95** on PAN2025 validation.
- **Reject if**: LightGBM shows no improvement on HC3, or significantly degrades PAN2025 performance.

## Limitations

- **Still uses TF-IDF**: The feature representation remains bag-of-words, which discards word order and context.
- **No semantic understanding**: Unlike Transformer-based models, LightGBM cannot learn contextual embeddings.
- **Potential overfitting**: More model capacity may lead to more severe overfitting if not regularized properly.