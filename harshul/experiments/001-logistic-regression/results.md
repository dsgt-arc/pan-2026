# Results: Logistic Regression Baseline for AI Detection

**Date Completed**: 2026-01-27
**Author**: harshul

## Summary

The hypothesis was confirmed: a simple logistic regression model with TF-IDF features provides a very strong baseline for AI text detection, achieving an ROC-AUC of 0.9953 on the validation set.

**Verdict**: confirmed

## Observations

### Quantitative

| Metric | Baseline (Random) | PAN2025 Val | HC3 Wiki (External) |
|--------|-------------------|-------------|----------------------|
| ROC-AUC | 0.5000 | 0.9953 | 0.5520 |
| Brier | 0.2500 | 0.0282 | 0.2706 |
| F1 | ~0.5000 | 0.9772 | 0.6079 |
| C@1 | ~0.5000 | 0.9705 | 0.5220 |
| F0.5u | ~0.5000 | 0.9732 | 0.5487 |

### Qualitative

- The model training was extremely fast, taking only a few seconds on a local CPU.
- TF-IDF with unigrams and bigrams seems highly effective at catching "fingerprints" of AI-generated text in this specific dataset.
- **CRITICAL**: The model shows **poor generalization** to external datasets. ROC-AUC dropped from 0.995 to 0.552 when tested on HC3 (Wiki), which is only slightly better than random.

## Analysis

The exceptional performance on the PAN2025 validation set (ROC-AUC > 0.99) contrasts sharply with the poor performance on HC3 (ROC-AUC ~0.55). This indicates:
1. **Model Overfitting to PAN2025**: The Logistic Regression model is picking up on specific artifacts or "accents" of the LLMs and prompts used for the PAN2025 dataset that do not exist in HC3.
2. **Dataset Specificity**: The stylistic markers captured (like specific bigrams) are highly sensitive to the distribution of the training data.
3. **Generalization Gap**: A simple linear baseline with TF-IDF is insufficient for robust, general-purpose AI text detection across different domains and generators.

### Confounders

- **Class Imbalance**: The training set had a 62/38 split (AI/Human), though the metrics (F1, ROC-AUC) still show strong performance across both classes.
- **Leakage**: Need to ensure no overlap between datasets, though they were loaded from separate files.

## Artifacts

| Type | Location |
|------|----------|
| Vectorizer | `./artifacts/vectorizer.joblib` |
| Model | `./artifacts/model.joblib` |
| Metrics | `./artifacts/val_metrics.json` |

## Next Steps

- [ ] Analyze the most important features (top coefficients) to understand what the model is looking at.
- [ ] Test the model's robustness on out-of-domain data or different AI models.
- [ ] Implement more complex architectures (e.g., Transformers) to see if the small remaining error can be eliminated.

**Leads to**: Potential deep dive into feature importance and investigating the "easy" nature of the task.