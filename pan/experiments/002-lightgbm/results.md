# Results: LightGBM Classifier with Stylometric Features

**Date Completed**: 2026-01-27
**Author**: harshul

## Summary

The hypothesis was partially confirmed in-distribution but failed significantly out-of-distribution: LightGBM with stylometric features improved PAN2025 metrics but showed severe degradation on the external HC3 dataset.

**Verdict**: rejected (failed to improve generalization)

## Observations

### Quantitative

| Experiment | PAN2025 Val ROC-AUC | HC3 External ROC-AUC |
|------------|---------------------|----------------------|
| LR Baseline (001) | 0.9953 | 0.5520 |
| LightGBM (Default, Combined) | 0.9964 | 0.3356 |
| LightGBM (Tuned, Combined) | **0.9980** | 0.3356 |
| TF-IDF Only (10k) | 0.9954 | **0.4775** |
| Stylo Only | 0.9635 | 0.2814 |
| Combined (5k + Stylo) | 0.9965 | 0.3597 |

### Qualitative

- **Stylometric Importance**: Stylometric features were massive predictors. `avg_word_length` and `longest_sentence_length` were the top 2 features by gain.
- **Overfitting**: The model seems to have learned dataset-specific "meta-features" that are completely inverted in the external HC3 dataset, leading to an ROC-AUC below 0.5.
- **Complexity Cost**: Adding more expressive power (LightGBM) and specific heuristics (Stylometrics) made the model more brittle despite the high in-distribution scores.

## Analysis

1. **Stylometric Overfitting**: The addition of hand-crafted stylometric features (word length, sentence complexity, etc.) significantly improved in-distribution performance (up to **0.998 ROC-AUC**) but caused a massive drop in external generalization. "Stylo Only" produced the worst generalization (0.28 ROC-AUC), suggesting these features captured PAN-specific artifacts rather than universal AI signatures.
2. **Hyperparameter Impact**: Tuning improved the PAN2025 validation score to near-perfection (0.998) but did not solve the generalization gap.
3. **Feature Resilience**: TF-IDF features alone generalized better than the combined model, though still poorly.
4. **The "Accent" Problem**: The model has essentially learned to detect the specific way AI was prompted or the specific LLMs used in the PAN dataset (e.g., preference for specific word lengths or punctuation patterns), which are inverted or absent in the HC3 dataset.

### Confounders

- **Feature Scaling**: Stylometric features were not scaled before being passed to LightGBM. While LightGBM handles scale differences better than linear models, this might have affected split choices.
- **Metric Definitions**: C@1 and F0.5u assume 0.5 prob is "unanswered", but LightGBM (like LR) rarely outputs exactly 0.5 without explicit calibration.

## Artifacts

| Type | Location |
|------|----------|
| Model | `./artifacts/model.joblib` |
| Top Features | `./artifacts/top_features.csv` |
| Metrics | `./artifacts/val_metrics.json` |

## Next Steps

- [ ] Investigate the distribution of `avg_word_length` in PAN2025 vs HC3 to confirm the inversion.
- [ ] Experiment with **Adversarial Training** or **Domain Adaptation** techniques to bridge the gap.
- [ ] Shift towards **Transformer-based models** (BERT/RoBERTa) which might learn more robust linguistic patterns than simple heuristics.

**Leads to**: Recognition that simple stylometric and bag-of-words features are too prone to domain-specific artifacts.