# Implementation Plan: LightGBM Classifier with Stylometric Features

## Summary

This experiment builds on the Logistic Regression baseline (001) by training a LightGBM classifier that combines:
1. **TF-IDF features** (reused from experiment 001)
2. **Stylometric features** (new hand-crafted linguistic statistics)

The goal is to improve out-of-distribution generalization (measured on HC3) while maintaining strong in-distribution performance (PAN2025 validation).

All code will be placed in `harshul/src/002-lightgbm/`.

---

## Detailed Tasks

### Phase 1: Project Setup

- [x] Create the directory `harshul/src/002-lightgbm/`.
- [x] Create `__init__.py` to make it a Python package.
- [x] Add `lightgbm` to `requirements.txt`.
- [x] Copy/import reusable modules from experiment 001 (`data_loader.py`, `preprocessing.py`, `metrics.py`).

### Phase 2: Stylometric Feature Engineering

- [x] Create `stylometric_features.py` with a function `extract_stylometric_features(text: str) -> dict`.
- [x] Implement the following features:
  - [x] **Word count**: `len(text.split())`
  - [x] **Character count**: `len(text)`
  - [x] **Sentence count**: Count of `.`, `!`, `?` terminators.
  - [x] **Average word length**: Total characters / word count.
  - [x] **Flesch reading ease**: Use `textstat` library or manual formula.
  - [x] **Digit count**: Count of numeric digits (`0-9`).
  - [x] **Uppercase word count**: Words that are fully capitalized.
  - [x] **Longest sentence length**: Max word count among all sentences.
  - [x] **Repeated token ratio**: Fraction of words appearing more than once.
  - [x] **Punctuation ending ratio**: Sentences ending with `.`, `!`, or `?` vs. total sentences.
- [x] Create `get_stylometric_df(df)` to apply feature extraction to a DataFrame.
- [x] Add unit tests for edge cases (empty text, single word, etc.).

### Phase 3: Feature Combination

- [x] Create `features.py` (or extend from 001) with a function to combine TF-IDF and stylometric features.
- [x] Use `scipy.sparse.hstack` to concatenate sparse TF-IDF matrix with dense stylometric features.
- [x] Ensure feature names are tracked for importance analysis.

### Phase 4: Model Training

- [x] Create `train.py` as the main training script.
- [x] Load data using `data_loader.py`.
- [x] Extract combined features (TF-IDF + stylometric).
- [x] Instantiate `LGBMClassifier` with:
  - `n_estimators=100`
  - `num_leaves=31` (default)
  - `learning_rate=0.1`
  - `random_state=42`
- [x] Fit the model on the training features.
- [x] Save the trained model using `joblib`.

### Phase 5: Evaluation

- [x] Create `evaluate.py` to evaluate on PAN2025 validation set.
- [x] Create or extend `evaluate_external.py` to evaluate on HC3.
- [x] Compute all metrics: ROC-AUC, Brier, C@1, F1, F0.5u.
- [x] Save metrics to `artifacts/` directory.

### Phase 6: Feature Importance Analysis

- [x] Extract feature importances from the trained LightGBM model.
- [x] Separate and rank TF-IDF features vs. stylometric features.
- [x] Save top 20 features to a JSON/CSV file.
- [x] (Optional) Create a visualization (bar chart) of feature importances.

### Phase 7: Results Documentation

- [x] Update `harshul/experiments/002-lightgbm/results.md` with final metrics.
- [x] Compare results to experiment 001 (Logistic Regression baseline).
- [x] Document observations about which feature types contributed most.
- [x] Update `harshul/README.md` experiments table.

### Phase 8: Hyperparameter Tuning

- [ ] Create `tune_hyperparams.py` for systematic hyperparameter search.
- [ ] Define search space for LightGBM parameters:
  - `n_estimators`: [50, 100, 200, 300]
  - `num_leaves`: [15, 31, 63, 127]
  - `learning_rate`: [0.01, 0.05, 0.1, 0.2]
  - `max_depth`: [-1, 5, 10, 15]
  - `min_child_samples`: [5, 10, 20, 50]
  - `reg_alpha` (L1): [0, 0.1, 1.0]
  - `reg_lambda` (L2): [0, 0.1, 1.0]
- [ ] Use `RandomizedSearchCV` or `Optuna` for efficient search.
- [ ] Use 5-fold cross-validation on training set.
- [ ] Optimize for ROC-AUC on validation set.
- [ ] Log all trials to `artifacts/hyperparameter_search.csv`.
- [ ] Save the best hyperparameters to `artifacts/best_params.json`.
- [ ] Retrain the final model with the best hyperparameters.
- [ ] Re-evaluate on both PAN2025 validation and HC3 external datasets.

### Phase 9: Feature Selection Optimization

- [ ] Create `tune_features.py` for feature selection experiments.
- [ ] Experiment with different TF-IDF configurations:
  - `max_features`: [5000, 10000, 20000, 50000]
  - `ngram_range`: [(1, 1), (1, 2), (1, 3)]
  - `min_df` / `max_df` thresholds.
- [ ] Experiment with stylometric feature subsets:
  - [ ] Test model with only TF-IDF features (no stylometric).
  - [ ] Test model with only stylometric features (no TF-IDF).
  - [ ] Test with top-k stylometric features based on importance.
- [ ] Use Recursive Feature Elimination (RFE) or permutation importance to prune low-value features.
- [ ] Log feature ablation results to `artifacts/feature_ablation.csv`.
- [ ] Identify the optimal feature set that maximizes external (HC3) performance.
- [ ] Retrain final model with the optimized feature set.
- [ ] Document findings in `results.md`.

### Phase 10: Final Model and Documentation

- [ ] Combine best hyperparameters and feature set from Phases 8-9.
- [ ] Train the final optimized model.
- [ ] Evaluate on both PAN2025 validation and HC3 external datasets.
- [ ] Update `results.md` with final optimized metrics.
- [ ] Compare to initial LightGBM and LR baseline results.
- [ ] Document lessons learned and recommendations for next experiments.

---

## File Structure

```
harshul/src/002-lightgbm/
├── __init__.py
├── stylometric_features.py  # Stylometric feature extraction
├── features.py              # Combine TF-IDF + stylometric
├── train.py                 # Main training script
├── evaluate.py              # PAN2025 validation evaluation
├── evaluate_external.py     # HC3 external evaluation
├── importance_analysis.py   # Feature importance analysis
├── tune_hyperparams.py      # Hyperparameter tuning
└── tune_features.py         # Feature selection optimization
```

---

## Dependencies

Ensure the following are installed:
- `lightgbm`
- `scikit-learn`
- `pandas`
- `numpy`
- `textstat` (for Flesch reading ease)
- `joblib`

---

## Success Criteria

| Metric | LR Baseline (HC3) | Target (LightGBM) |
|--------|-------------------|-------------------|
| ROC-AUC | 0.5520 | ≥ 0.65 |
| F1 | 0.6079 | ≥ 0.70 |

---

## Next Steps

1. ~~Complete Phase 1-7 (core implementation).~~ ✅
2. Execute Phase 8 (hyperparameter tuning) to find optimal LightGBM settings.
3. Execute Phase 9 (feature selection) to identify features that generalize well.
4. Combine findings in Phase 10 for the final optimized model.
5. Update `results.md` with optimized metrics and analysis.
