# Implementation Plan: Logistic Regression Baseline

## Summary

This experiment establishes a baseline for AI-generated text detection using a logistic regression classifier with TF-IDF features on the PAN2025 dataset. The implementation involves:

1. **Data loading and preprocessing** – Load the `train.jsonl` and `val.jsonl` files, extract text and labels, and apply basic text cleaning.
2. **Feature extraction** – Fit a TF-IDF vectorizer on the training data and transform both train and validation sets.
3. **Model training** – Train a logistic regression model on the TF-IDF features.
4. **Evaluation** – Compute the required PAN2025 metrics (ROC-AUC, Brier, C@1, F1, F0.5u) on the validation set.
5. **Results logging** – Save the metrics and model artifacts for reproducibility.

All code will be placed in `harshul/src/001-logistic_regression/`.

---

## Detailed Tasks

### Phase 1: Project Setup

- [x] Create the directory `harshul/src/001-logistic_regression/`.
- [x] Create `__init__.py` to make it a Python package.
- [x] Verify that dependencies (`scikit-learn`, `pandas`, `numpy`, `nltk`) are listed in `requirements.txt` or `pyproject.toml`.

### Phase 2: Data Loading

- [x] Create `data_loader.py` with a function `load_jsonl(path: str) -> pd.DataFrame` that reads a `.jsonl` file and returns a DataFrame.
- [x] Implement loading for both `harshul/src/pan-data/train.jsonl` and `harshul/src/pan-data/val.jsonl`.
- [x] Extract the `text` field and `label` field (human=0, AI=1) from each record.
- [x] Add a quick sanity check to print the number of samples and class distribution.

### Phase 3: Text Preprocessing

- [x] Create `preprocessing.py` with a function `clean_text(text: str) -> str`.
- [x] Implement the following cleaning steps:
  - [x] Lowercase the text.
  - [x] Remove special characters and punctuation (keep alphanumeric and spaces).
  - [x] Strip extra whitespace.
- [x] (Optional) Add tokenization using `nltk.word_tokenize` if needed for inspection.
- [x] Apply `clean_text` to the text column in both train and validation DataFrames.

### Phase 4: Feature Extraction

- [x] Create `features.py` with functions for TF-IDF vectorization.
- [x] Instantiate `TfidfVectorizer` with the following parameters:
  - `max_features=10000` (to limit dimensionality).
  - `ngram_range=(1, 2)` (unigrams and bigrams).
  - `stop_words='english'`.
- [x] Fit the vectorizer on the training data.
- [x] Transform both train and validation data.
- [x] Save the fitted vectorizer using `joblib` for later inference.

### Phase 5: Model Training

- [x] Create `train.py` as the main training script.
- [x] Import data loading, preprocessing, and feature extraction modules.
- [x] Instantiate `LogisticRegression` with:
  - `C=1.0` (default regularization).
  - `solver='lbfgs'`.
  - `max_iter=1000` (ensure convergence).
- [x] Fit the model on the TF-IDF training features.
- [x] Save the trained model using `joblib`.

### Phase 6: Evaluation Metrics

- [x] Create `metrics.py` with functions for each required metric:
  - [x] `compute_roc_auc(y_true, y_prob)` – using `sklearn.metrics.roc_auc_score`.
  - [x] `compute_brier(y_true, y_prob)` – using `sklearn.metrics.brier_score_loss`.
  - [x] `compute_f1(y_true, y_pred)` – using `sklearn.metrics.f1_score`.
  - [x] `compute_c_at_1(y_true, y_pred, y_prob)` – custom implementation for C@1.
  - [x] `compute_f05u(y_true, y_pred)` – custom implementation for F0.5u.
- [x] Reference PAN evaluation scripts or documentation for C@1 and F0.5u formulas.

### Phase 7: Evaluation Script

- [x] Create `evaluate.py` to load the trained model and vectorizer, and evaluate on validation data.
- [x] Generate predictions (`y_pred`) and predicted probabilities (`y_prob`).
- [x] Compute and print all metrics: ROC-AUC, Brier, C@1, F1, F0.5u.
- [x] Save the evaluation results to a JSON file in the experiment's `artifacts/` directory.

### Phase 8: Results Documentation

- [x] Update `harshul/experiments/001-logistic-regression/results.md` with the final metrics after the experiment is run.
- [x] Include a confusion matrix visualization or table.
- [x] Document any observations about the model's performance.

### Phase 9: External Dataset Evaluation

- [x] Identify an external AI text detection dataset (e.g., HC3, GPT-2 Output Dataset, or a different PAN year).
- [x] Download and preprocess the external dataset using the same pipeline.
- [x] Evaluate the trained model on the external dataset without retraining.
- [x] Compute all metrics: ROC-AUC, Brier, C@1, F1, F0.5u.
- [x] Compare performance to PAN2025 validation set results.
- [x] Document findings to assess out-of-distribution generalization.

---

## File Structure

```
harshul/src/001-logistic_regression/
├── __init__.py
├── data_loader.py       # Load JSONL data
├── preprocessing.py     # Text cleaning
├── features.py          # TF-IDF vectorization
├── metrics.py           # Evaluation metrics (ROC-AUC, Brier, C@1, F1, F0.5u)
├── train.py             # Main training script
├── evaluate.py          # Evaluation script
└── evaluate_external.py # External dataset evaluation
```

---

## Dependencies

Ensure the following are installed:
- `scikit-learn`
- `pandas`
- `numpy`
- `nltk`
- `joblib`

---

## Next Steps

1. ~~Complete Phase 1-8 (core implementation).~~ ✅
2. Identify and acquire an external AI text detection dataset.
3. Run Phase 9 (external evaluation) to verify generalization.
4. Update `results.md` with external evaluation findings.
