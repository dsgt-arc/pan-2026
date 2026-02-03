# Proposal: Logistic Regression Baseline for AI Detection

**Status**: completed
**Created**: 2026-01-27
**Author**: harshul

## Hypothesis

A simple logistic regression model using TF-IDF word features will provide a robust and computationally efficient baseline for classifying text as human or AI-generated. We expect this model to achieve significantly better than random performance, serving as a primary benchmark for more complex architectures.

## Background

Establishing a strong, simple baseline is a critical first step in any machine learning project. Logistic regression is a well-understood, linear model that often performs surprisingly well on text classification tasks when paired with TF-IDF features. By setting this baseline, we can quantify the performance gains (or lack thereof) provided by more complex methods like pretrained Transformers or ensemble techniques.

## Method

### Approach

We will train a binary logistic regression classifier. The input text will be converted into a bag-of-words representation using TF-IDF (Term Frequency-Inverse Document Frequency) to capture the importance of specific words while normalizing for document length and commonality across the corpus.

### Setup

| Component | Details |
|-----------|---------|
| Data | PAN2025 dataset. |
| Compute | Local CPU or single-node CPU instance (minimal compute required). |
| Dependencies | `scikit-learn`, `pandas`, `numpy`, `nltk` |
| Code | `harshul/src/001-logistic_regression`|

### Procedure

1. **Data Preprocessing**: Clean text by removing special characters, lowercasing, and tokenization.
2. **Feature Extraction**: Apply `TfidfVectorizer` to the training set and transform the test set.
3. **Model Training**: Fit a `LogisticRegression` model with default hyperparameters (C=1.0, L2 penalty).
4. **Hyperparameter Tuning**: (Optional) Perform a grid search over `C` values if time permits.
5. **Evaluation**: Generate a classification report and confusion matrix on the test set.

### Variables

- **Independent**: TF-IDF token features (unigrams/bigrams), regularization strength `C`.
- **Dependent**: ROC-AUC, Brier, C@1, F1, F0.5u.
- **Controlled**: Dataset split, preprocessing pipeline, max number of features for TF-IDF.

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| ROC-AUC | Area Under the Receiver Operating Characteristic Curve; measures the model's ability to distinguish between human and AI classes across all thresholds. |
| Brier Score | The mean squared difference between predicted probabilities and the actual outcome; measures the calibration of the classifier. |
| C@1 | A metric that rewards correct answers and penalizes incorrect ones, while allowing for "unsure" predictions (though LR will likely be forced to decide). |
| F1 | The harmonic mean of Precision and Recall, providing a balance between the two. |
| F0.5u | A variant of the F0.5 score (which weights precision higher than recall) designed for tasks with potentially unanswered cases. |

### Baseline

This experiment *is* the initial baseline using the PAN2025 dataset. The performance of a random classifier or majority class will serve as the trivial comparison point.

### Success Criteria

- **Confirm if**: The model achieves an ROC-AUC and F1-score significantly better than random guessing (e.g., ROC-AUC > 0.55).
- **Reject if**: The model's performance is indistinguishable from a random classifier or a simple majority-class baseline.

## Limitations

- **Linearity**: Cannot capture complex, non-linear linguistic patterns that LLMs might exhibit.
- **Context Loss**: TF-IDF/Bag-of-words ignores word order and syntactic structure.
- **Out-of-Vocabulary**: Simple tokenization might struggle with new terms or highly specific jargon not seen in training.