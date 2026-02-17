# Proposal: Ensemble Baseline

**Status**: in-progress
**Created**: 2026-02-16
**Author**: harshul

## Hypothesis

An ensemble of the three best-performing models (LR, DeBERTa, Qwen LoRA) will achieve higher OOD generalization than any individual model, while maintaining strong in-distribution performance. Specifically, a weighted ensemble should achieve ROC-AUC > 0.92 average OOD (vs. Qwen LoRA's 0.91).

## Background

Experiment 006 established Qwen LoRA as the clear leader with avg OOD ROC-AUC of 0.91. However, each model captures different signals:
- **LR (001)**: TF-IDF n-gram patterns, well-calibrated probabilities
- **DeBERTa (003)**: Contextual embeddings from 86M-param transformer
- **Qwen LoRA (006)**: 1.5B-param LLM with LoRA, strongest OOD generalization

Ensembling diverse models is a well-established technique for improving robustness. Even if the Qwen model dominates, LR/DeBERTa may contribute complementary signal on edge cases.

Reference results from [experiment 006](file:///Users/tanish/Desktop/CLEF2026/pan-2026/pan/experiments/006-lora-baseline/results.md):

| Model | PAN2025 | HC3 | RAID | Avg OOD |
|-------|---------|-----|------|---------|
| LR (001) | 0.9953 | 0.5517 | 0.5727 | 0.5622 |
| DeBERTa (003) | 0.9948 | 0.5939 | 0.5879 | 0.5909 |
| Qwen LoRA (006) | 0.9999 | 0.9982 | 0.8152 | 0.9067 |

## Method

### Approach

Combine predictions from all three models using multiple ensemble strategies (mean, weighted mean, majority vote) and evaluate on PAN2025 validation + HC3 + RAID.

### Setup

| Component | Details |
|-----------|---------|
| Data | PAN2025 val (3,589), HC3 Wiki (1,684), RAID (2,000) |
| Compute | GPU required for Qwen LoRA inference |
| Dependencies | transformers, peft, torch, scikit-learn, joblib |
| Code | `notebooks/007-ensemble-baseline/ensemble_evaluation.ipynb` |

### Procedure

1. Load all 3 models (LR, DeBERTa, Qwen LoRA)
2. Generate per-model probability predictions on each dataset
3. Combine predictions using 3 ensemble strategies
4. Compute all 5 standard metrics for each strategy
5. Compare against individual model baselines

### Variables

- **Independent**: Ensemble strategy (mean, weighted, majority vote)
- **Dependent**: ROC-AUC, Brier, F1, C@1, F0.5u
- **Controlled**: Same datasets, same individual model predictions

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| ROC-AUC | Ranking ability (threshold-independent) |
| Brier Score | Probability calibration |
| F1 Score | Balanced precision/recall |
| C@1 | PAN metric with abstention |
| F0.5u | PAN metric penalizing false negatives |

### Baseline

Individual model results from experiment 006 (see table above).

### Success Criteria

- **Confirm if**: Any ensemble strategy achieves avg OOD ROC-AUC > 0.91 (Qwen LoRA solo)
- **Reject if**: All ensemble strategies perform worse than Qwen LoRA alone

## Limitations

- Ensemble is dominated by Qwen LoRA (much stronger than LR/DeBERTa on OOD)
- Only 3 OOD datasets evaluated (TuringBench, M4, Ghostbuster excluded)
- No learned ensemble weights (future work: stacking with held-out set)