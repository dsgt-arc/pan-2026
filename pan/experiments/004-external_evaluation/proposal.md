# Proposal: External Evaluation Benchmark

**Status**: proposed
**Created**: 2026-02-02
**Author**: harshul

## Hypothesis

Models trained on the PAN2025 dataset will exhibit varying degrees of out-of-distribution (OOD) generalization when evaluated on external datasets, with transformer-based models (DeBERTa) outperforming traditional ML baselines (Logistic Regression, LightGBM) on diverse AI-generated text sources.

## Background

Previous experiments (001-003) showed concerning results:
- All models achieve >99% ROC-AUC on PAN2025 validation (in-distribution)
- OOD performance on HC3 drops dramatically (LR: 0.55, LightGBM: 0.34, DeBERTa: 0.59)
- Current evaluation is limited to a single external dataset (HC3 Wiki)

A comprehensive external evaluation across multiple datasets and AI generators is needed to:
1. Understand true generalization capabilities
2. Identify which model types are most robust
3. Guide future model development toward real-world deployment

## Method

### Approach

Systematically evaluate all three trained models (LR, LightGBM, DeBERTa) on multiple external datasets spanning different:
- AI generators (ChatGPT, GPT-4, Claude, LLaMA, etc.)
- Domains (Wikipedia, Reddit, academic writing, news)
- Text lengths (short, medium, long)

### Setup

| Component | Details |
|-----------|---------|
| Models | 001-LR, 002-LightGBM, 003-DeBERTa |
| External Data | HC3-Wiki, HC3-Reddit, M4, RAID, TuringBench (to be sourced) |
| Compute | CPU (local Mac) |
| Dependencies | scikit-learn, lightgbm, transformers, torch |
| Code | `src/004-external_evaluation/` |

### Procedure

1. Source and preprocess external datasets (HC3, M4, RAID, etc.)
2. Load each trained model from artifacts/HuggingFace
3. Run inference on all external datasets
4. Compute standardized metrics for each model-dataset pair
5. Generate comparison tables and visualizations
6. Analyze patterns in OOD performance

### Variables

- **Independent**: Model type, external dataset
- **Dependent**: ROC-AUC, Brier, F1, C@1, F0.5u
- **Controlled**: Inference settings, chunking strategy, threshold (0.5)

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| ROC-AUC | Primary ranking metric (threshold-independent) |
| Brier Score | Probability calibration quality |
| F1 Score | Balanced precision/recall at 0.5 threshold |
| C@1 | PAN metric accounting for abstention |
| F0.5u | PAN metric penalizing false negatives |

### Baseline

| Model | PAN2025 Val | HC3 Wiki (current) |
|-------|-------------|-------------------|
| Logistic Regression | 0.9953 | 0.5520 |
| LightGBM | 0.9980 | 0.3356 |
| DeBERTa | 0.9948 | 0.5939 |

### Success Criteria

- **Useful if**: We establish a comprehensive benchmark with ≥3 external datasets
- **Notable if**: DeBERTa consistently outperforms baselines across all OOD datasets
- **Concerning if**: All models perform near-random (0.5 ROC-AUC) on all OOD data

## Limitations

- External dataset availability may be limited
- Some datasets may have different label definitions (human/AI)
- Cannot evaluate on unseen AI models released after dataset creation
- Computational cost of running DeBERTa on large datasets (CPU-only)