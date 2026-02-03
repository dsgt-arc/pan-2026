# PAN 2026: AI-Generated Text Detection

This repository contains our team's work on the [PAN 2026 Shared Task](https://pan.webis.de/) for detecting AI-generated text. The goal is to build robust classifiers that can distinguish between human-written and machine-generated text, with a focus on **out-of-distribution generalization**.

## Task Overview

The PAN 2026 AI Detection task challenges participants to:
1. **Classify text** as either human-written (0) or AI-generated (1)
2. **Generalize across domains** - models should work on unseen text generators and topics
3. **Handle long documents** - some texts exceed 6,500 tokens

### Key Metrics
- **ROC-AUC**: Primary ranking metric
- **Brier Score**: Probability calibration
- **F1, C@1, F0.5u**: Additional evaluation metrics

## Current Progress

| Experiment | Model | PAN2025 Val ROC-AUC | HC3 (OOD) ROC-AUC | Status |
|------------|-------|---------------------|-------------------|--------|
| 001 | Logistic Regression + TF-IDF | 0.9953 | 0.5520 | Complete |
| 002 | LightGBM + Stylometric Features | 0.9980 | 0.3356 | Complete |
| 003 | DeBERTa-v3 Fine-tuning | 0.9948 | **0.5939** | Complete |

### Key Findings

1. **In-distribution performance is high** - All models achieve >99% ROC-AUC on PAN2026 validation
2. **OOD generalization is the challenge** - Performance drops significantly on HC3 external dataset
3. **DeBERTa provides modest improvement** - +4.2% OOD ROC-AUC vs LR baseline, but still far from target
4. **Stylometric features overfit** - LightGBM with stylometrics performed *worse than random* on OOD data

## Repository Structure

```
pan-2026/
├── README.md                 # This file
├── pyproject.toml            # Package configuration
├── requirements.txt          # Dependencies
│
├── harshul/                  # User workspace
│   ├── src/                  # Python source code
│   │   ├── 001-logistic_regression/
│   │   ├── 002-lightgbm/
│   │   ├── 003-deberta/      # DeBERTa model code
│   │   │   ├── load_model.py     # Model loading with HF + local cache
│   │   │   ├── evaluate.py       # PAN2026 validation evaluation
│   │   │   ├── evaluate_external.py  # HC3 OOD evaluation
│   │   │   ├── chunking.py       # Sliding window for long docs
│   │   │   └── metrics.py        # PAN metrics implementation
│   │   └── pan-data/         # Dataset files (gitignored)
│   │
│   ├── experiments/          # Experiment tracking
│   │   ├── 000-template/     # Template for new experiments
│   │   ├── 001-logistic-regression/
│   │   ├── 002-lightgbm/
│   │   └── 003-deberta/
│   │       ├── proposal.md   # Hypothesis and approach
│   │       ├── tasks.md      # Implementation checklist
│   │       ├── results.md    # Final results and analysis
│   │       └── artifacts/    # Model weights, metrics JSON
│   │
│   └── notebooks/            # Jupyter notebooks
│       └── 003-deberta/      # Training notebook for Colab
│
├── scripts/                  # Utility scripts
└── docs/                     # Documentation
```

