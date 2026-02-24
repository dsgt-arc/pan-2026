# Proposal: LoRA Replication (Clean Notebook Workflow)

**Status**: completed
**Created**: 2026-02-22
**Author**: tanish

## Hypothesis

Replication of experiment 006. A Qwen2.5-1.5B model fine-tuned with LoRA adapters for sequence classification achieves ROC-AUC > 0.98 on PAN2025 validation and ROC-AUC ≥ 0.60 on HC3, matching the results from experiment 006.

This experiment uses the exact same hyperparameters as 006 but restructures the code into a cleaner notebook-based workflow with reusable utility functions.

## Background

Experiment 006 demonstrated that LoRA fine-tuning of Qwen2.5-1.5B dramatically outperforms all prior models:
- PAN2025 Val ROC-AUC: 0.9999
- HC3 ROC-AUC: 0.9982
- RAID ROC-AUC: 0.8152

This replication restructures the code for readability and reuse. Training and evaluation each get a notebook that calls into shared Python modules in `src/009-lora-replication/`.

## Method

### Approach

Same as 006: fine-tune Qwen2.5-1.5B with LoRA adapters (rank 16, alpha 32) on PAN2025 training data. Evaluate on PAN2025 val, HC3, and RAID using chunked inference and all 5 PAN metrics.

### Setup

| Component | Details |
|-----------|---------|
| Data | PAN2025 train (23,707), PAN2025 val (3,589), HC3 Wiki, RAID (2,000) |
| Compute | Google Colab (GPU) |
| Dependencies | `peft`, `transformers`, `torch`, `datasets`, `scikit-learn` |
| Base Model | `Qwen/Qwen2.5-1.5B` |
| Code | `src/009-lora-replication/`, `notebooks/009-lora-replication/` |

### Procedure

1. Load and tokenize PAN2025 data (left-padded for causal LM)
2. Create Qwen2.5-1.5B + LoRA model for sequence classification
3. Train with HF Trainer (3 epochs, lr=2e-4, effective batch size 16)
4. Save adapter, push to HuggingFace
5. Evaluate on PAN2025 val, HC3, RAID with chunked inference
6. Compare metrics against 006 results

### Variables

- **Independent**: None (replication)
- **Dependent**: ROC-AUC, Brier Score, F1, C@1, F0.5u
- **Controlled**: All hyperparameters match 006

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| ROC-AUC | Primary ranking metric |
| Brier Score | Probability calibration |
| F1 Score | Balanced precision/recall at 0.5 |
| C@1 | PAN metric with abstention |
| F0.5u | PAN metric penalizing false negatives |

### Baseline (from 006)

| Dataset | ROC-AUC |
|---------|---------|
| PAN2025 Val | 0.9999 |
| HC3 Wiki | 0.9982 |
| RAID | 0.8152 |

### Success Criteria

- **Confirm if**: Results match 006 within ±0.01 ROC-AUC
- **Reject if**: Results differ by > 0.05 ROC-AUC

## Limitations

- Same limitations as 006 (single model, fixed rank, limited OOD datasets)
- Stochastic training means exact reproduction depends on random seeds