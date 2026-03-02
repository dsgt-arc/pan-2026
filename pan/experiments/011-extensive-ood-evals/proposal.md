# Proposal: Extensive Out-of-Distribution Evaluation

**Status**: completed
**Created**: 2026-03-02
**Author**: harshul

## Hypothesis

The Qwen2.5-1.5B LoRA model trained with obfuscation augmentation (experiment 010) maintains strong generalization across a diverse set of 7+ out-of-distribution datasets beyond the HC3/RAID pair used so far. Specifically:

- **Avg ROC-AUC ≥ 0.90** across all 7 OOD datasets
- **No dataset with ROC-AUC < 0.70** (floor performance)
- **Calibration**: Avg Brier Score ≤ 0.10 across OOD datasets

## Background

Experiments 006 through 010 have evaluated the LoRA model on only 2 OOD datasets (HC3 Wiki and RAID), both showing near-perfect performance (>0.99 ROC-AUC). However, this narrow OOD evaluation creates blind spots:

1. **Domain diversity**: HC3 is Q&A-focused, RAID is multi-domain but limited. We haven't tested on news, essays, Wikipedia, creative writing, or scientific domains.
2. **Generator diversity**: Both HC3 (ChatGPT) and RAID (multi-generator) represent a limited set of generative models. Datasets like MAGE (27 LLMs) and TuringBench (GPT-1/2/3, GROVER, etc.) cover a wider range.
3. **Attack robustness**: RAID contains some adversarial perturbations, but we haven't explicitly benchmarked on adversarial-only subsets.
4. **Experiment 004** used synthetic stand-ins (resampled from HC3) for TuringBench, M4, and GhostBuster — not real data.

This experiment provides a comprehensive robustness profile before submission.

**References**:
- [010 Obfuscation Results](../010-obfuscation/results.md)
- [004 External Evaluation](../004-external_evaluation/results.md)

## Method

### Approach

Evaluate the experiment 010 adapter (`hersheys-baklava/qwen-lora-pan2026-010-obfuscation`) on 7 properly sourced OOD datasets in a self-contained Google Colab notebook. All datasets are loaded directly from HuggingFace. No local data dependencies.

### Setup

| Component | Details |
|-----------|---------|
| **Model** | `hersheys-baklava/qwen-lora-pan2026-010-obfuscation` (LoRA adapter on Qwen2.5-1.5B) |
| **Compute** | Google Colab T4 GPU |
| **Dependencies** | transformers, peft, datasets, scikit-learn, numpy, pandas, matplotlib |
| **Code** | Self-contained Colab notebook + `src/011-extensive-ood-evals/` |

### OOD Datasets

| # | Dataset | HuggingFace ID | Domain | Generators | Samples |
|---|---------|----------------|--------|------------|---------|
| 1 | MAGE | `yaful/MAGE` | Multi-domain (news, review, essay, etc.) | 27 LLMs | ~1000 |
| 2 | OpenGPTText | `aadityaubhat/GPT-wiki-intro` | Wikipedia intros | GPT-3.5 | ~1000 |
| 3 | TuringBench | `turingbench/TuringBench` | News articles | GPT-1/2/3, GROVER, etc. | ~1000 |
| 4 | AI Detection Pile | `artem9k/ai-text-detection-pile` | Long-form essays | GPT-2/3/ChatGPT/GPTJ | ~1000 |
| 5 | M4 (SemEval 2024) | `mbzuai-nlp/SemEval2024-task8` | Multi-domain | Multi-generator | ~1000 |
| 6 | HC3 Multi-Domain | `Hello-SimpleAI/HC3` (reddit_eli5, finance, medicine) | Q&A (non-wiki) | ChatGPT | ~1000 |
| 7 | RAID Adversarial | `liamdugan/raid` (adversarial subset) | Multi-domain | Multi-generator + attacks | ~1000 |

### Procedure

1. Install dependencies and load the 010 adapter from HuggingFace
2. Download and preprocess all 7 OOD datasets
3. Standardize formats to `{text, label}` (0=human, 1=AI)
4. Run chunked inference on each dataset
5. Compute 5 PAN metrics per dataset
6. Generate comparison visualizations
7. Save all results to artifacts

### Variables

- **Independent**: OOD dataset (7 levels)
- **Dependent**: ROC-AUC, Brier Score, F1, C@1, F0.5u
- **Controlled**: Model (010 adapter), inference pipeline, chunking strategy

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| ROC-AUC | Primary ranking metric |
| Brier Score | Probability calibration |
| F1 Score | Balanced precision/recall at 0.5 |
| C@1 | PAN metric with abstention |
| F0.5u | PAN metric penalizing false negatives |

### Baseline (from 010)

| Dataset | ROC-AUC |
|---------|---------|
| PAN2025 Val | 0.9995 |
| HC3 Wiki | 0.9985 |
| RAID | 0.9964 |

### Success Criteria

- **Confirm if**: Avg OOD ROC-AUC ≥ 0.90 AND no dataset ROC-AUC < 0.70
- **Reject if**: Avg OOD ROC-AUC < 0.80 OR any dataset ROC-AUC < 0.50

## Limitations

- Sample sizes capped at ~1000 per dataset for Colab runtime feasibility
- Some datasets (TuringBench, AI Detection Pile) include older generators (GPT-2) that may be easier to detect
- MAGE label convention is inverted (0=machine, 1=human) — must be mapped correctly
- M4/SemEval dataset availability depends on HuggingFace access
- Colab T4 GPU memory constrains batch size and sequence length