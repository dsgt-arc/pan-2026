# Proposal: LoRA Fine-Tuning Baseline for AI Text Detection

**Status**: in-progress
**Created**: 2026-02-11
**Author**: tanish

## Hypothesis

A small causal language model (`Qwen/Qwen2.5-1.5B`) fine-tuned with **LoRA (Low-Rank Adaptation)** for sequence classification can achieve competitive in-distribution performance (ROC-AUC > 0.98 on PAN2025 validation) while maintaining or improving out-of-distribution generalization (ROC-AUC ≥ 0.60 on HC3) compared to full fine-tuning of an encoder model (Experiment 003, HC3 ROC-AUC = 0.5939), with significantly fewer trainable parameters. Using a decoder-only LLM (rather than an encoder like DeBERTa) may capture richer distributional signals from the autoregressive pre-training objective, which is closely aligned with how AI-generated text is produced.

## Background

Previous experiments show:
- **Full fine-tuning (003-deberta)**: Near-perfect in-distribution (0.9948 ROC-AUC) but modest OOD generalization (0.5939 on HC3). All model parameters updated → risk of overfitting to training distribution.
- **Zero-shot (005-ollama)**: Below random chance (0.46 ROC-AUC) — zero-shot approaches are not viable.
- **Experiment 004**: DeBERTa leads OOD across 5 external datasets (avg 0.579) but still far from target.

**LoRA** freezes pre-trained weights and injects small trainable low-rank matrices into attention layers. This approach:
1. **Reduces overfitting risk** — fewer parameters updated → may preserve pre-trained generalization
2. **Efficient training** — ~0.1–1% of parameters trainable → faster training, lower memory
3. **Preserves pre-trained features** — frozen backbone retains broad language understanding
4. **Decoder-only architecture** — LLMs like Qwen are pre-trained autoregressively, giving them an inherent "model" of how text is generated; this inductive bias may help distinguish human vs. machine text
5. **Baseline for future work** — establishes whether parameter-efficient fine-tuning of an LLM improves OOD

## Method

### Approach

Fine-tune `Qwen/Qwen2.5-1.5B` using LoRA adapters on sequence classification. The model is loaded via `AutoModelForSequenceClassification`, which adds a classification head on top of the causal LM and uses the last non-padding token's representation for the prediction. Train on PAN2025 training data, evaluate on PAN2025 validation (in-distribution) and HC3 Wiki (out-of-distribution). Use the same evaluation pipeline and metrics as prior experiments for direct comparison.

### Setup

| Component | Details |
|-----------|---------|
| Data | PAN2025 train (23,707 samples), PAN2025 val (3,589), HC3 Wiki external |
| Compute | Local CPU/MPS (Mac); GPU via Colab for training |
| Dependencies | `peft`, `transformers`, `torch`, `datasets`, `scikit-learn`, `bitsandbytes` |
| Base Model | `Qwen/Qwen2.5-1.5B` (1.5B parameter causal LM) |
| Code | `src/006-lora-baseline/` |

### Procedure

1. **Setup**: Install `peft` library for LoRA support, `bitsandbytes` for efficient loading
2. **Data Preparation**: Load and tokenize PAN2025 train/val with left-padding (required for causal LM classification)
3. **Model Configuration**: Load Qwen2.5-1.5B for sequence classification with LoRA config (r=16, alpha=32, target q_proj/v_proj)
4. **Training**: Fine-tune with HuggingFace Trainer (3 epochs, lr=2e-4, batch_size=16 effective)
5. **In-Distribution Evaluation**: Evaluate on PAN2025 validation set with chunking for long docs
6. **OOD Evaluation**: Evaluate on HC3 Wiki dataset
7. **Comparison**: Compare all 5 standard metrics against experiments 001–005

### Variables

- **Independent**: Fine-tuning method (LoRA vs full fine-tuning), base architecture (causal LM vs encoder), LoRA rank (r), LoRA alpha
- **Dependent**: ROC-AUC, Brier Score, F1, C@1, F0.5u
- **Controlled**: Training data, evaluation data, max sequence length (512), chunking strategy

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| ROC-AUC | Primary ranking metric (threshold-independent) |
| Brier Score | Measures probability calibration quality |
| F1 Score | Balanced precision/recall at 0.5 threshold |
| C@1 | PAN metric accounting for abstention |
| F0.5u | PAN metric penalizing false negatives |

### Baseline

| Model | PAN2025 Val ROC-AUC | HC3 ROC-AUC |
|-------|---------------------|-------------|
| Logistic Regression (001) | 0.9953 | 0.5520 |
| LightGBM (002) | 0.9980 | 0.3356 |
| DeBERTa Full FT (003) | 0.9948 | 0.5939 |
| Ollama Zero-Shot (005) | 0.4606 | 0.4162 |

### Success Criteria

- **Confirm if**: LoRA achieves ROC-AUC ≥ 0.98 on PAN2025 val AND ROC-AUC ≥ 0.60 on HC3 (matching or beating full fine-tuning)
- **Promising if**: LoRA achieves comparable PAN2025 val (≥ 0.95) with improved HC3 (> 0.5939)
- **Reject if**: ROC-AUC < 0.90 on PAN2025 val OR ROC-AUC < 0.50 on HC3

## Limitations

- LoRA rank selection (r=16) is a single configuration; optimal rank may differ
- Training on CPU/MPS will be slow — Colab GPU recommended
- Same chunking strategy as 003 — long-document handling inherited
- Only testing one LLM (Qwen2.5-1.5B) — results may differ for Llama, Gemma, or larger models
- HC3 is only one OOD dataset — broader evaluation (004-style) deferred to follow-up
- Causal LM classification uses last-token pooling, which may behave differently from encoder [CLS] pooling