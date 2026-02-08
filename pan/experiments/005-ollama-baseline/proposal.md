# Proposal: Zero-Shot LLM Detection via Ollama

**Status**: completed
**Created**: 2026-02-05
**Author**: harshul

## Hypothesis

A locally-hosted LLM running via **Ollama** can perform zero-shot classification of AI-generated vs. human-written text **without any training or fine-tuning**. We hypothesize that modern LLMs have inherent knowledge about AI writing patterns and can achieve **ROC-AUC > 0.65** on both in-distribution (PAN2025 validation) and out-of-distribution (HC3) datasets through prompt-based inference alone.

## Background

Previous experiments (001-004) relied on supervised learning approaches that required training on labeled data:
- **In-distribution performance** is excellent (>99% ROC-AUC on PAN2025)
- **Out-of-distribution performance** degrades significantly (0.33-0.59 ROC-AUC on HC3)

Zero-shot LLM detection offers several advantages:
1. **No training required** — eliminates overfitting to dataset-specific artifacts
2. **Transferable knowledge** — LLMs may recognize AI writing patterns across domains
3. **Practical baseline** — establishes what's achievable with off-the-shelf models
4. **Interpretability** — prompts can be inspected and refined

This experiment explores whether the knowledge encoded in pre-trained LLMs during large-scale training enables them to distinguish AI-generated text from human writing without task-specific fine-tuning.

## Method

### Approach

We will load an open-source LLM (e.g., Llama 3, Mistral, or Qwen) via **Ollama** and prompt it to classify texts as "human" or "AI-generated". The model outputs a binary prediction (and optionally a confidence score) for each text sample. No training or parameter updates are performed—this is **pure zero-shot inference**.

### Setup

| Component | Details |
|-----------|---------|
| Data | PAN2025 validation set, HC3 Wiki external dataset |
| Compute | Local CPU/GPU via Ollama (Mac M-series or NVIDIA GPU) |
| Dependencies | `ollama`, `ollama-python`, `pandas`, `scikit-learn` |
| Models | Ollama-hosted LLMs (e.g., `llama3`, `mistral`, `qwen2`) |
| Code | `src/005-ollama-baseline/` |

### Procedure

1. **Environment Setup**: Install Ollama and download target model(s) locally.
2. **Prompt Engineering**: Design a zero-shot classification prompt that instructs the model to determine if a given text is human-written or AI-generated.
3. **Inference Pipeline**: Build a script to iterate through dataset samples, send each text to the LLM via Ollama API, and parse the model's response.
4. **PAN2025 Evaluation**: Run inference on the PAN2025 validation set and compute all metrics.
5. **HC3 Evaluation**: Run inference on the HC3 Wiki dataset to assess OOD generalization.
6. **Results Analysis**: Compare zero-shot performance against trained baselines.

### Variables

- **Independent**: LLM model (via Ollama), prompt design
- **Dependent**: ROC-AUC, Brier Score, F1, C@1, F0.5u
- **Controlled**: Dataset samples, response parsing logic, temperature (set to 0 for determinism)

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

Comparison against trained models from previous experiments:

| Model | PAN2025 Val ROC-AUC | HC3 ROC-AUC |
|-------|---------------------|-------------|
| Logistic Regression (001) | 0.9953 | 0.5520 |
| LightGBM (002) | 0.9980 | 0.3356 |
| DeBERTa (003) | 0.9948 | 0.5939 |

### Success Criteria

- **Promising if**: Zero-shot ROC-AUC ≥ 0.65 on **both** PAN2025 val and HC3
- **Interesting if**: OOD performance (HC3) exceeds trained baselines despite no training
- **Weak if**: ROC-AUC < 0.55 on either dataset (near random)
- **Failed if**: Model consistently outputs single class or unparseable responses

## Limitations

- **Inference speed**: LLM inference is significantly slower than trained classifiers (~seconds per sample vs. milliseconds)
- **Token limits**: Long texts may need truncation to fit context windows
- **Prompt sensitivity**: Results may vary significantly based on prompt wording
- **Cost/resources**: Running larger models (70B+) requires substantial hardware
- **Non-determinism**: Even with temperature=0, some models exhibit variability
- **No probability calibration**: Converting LLM responses to calibrated probabilities is non-trivial