# Results: Zero-Shot LLM Detection via Ollama

**Date Completed**: 2026-02-07
**Author**: harshul

## Summary

Zero-shot classification using **Qwen2 7B via Ollama** performs **below random chance** on AI text detection, achieving ROC-AUC of 0.46 on PAN2025 validation and 0.42 on HC3. The model exhibited strong prediction bias and failed to generalize.

**Verdict**: ❌ **Rejected** — Zero-shot approach does not meet success criteria.

---

## Key Results

### ROC-AUC Comparison

| Model | PAN2025 Val | HC3 (OOD) | Avg |
|-------|-------------|-----------|-----|
| LR (001) | 0.9953 | 0.5520 | 0.774 |
| LightGBM (002) | 0.9980 | 0.3356 | 0.667 |
| DeBERTa (003) | 0.9948 | 0.5939 | 0.794 |
| **Ollama (005)** | **0.4606** | **0.4162** | **0.438** |

### Full Metrics

| Metric | PAN2025 Val | HC3 Wiki |
|--------|-------------|----------|
| ROC-AUC | 0.4606 | 0.4162 |
| Brier Score | 0.6680 | 0.5900 |
| F1 Score | 0.2511 | 0.5280 |
| C@1 | 0.3320 | 0.4100 |
| F0.5u | 0.3911 | 0.4666 |
| Accuracy | 33.2% | 41.0% |

---

## Observations

### Quantitative

| Dataset | AI Predictions | Human Predictions | True AI % |
|---------|----------------|-------------------|-----------|
| PAN2025 Val | 90 (18%) | 410 (82%) | ~50% |
| HC3 Wiki | 381 (76%) | 119 (24%) | ~50% |

### Qualitative

1. **Strong prediction bias**: The model predicts "HUMAN" for 82% of PAN2025 samples but "AI" for 76% of HC3 samples
2. **Inconsistent behavior**: Same prompt yields opposite biases on different datasets
3. **Below random performance**: ROC-AUC < 0.5 indicates predictions are inversely correlated with truth
4. **Zero parse failures**: All 1000 responses were successfully parsed as HUMAN/AI

---

## Analysis

### Why Zero-Shot Failed

1. **Task ambiguity**: The model was not trained specifically for AI detection and lacks a clear internal representation of what distinguishes AI-generated text
2. **Dataset-specific patterns**: The model may be picking up on superficial cues (text length, topics) rather than AI-writing signatures
3. **Prompt limitations**: A simple "classify as HUMAN or AI" prompt may not provide enough context for the model to reason effectively
4. **Model knowledge cutoff**: Qwen2's training data may not include sufficient examples of modern AI-generated text

### Prediction Bias Analysis

- **PAN2025**: Heavily biased toward HUMAN (82%) — model sees most text as human-written
- **HC3**: Heavily biased toward AI (76%) — model sees most text as AI-generated
- This suggests the model is not learning generalizable features but responding to dataset-specific characteristics

### Confounders

- Only tested one model (Qwen2 7B) — other models may perform differently
- Used only the default prompt — alternative prompts (CoT, few-shot) may improve results
- Sampled 500 texts per dataset — larger samples might stabilize estimates

---

## Artifacts

| Type | Location |
|------|----------|
| Predictions (PAN2025) | `artifacts/pan2025_val_predictions.json` |
| Predictions (HC3) | `artifacts/hc3_predictions.json` |
| Metrics (PAN2025) | `artifacts/pan2025_val_results.json` |
| Metrics (HC3) | `artifacts/hc3_results.json` |
| Combined Results | `artifacts/all_results.json` |

---

## Conclusions

1. **Zero-shot LLM detection is not viable** with current models and prompts
2. **Trained models significantly outperform** zero-shot approaches (0.55-0.99 vs 0.42-0.46 ROC-AUC)
3. **Even simple baselines (LR) beat zero-shot** by a large margin
4. **Model bias is a major issue** — predictions are heavily skewed toward one class per dataset

---

## Next Steps

- [ ] Try larger models (e.g., Llama 3 70B) if compute allows
- [ ] Experiment with chain-of-thought prompting to improve reasoning
- [ ] Try few-shot prompting with labeled examples
- [ ] Consider fine-tuning an open-source LLM on AI detection task
- [ ] Investigate why prediction bias differs so dramatically between datasets

**Leads to**: Future experiments should focus on fine-tuning approaches rather than zero-shot inference for AI text detection.