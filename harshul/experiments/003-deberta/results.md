# Results: DeBERTa-v3 Fine-tuning for AI Text Detection

**Date Completed**: 2026-01-31
**Author**: harshul

## Summary

DeBERTa-v3 achieves near-perfect performance on the in-distribution PAN2025 validation set (ROC-AUC: 0.9948) but shows **limited improvement** on out-of-distribution generalization compared to the LR baseline (HC3 ROC-AUC: 0.5939 vs 0.5520).

**Verdict**: Partially confirmed - DeBERTa slightly improves OOD generalization over baselines but does not achieve the target of ≥0.70 ROC-AUC on HC3.

## Observations

### Quantitative

| Metric | LR Baseline (001) | LightGBM (002) | DeBERTa (003) | Delta vs LR |
|--------|-------------------|----------------|---------------|-------------|
| **PAN2025 Validation** |
| ROC-AUC | 0.9953 | 0.9980 | **0.9948** | -0.0005 |
| Brier | 0.0282 | - | 0.1106 | +0.0824 |
| F1 | 0.9772 | - | 0.8834 | -0.0938 |
| C@1 | 0.9705 | - | 0.8654 | -0.1051 |
| F0.5u | 0.9732 | - | 0.9498 | -0.0234 |
| **HC3 External (OOD)** |
| ROC-AUC | 0.5520 | 0.3356 | **0.5939** | **+0.0419** |
| Brier | 0.2706 | - | 0.4939 | +0.2233 |
| F1 | 0.6079 | - | 0.0024 | -0.6055 |
| C@1 | 0.5220 | - | 0.5006 | -0.0214 |
| F0.5u | 0.5487 | - | 0.0059 | -0.5428 |

### Training Details

- **Model**: `microsoft/deberta-v3-base`
- **Training Platform**: Google Colab (GPU)
- **Epochs**: 3
- **Batch Size**: 2 (with gradient accumulation)
- **Learning Rate**: 2e-5
- **Max Sequence Length**: 512 tokens (with chunking for longer docs)
- **Model hosted at**: [hersheys-baklava/deberta-pan2025](https://huggingface.co/hersheys-baklava/deberta-pan2025)

### Long Document Strategy

- **Chunking**: Sliding window with 512 tokens and 256 token stride (50% overlap)
- **Aggregation**: Mean pooling of chunk probabilities
- **Max supported length**: ~6,500 tokens (yields ~25 chunks)

### Qualitative Observations

1. **Strong in-distribution performance**: DeBERTa maintains near-perfect ROC-AUC (0.9948) on PAN2025, matching previous baselines.

2. **Modest OOD improvement**: HC3 ROC-AUC improved from 0.5520 (LR) to 0.5939 (+4.2 percentage points), suggesting DeBERTa captures slightly more generalizable patterns.

3. **Threshold calibration issue**: The extremely low F1 (0.0024) and F0.5u (0.0059) on HC3 indicate the model is predicting almost everything as "Human" (class 0). This suggests the probability threshold (0.5) is miscalibrated for the external dataset.

4. **Higher Brier score**: DeBERTa has higher Brier scores than LR baseline, indicating less well-calibrated probability estimates.

## Analysis

### Key Findings

1. **ROC-AUC improved, but threshold is wrong**
   - The improved ROC-AUC (0.5939) indicates DeBERTa's probability rankings are better than random.
   - However, the near-zero F1 shows the 0.5 decision threshold is inappropriate for HC3.
   - The model likely outputs lower probabilities on average for HC3 data.

2. **Distribution shift problem persists**
   - Despite using a transformer model, the domain shift between PAN2025 and HC3 remains a major challenge.
   - The model has learned PAN2025-specific patterns that don't transfer well.

3. **LightGBM performed worse than random on OOD**
   - LightGBM's ROC-AUC of 0.3356 was actually *inverted* (below random).
   - DeBERTa avoids this inversion, suggesting more robust features.

### Recommendations

1. **Recalibrate threshold for OOD**: Find optimal threshold on a small OOD calibration set.
2. **Domain-adaptive fine-tuning**: Fine-tune on a mix of PAN2025 and HC3-like data.
3. **Analyze prediction distributions**: Compare probability histograms between datasets.

### Confounders

- CPU inference is slower (~1 hour for 3.5k samples), but doesn't affect metrics.
- The chunking strategy may lose some context at chunk boundaries.
- HC3 data may have different label semantics or text characteristics.

## Artifacts

| Type | Location |
|------|----------|
| Model | `./artifacts/model/` (local cache) |
| HuggingFace | [hersheys-baklava/deberta-pan2025](https://huggingface.co/hersheys-baklava/deberta-pan2025) |
| Val Metrics | `./artifacts/val_metrics.json` |
| External Metrics | `./artifacts/external_metrics.json` |

## Next Steps

- [ ] Analyze probability distribution differences between PAN2025 and HC3
- [ ] Find optimal threshold for HC3 dataset
- [ ] Generate confusion matrix and ROC curve plots
- [ ] Test with threshold recalibration
- [ ] Investigate domain adaptation techniques

**Leads to**: Understanding that raw OOD generalization remains challenging even for transformers. Next experiment should focus on domain adaptation or multi-domain training.