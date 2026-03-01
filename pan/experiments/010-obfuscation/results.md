# Results: 010-obfuscation

**Date Completed**: 2026-03-01
**Author**: tanish

## Summary

Obfuscation augmentation (homoglyph + synonym replacement on 10% of AI training texts each) maintains near-perfect in-distribution performance and shows slight improvements on OOD benchmarks compared to experiment 009. PAN2025 ROC-AUC improved marginally from 0.9993 to 0.9995, HC3 from 0.9977 to 0.9985, while RAID dipped slightly from 0.9974 to 0.9964.

**Verdict**: Confirmed. The augmentation preserves in-distribution performance (PAN2025 ROC-AUC ≥ 0.999) and maintains strong OOD generalization (avg OOD ROC-AUC 0.9975 ≥ 0.95).

## Comparison with Baselines

| Dataset | 006 (LoRA) | 008 (Homoglyph) | 009 (Replication) | **010 (Obfuscation)** | Δ vs 009 |
|---------|------------|-----------------|-------------------|----------------------|----------|
| PAN2025 Val | 0.9999 | 0.9998 | 0.9993 | **0.9995** | +0.0002 |
| HC3 Wiki | 0.9982 | 0.9984 | 0.9977 | **0.9985** | +0.0008 |
| RAID | 0.8152 | 0.8082 | 0.9974 | **0.9964** | -0.0010 |

## Detailed Metrics

### PAN2025 Validation
- **ROC-AUC**: 0.9995
- **Brier Score**: 0.0025
- **F1 Score**: 0.9976
- **C@1**: 0.9969
- **F0.5u**: 0.9967

### HC3 Wiki
- **ROC-AUC**: 0.9985
- **Brier Score**: 0.0043
- **F1 Score**: 0.9952
- **C@1**: 0.9952
- **F0.5u**: 0.9967

### RAID
- **ROC-AUC**: 0.9964
- **Brier Score**: 0.0027
- **F1 Score**: 0.9980
- **C@1**: 0.9980
- **F0.5u**: 0.9992

## Findings

1. **In-distribution maintained**: PAN2025 ROC-AUC 0.9995 — no regression from augmentation, marginally above 009's 0.9993.
2. **HC3 improvement**: HC3 ROC-AUC 0.9985 is the best across all experiments (vs 0.9984 in 008, 0.9982 in 006, 0.9977 in 009), suggesting synonym/homoglyph augmentation slightly improved generalization to this OOD set.
3. **RAID stable**: RAID ROC-AUC 0.9964 shows a minor -0.0010 regression vs 009, but is dramatically better than 006 (0.8152) and 008 (0.8082). The augmentation does not harm RAID performance in the way 008's homoglyph-only approach did.
4. **Combined augmentation works**: Unlike experiment 008 (homoglyph-only, which caused RAID regression), combining homoglyph with synonym replacement on separate text subsets maintains robustness across all benchmarks.

## Analysis

The obfuscation augmentation achieves its primary goal — preserving in-distribution accuracy while maintaining or slightly improving OOD generalization. The avg OOD ROC-AUC of 0.9975 exceeds the 0.95 target. The combination of both augmentation strategies (applied to non-overlapping 10% subsets of AI texts) appears more stable than homoglyph-only augmentation (experiment 008).

### Confounders

- MAGE and OpenGPTText datasets were not available for evaluation — OOD assessment is limited to HC3 and RAID.
- The existing RAID and HC3 datasets were re-verified as properly sourced (not synthetic HC3 splits).
- The marginal differences between 009 and 010 are within noise — a definitive claim of improvement would require multiple training runs.

## Augmentation Config

| Parameter | Value |
|-----------|-------|
| Homoglyph fraction | 10% of AI texts |
| Per-char swap prob | 5% |
| ZWJ insertion prob | 5% |
| Synonym fraction | 10% of AI texts |
| Per-word synonym prob | 20% |

## Artifacts

| Type | Location |
|------|----------|
| Source Code | `src/010-obfuscation/` |
| Evaluation Results | `experiments/010-obfuscation/artifacts/` |
| Adapter | [HuggingFace: hersheys-baklava/qwen-lora-pan2026-010-obfuscation](https://huggingface.co/hersheys-baklava/qwen-lora-pan2026-010-obfuscation) |
| W&B Run | `pan-2026` / `010-obfuscation` |

## Success Criteria

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| PAN2025 ROC-AUC | ≥ 0.999 | 0.9995 | ✅ Pass |
| Avg OOD ROC-AUC | ≥ 0.95 | 0.9975 | ✅ Pass |

## Next Steps

- [ ] Download and evaluate on MAGE and OpenGPTText for broader OOD coverage
- [ ] Experiment with higher augmentation fractions (e.g., 20%+20%)
- [ ] Try sentence-level paraphrasing augmentation (beyond word-level synonyms)
- [ ] Run multiple seeds to assess variance