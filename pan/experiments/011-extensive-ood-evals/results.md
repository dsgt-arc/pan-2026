# Results: 011-extensive-ood-evals

**Date Completed**: 2026-03-02
**Author**: tanish

## Summary

Evaluated the experiment 010 obfuscation-augmented Qwen 2.5-1.5B LoRA model on **5 out of 7 planned OOD datasets**. TuringBench and HC3 Multi-Domain failed to load at runtime (likely due to dataset availability or format issues on HuggingFace) and were excluded.

The model shows **extreme polarization** in OOD generalization. It achieves near-perfect performance on M4/SemEval 2024 (0.9915) and RAID Adversarial (1.0000), but **catastrophically fails** on AI Detection Pile (0.4119 — worse than random) and underperforms on MAGE (0.6702). This reveals that the model's generalization is highly domain-dependent rather than broadly robust.

**Overall average ROC-AUC: 0.7975** — below the 0.90 target. The evaluation **fails all three success criteria**.

## Results Table

| Dataset | ROC-AUC | Brier | F1 | C@1 | F0.5u |
|---------|---------|-------|-----|------|-------|
| MAGE | 0.6702 | 0.3530 | 0.4316 | 0.6260 | 0.6272 |
| OpenGPTText | 0.9139 | 0.1985 | 0.7241 | 0.7820 | 0.8614 |
| AI Detection Pile | 0.4119 | 0.4544 | 0.2504 | 0.5215 | 0.3802 |
| M4 / SemEval 2024 | 0.9915 | 0.0139 | 0.9848 | 0.9850 | 0.9926 |
| RAID Adversarial | 1.0000 | 0.0033 | 0.9960 | 0.9960 | 0.9984 |
| ~~TuringBench~~ | — | — | — | — | — |
| ~~HC3 Multi-Domain~~ | — | — | — | — | — |
| **AVERAGE (5 datasets)** | **0.7975** | **0.2046** | **0.6774** | **0.7821** | **0.7720** |

## Baseline (Experiment 010 — In-Distribution)

| Dataset | ROC-AUC |
|---------|---------|
| PAN2025 Val | 0.9995 |
| HC3 Wiki | 0.9985 |
| RAID | 0.9964 |

## Per-Dataset Analysis

### Tier 1: Near-Perfect (ROC-AUC ≥ 0.99)

- **RAID Adversarial (1.0000)**: The model handles adversarial perturbations flawlessly. This is unsurprising since RAID was part of the experiment 010 evaluation suite, but the adversarial subset specifically tests robustness to attacks — the model passes this with flying colors. Brier score of 0.0033 indicates excellent calibration.

- **M4 / SemEval 2024 (0.9915)**: Near-perfect discrimination on this multi-domain benchmark. All metrics are above 0.98, and the Brier score of 0.0139 shows highly calibrated predictions. This suggests strong generalization to the types of AI-generated text represented in this competition dataset.

### Tier 2: Good (ROC-AUC ≥ 0.90)

- **OpenGPTText (0.9139)**: Solid performance on GPT-generated Wikipedia introductions. The model correctly identifies most AI text, though the F1 of 0.7241 suggests some difficulty at the decision boundary. The Brier score of 0.1985 indicates moderate calibration issues — the model is less confident in its predictions for this domain.

### Tier 3: Poor (ROC-AUC < 0.70)

- **MAGE (0.6702)**: Below the 0.70 floor target. MAGE spans 27 different LLMs across multiple domains. The model's Brier score of 0.3530 and F1 of 0.4316 suggest it struggles to distinguish many of these generators from human text. This is a significant generalization gap.

- **AI Detection Pile (0.4119)**: **Worse than random chance** (0.50). The model's predictions are actively inverted for this dataset — it tends to classify AI text as human and vice versa. The Brier score of 0.4544 is near the worst possible value. This dataset contains long-form essays from GPT-2/3/ChatGPT/GPT-J, suggesting the model has learned features that are anti-correlated with AI-generation signals for these specific generators and text lengths.

## Visualizations

### ROC-AUC Comparison
![ROC-AUC Comparison](artifacts/ood_roc_comparison.png)

The bar chart vividly illustrates the three-tier split: RAID and M4 at ~1.0, OpenGPTText at ~0.91, and MAGE and AI Detection Pile far below the floor threshold.

### Metrics Heatmap
![Metrics Heatmap](artifacts/ood_metrics_heatmap.png)

The heatmap reveals that the performance gap is consistent across all metrics — it's not just ROC-AUC. AI Detection Pile shows uniformly poor metrics, while M4 and RAID are dark blue across the board.

### ROC Curves
![ROC Curves](artifacts/ood_roc_curves.png)

The ROC curve overlay clearly shows the AI Detection Pile curve (green) falling *below* the diagonal, confirming anti-correlated predictions. MAGE (pink) tracks loosely above the diagonal but far from the top-left corner. M4 and RAID curves are practically glued to the perfect-classification corner.

### Calibration Curves
![Calibration Curves](artifacts/ood_calibration_curves.png)

Calibration varies dramatically by dataset. M4 and OpenGPTText follow the ideal diagonal reasonably well, while MAGE and AI Detection Pile show erratic, unreliable calibration. The AI Detection Pile calibration curve (green) trends downward at high predicted probabilities — meaning the model is most confident precisely when it is most wrong.

## Success Criteria

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Avg OOD ROC-AUC | ≥ 0.90 | 0.7975 | ❌ Fail |
| Floor ROC-AUC (no dataset below) | ≥ 0.70 | 0.4119 (AI Detection Pile) | ❌ Fail |
| Avg Brier Score | ≤ 0.10 | 0.2046 | ❌ Fail |

## Key Takeaways

1. **The model is not broadly generalizable.** Despite excellent in-distribution performance (0.9995 on PAN2025) and strong results on some OOD benchmarks, it catastrophically fails on others. The model has overfit to PAN-style AI text rather than learning general AI-detection features.

2. **Domain matters more than expected.** The model excels on datasets whose text distribution is closer to PAN2025's (news, Q&A, adversarial rewrites) but fails on datasets with different characteristics (multi-generator MAGE, long-form essay Pile).

3. **Anti-correlated predictions are a red flag.** The AI Detection Pile result (0.4119) suggests the model has learned heuristics that are counter-productive for certain text types — potentially length-based or formatting-based features that happen to correlate differently in the Pile.

4. **Two datasets (TuringBench, HC3 Multi-Domain) failed to load.** This reduces the evaluation scope from 7 to 5 datasets. These should be investigated and included in a follow-up.

5. **For CLEF submission, these weaknesses may not be fatal** if PAN2026's test distribution is similar to PAN2025's. However, the results highlight risks if the competition introduces novel generators or text styles.

## Artifacts

All artifacts are stored on HuggingFace: [`hersheys-baklava/pan2026-011-ood-eval-results`](https://huggingface.co/datasets/hersheys-baklava/pan2026-011-ood-eval-results)

| File | Description |
|------|-------------|
| `all_ood_results.json` | Combined metrics for all datasets |
| `ood_roc_comparison.png` | ROC-AUC bar chart |
| `ood_metrics_heatmap.png` | All metrics heatmap |
| `ood_roc_curves.png` | Overlaid ROC curves |
| `ood_calibration_curves.png` | Calibration reliability diagrams |
| `*_results.json` | Per-dataset metrics |
| `*_predictions.json` | Per-dataset predictions and labels |
| `results_draft.md` | Auto-generated results draft |