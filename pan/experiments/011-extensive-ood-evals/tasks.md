# Tasks: 011-extensive-ood-evals

Evaluate the experiment 010 obfuscation-augmented Qwen2.5-1.5B LoRA model across 7 out-of-distribution datasets. All evaluation runs in a self-contained Google Colab notebook.

## Datasets

| # | Dataset | Source | Domain | ~Samples |
|---|---------|--------|--------|----------|
| 1 | MAGE | yaful/MAGE | Multi-domain, 27 LLMs | 1000 |
| 2 | OpenGPTText | aadityaubhat/GPT-wiki-intro | Wikipedia | 1000 |
| 3 | TuringBench | turingbench/TuringBench | News | 1000 |
| 4 | AI Detection Pile | artem9k/ai-text-detection-pile | Essays | 1000 |
| 5 | M4 (SemEval 2024) | mbzuai-nlp/SemEval2024-task8 | Multi-domain | 1000 |
| 6 | HC3 Multi-Domain | Hello-SimpleAI/HC3 (non-wiki) | Q&A | 1000 |
| 7 | RAID Adversarial | liamdugan/raid (adversarial) | Multi-domain | 1000 |

## Phase 1: Setup ✅
- [x] Create experiment folder structure
- [x] Write proposal.md
- [x] Write tasks.md

## Phase 2: Implementation ✅
- [x] Create self-contained Colab evaluation notebook
- [x] Implement dataset loading for all 7 OOD datasets
- [x] Implement model loading from HuggingFace
- [x] Implement chunked inference pipeline
- [x] Implement PAN metrics computation
- [x] Implement visualization generation

## Phase 3: Execution ✅
- [x] Run notebook on Google Colab
- [x] ~~Verify all 7 datasets download correctly~~ 5/7 loaded (TuringBench, HC3 Multi-Domain failed)
- [x] Complete inference on all datasets
- [x] Save results to artifacts (uploaded to HuggingFace)

## Phase 4: Analysis ✅
- [x] Write results.md
- [x] Update README.md experiment table
- [x] Update proposal.md status

## Artifacts

| File | Description |
|------|-------------|
| `011_extensive_ood_evals.py` | Self-contained Colab notebook (as .py for version control) |
| `artifacts/all_ood_results.json` | Combined metrics for all datasets |
| `artifacts/ood_roc_comparison.png` | ROC-AUC bar chart |
| `artifacts/ood_metrics_heatmap.png` | Full metrics heatmap |
| `artifacts/*_results.json` | Per-dataset metrics |

## Success Criteria

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Avg OOD ROC-AUC | ≥ 0.90 | 0.7975 | ❌ Fail |
| Floor ROC-AUC | ≥ 0.70 (all datasets) | 0.4119 (AI Detection Pile) | ❌ Fail |
| Avg Brier Score | ≤ 0.10 | 0.2046 | ❌ Fail |
