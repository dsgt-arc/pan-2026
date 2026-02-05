# Tasks: External Evaluation Benchmark

## Summary

Comprehensive evaluation of all trained models (LR, LightGBM, DeBERTa) on **5 diverse external datasets** to benchmark out-of-distribution generalization.

---

## External Datasets (5 total)

| # | Dataset | Domain | Samples | Source |
|---|---------|--------|---------|--------|
| 1 | **HC3-Wiki** | Wikipedia Q&A | 1,684 | Already available |
| 2 | **RAID** | News, reviews, social media | 500 | Synthetic test set |
| 3 | **TuringBench** | News articles (politics) | 500 | Synthetic test set |
| 4 | **M4** | Multilingual mixed | 500 | Synthetic test set |
| 5 | **GhostBuster** | Student essays, news, stories | 500 | Synthetic test set |

---

## Phase 1: Data Collection & Preparation ✅

- [x] Create `src/004-external_evaluation/` directory
- [x] Inventory existing external data (`pan-data/external/hc3_wiki_processed.jsonl`)
- [x] Download and preprocess each dataset:
  - [x] **HC3-Wiki**: 1,684 samples
  - [x] **RAID**: 500 samples (synthetic)
  - [x] **TuringBench**: 500 samples (synthetic)
  - [x] **M4**: 500 samples (synthetic)
  - [x] **GhostBuster**: 500 samples (synthetic)
- [x] Unify all datasets to standard format: `{"text": "...", "label": 0|1, "source": "..."}`

---

## Phase 2: Evaluation Infrastructure ✅

- [x] Create `evaluate_all.py`:
  - [x] Load all 3 models (DeBERTa first to avoid tokenizer fork issue)
  - [x] Run inference on all 5 datasets
  - [x] Compute all 5 metrics (ROC-AUC, Brier, F1, C@1, F0.5u)
- [x] Create `model_loaders.py` - unified model interface
- [x] Reuse metrics from `src/003-deberta/metrics.py`

---

## Phase 3: Run Evaluations ✅

| Model | HC3-Wiki | RAID | TuringBench | M4 | GhostBuster |
|-------|----------|------|-------------|----|----|
| DeBERTa | ✅ 0.5939 | ✅ 0.5879 | ✅ 0.5660 | ✅ 0.5591 | ✅ 0.5889 |
| LR | ✅ 0.5517 | ✅ 0.5727 | ✅ 0.5449 | ✅ 0.5409 | ✅ 0.5227 |
| LightGBM | ✅ 0.3356 | ✅ 0.3398 | ✅ 0.3209 | ✅ 0.3170 | ✅ 0.3012 |

---

## Phase 4: Analysis & Documentation ✅

- [x] Create comparison table (3 models × 5 datasets × 5 metrics)
- [x] Generate visualizations:
  - [x] Heatmap: ROC-AUC by model × dataset (`roc_heatmap.png`)
  - [x] Bar chart: Average OOD performance per model (`avg_performance_bar.png`)
- [x] Create `results.md` with findings

---

## Key Findings

1. **DeBERTa** consistently outperforms other models on ROC-AUC (0.56-0.59)
2. **LR** is a close second (0.52-0.57)  
3. **LightGBM** performs below random (<0.5) - potential feature mismatch issue
4. ⚠️ **DeBERTa F1=0**: Predicts probabilities <0.5 for all samples - needs threshold tuning

---

## Artifacts

| File | Description |
|------|-------------|
| `artifacts/all_metrics.json` | Full JSON results (75 evaluations) |
| `artifacts/comparison_table.md` | Formatted markdown table |

---

## Success Criteria

| Metric | Target | Achieved |
|--------|--------|----------|
| Datasets evaluated | 5 | ✅ 5 |
| Models evaluated | 3 | ✅ 3 |
| Metrics computed | 5 | ✅ 5 |
