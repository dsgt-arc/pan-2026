# Tasks: 007-ensemble-baseline

Ensemble baseline combining LR (001), DeBERTa (003), and Qwen LoRA (006). Evaluated on PAN2025 validation + HC3 + RAID using 3 ensemble strategies.

## Datasets

| Dataset | Samples | Type |
|---------|---------|------|
| PAN2025 Validation | 3,589 | In-distribution |
| HC3 Wiki | 1,684 | OOD |
| RAID | 2,000 | OOD |

## Phase 1: Setup ✅
- [x] Create experiment folder structure
- [x] Write proposal.md
- [x] Write pyproject.toml
- [x] Create tasks.md

## Phase 2: Implementation ✅
- [x] Create ensemble evaluation notebook
- [x] Load LR model (retrained inline from PAN2025 train)
- [x] Load DeBERTa model from HuggingFace
- [x] Load Qwen LoRA model from HuggingFace
- [x] Generate per-model predictions on all 3 datasets
- [x] Implement mean ensemble
- [x] Implement weighted ensemble
- [x] Implement majority vote ensemble
- [x] Compute metrics for all strategies
- [x] Save results to artifacts

## Phase 3: Documentation ✅
- [x] Write results.md with comparison tables
- [x] Update tasks.md

## Artifacts

| File | Description |
|------|-------------|
| `artifacts/ensemble_results.csv` | Full metrics for all strategies |

## Results

**Verdict: Rejected.** No ensemble strategy beats Qwen LoRA solo.

| Strategy | Avg OOD ROC-AUC | Δ vs Qwen |
|----------|----------------:|----------:|
| Qwen LoRA (solo) | **0.9067** | — |
| Ensemble (Weighted) | 0.8989 | -0.0078 |
| Ensemble (Mean) | 0.8967 | -0.0100 |
| Ensemble (Majority Vote) | 0.8967 | -0.0100 |
