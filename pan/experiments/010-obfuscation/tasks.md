# Tasks: 010-obfuscation

Obfuscation augmentation (homoglyph + synonym replacement) for robust AI text detection. Builds on experiment 009's LoRA pipeline and experiment 008's homoglyph technique, adding synonym substitution. Includes proper OOD data collection and comprehensive evaluation.

## Datasets

| Dataset | Samples | Type |
|---------|---------|------|
| PAN2025 Train | ~23,707 | Training |
| PAN2025 Val | ~3,589 | In-distribution eval |
| HC3 Wiki | ~1,684 | OOD eval |
| RAID | ~2,000 | OOD eval |

## Phase 1: Setup
- [x] Create experiment folder structure
- [x] Write proposal.md
- [x] Write pyproject.toml
- [x] Create tasks.md
- [x] Create src/010-obfuscation/ source directory

## Phase 2: Implementation — Augmentation
- [x] Implement homoglyph augmentation (confusable_homoglyphs + ZWJ)
- [x] Implement synonym replacement (WordNet)
- [x] Integrate augmentation into data pipeline

## Phase 3: Implementation — Training
- [x] Fork 009 training pipeline
- [x] Adapt config with 010-specific paths and augmentation params
- [x] Write run_training.py with augmentation integration

## Phase 4: OOD Data & Evaluation
- [x] Write pull_ood_data.py to properly download OOD datasets
- [x] Verify OOD data is genuinely different from HC3 splits
- [x] Write run_evaluation.py for all datasets
- [x] Evaluate on PAN2025 val, HC3, RAID

## Phase 5: Results & Documentation
- [x] Write results.md
- [x] Update experiment tracker (README)

## Artifacts

| File | Description |
|------|-------------|
| `artifacts/all_results.json` | Combined evaluation metrics |
| `artifacts/pan2025_val_results.json` | PAN2025 validation results |
| `artifacts/hc3_results.json` | HC3 Wiki OOD results |
| `artifacts/raid_results.json` | RAID OOD results |

## Success Criteria

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| PAN2025 ROC-AUC | ≥ 0.999 | 0.9995 | ✅ Pass |
| Avg OOD ROC-AUC | ≥ 0.95 | 0.9975 | ✅ Pass |
