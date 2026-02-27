# Tasks: 010-obfuscation

Obfuscation augmentation (homoglyph + synonym replacement) for robust AI text detection. Builds on experiment 009's LoRA pipeline and experiment 008's homoglyph technique, adding synonym substitution. Includes proper OOD data collection and comprehensive evaluation.

## Datasets

| Dataset | Samples | Type |
|---------|---------|------|
| PAN2025 Train | ~23,707 | Training |
| PAN2025 Val | ~3,589 | In-distribution eval |
| HC3 Wiki | ~1,684 | OOD eval |
| RAID | ~2,000 | OOD eval |
| MAGE | TBD | OOD eval (new) |
| OpenGPTText | TBD | OOD eval (new) |

## Phase 1: Setup
- [x] Create experiment folder structure
- [x] Write proposal.md
- [x] Write pyproject.toml
- [x] Create tasks.md
- [ ] Create src/010-obfuscation/ source directory

## Phase 2: Implementation — Augmentation
- [ ] Implement homoglyph augmentation (confusables + ZWJ)
- [ ] Implement synonym replacement
- [ ] Integrate augmentation into data pipeline

## Phase 3: Implementation — Training
- [ ] Fork 009 training pipeline
- [ ] Adapt config with 010-specific paths and augmentation params
- [ ] Write run_training.py with augmentation integration

## Phase 4: OOD Data & Evaluation
- [ ] Write pull_ood_data.py to properly download OOD datasets
- [ ] Verify OOD data is genuinely different from HC3 splits
- [ ] Write run_evaluation.py for all datasets
- [ ] Evaluate on PAN2025 val + all OOD

## Phase 5: Results & Documentation
- [ ] Write results.md
- [ ] Update experiment tracker

## Artifacts

| File | Description |
|------|-------------|
| `artifacts/all_results.json` | Combined evaluation metrics |
| `artifacts/lora_adapter/` | Trained LoRA adapter |
| `artifacts/*_results.json` | Per-dataset result files |

## Success Criteria

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| PAN2025 ROC-AUC | ≥ 0.999 | — | Pending |
| Avg OOD ROC-AUC | ≥ 0.95 | — | Pending |
