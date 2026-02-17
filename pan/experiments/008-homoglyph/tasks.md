# Tasks: 008-homoglyph

Homoglyph data augmentation for robust AI text detection. Replicates the mdok augmentation technique (replacing characters with unicode look-alikes) on our Qwen2.5-1.5B LoRA setup from experiment 006.

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

## Phase 2: Implementation ✅
- [x] Create training notebook with homoglyph augmentation
  - [x] Download PAN2025 data
  - [x] Implement homoglyph augmentation (confusables + ZWJ)
  - [x] Augment 10% of AI training samples
  - [x] Fine-tune Qwen2.5-1.5B LoRA (same config as 006)
  - [x] Push adapter to HuggingFace

## Phase 3: Evaluation ✅
- [x] Create evaluation notebook
  - [x] Evaluate on PAN2025 val, HC3, RAID
  - [x] Compare against experiment 006 baseline
  - [x] Generate comparison tables and heatmap

## Phase 4: Results & Documentation ✅
- [x] Write results.md
- [x] Update tasks.md

## Artifacts

| File | Description |
|------|-------------|
| `artifacts/homoglyph_comparison.csv` | Full metrics comparison |
| `artifacts/homoglyph_heatmap.png` | Visual comparison heatmap |

## Success Criteria

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| PAN2025 ROC-AUC | ≥ 0.999 | 0.9998 | ✅ Met |
| Avg OOD ROC-AUC | ≥ 0.91 | 0.9033 | ❌ Not met |
