# Tasks: 009-lora-replication

Clean replication of experiment 006 (Qwen2.5-1.5B LoRA fine-tuning). Same hyperparameters, restructured as notebooks + reusable source modules.

## Datasets

| Dataset | Samples | Purpose |
|---------|---------|---------|
| PAN2025 Train | 23,707 | Training |
| PAN2025 Val | 3,589 | In-distribution evaluation |
| HC3 Wiki | ~1,684 | OOD evaluation |
| RAID | 2,000 | OOD evaluation |

---

## Phase 1: Setup ✅

- [x] Create experiment folder from template
- [x] Write proposal.md
- [x] Update pyproject.toml
- [x] Create src/009-lora-replication/ directory

## Phase 2: Implementation ✅

- [x] Write config.py (paths, hyperparameters)
- [x] Write data.py (JSONL loading, tokenization)
- [x] Write model.py (LoRA model creation, loading)
- [x] Write train.py (Trainer setup, metrics callback)
- [x] Write evaluate.py (chunked inference, metrics)

## Phase 3: Notebooks

- [ ] Write training.ipynb
- [ ] Write evaluation.ipynb

## Phase 4: Training

- [ ] Run training notebook on Colab
- [ ] Save adapter + push to HuggingFace

## Phase 5: Evaluation

- [ ] Run evaluation notebook on Colab
- [ ] Evaluate on PAN2025 val, HC3, RAID
- [ ] Compare against 006 results

## Phase 6: Documentation

- [ ] Write results.md
- [ ] Update experiments README

---

## Artifacts

| File | Description |
|------|-------------|
| `artifacts/all_results.json` | Combined evaluation results |
| `artifacts/pan2025_val_results.json` | PAN2025 val metrics |
| `artifacts/hc3_results.json` | HC3 OOD metrics |
| `artifacts/raid_results.json` | RAID OOD metrics |

## Success Criteria

| Criterion | Target | Status |
|-----------|--------|--------|
| PAN2025 Val ROC-AUC | ≥ 0.98 | Pending |
| HC3 ROC-AUC | ≥ 0.60 | Pending |
| Results match 006 | ±0.01 ROC-AUC | Pending |
