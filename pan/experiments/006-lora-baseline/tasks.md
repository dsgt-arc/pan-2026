# Tasks: 006-lora-baseline

LoRA (Low-Rank Adaptation) fine-tuning of Qwen2.5-1.5B (a causal LLM) for AI text detection. Compares parameter-efficient fine-tuning of a decoder-only model against full fine-tuning of an encoder (003-deberta) using the same evaluation metrics and datasets.

## Datasets

| Dataset | Samples | Purpose |
|---------|---------|---------|
| PAN2025 Train | 23,707 | Training |
| PAN2025 Val | 3,589 | In-distribution evaluation |
| HC3 Wiki | ~3,500 | OOD evaluation |

---

## Phase 1: Environment Setup ✅

- [x] Create experiment folder structure
- [x] Write proposal.md
- [x] Update pyproject.toml with `peft` dependency
- [x] Install dependencies (`peft`)

## Phase 2: Implementation ✅

- [x] Create data loading / tokenization module (left-padded for causal LM)
- [x] Create LoRA model configuration module (Qwen2.5-1.5B + classification head)
- [x] Create training script with HuggingFace Trainer + LoRA
- [x] Create evaluation script with chunking support
- [x] Create main run_experiment.py
- [x] Create Colab training notebook (`notebooks/006-lora-baseline/lora_training.ipynb`)

## Phase 3: Training

- [ ] Test training on small sample (10 samples)
- [ ] Run full training on PAN2025 train
- [ ] Save LoRA adapter weights

## Phase 4: Evaluation

- [ ] Evaluate on PAN2025 validation (in-distribution)
- [ ] Evaluate on HC3 Wiki (out-of-distribution)
- [ ] Compute all 5 standard metrics (ROC-AUC, Brier, F1, C@1, F0.5u)

## Phase 5: Analysis & Documentation

- [ ] Compare results against baselines (001–005)
- [ ] Report trainable parameter count vs full fine-tuning
- [ ] Write results.md with findings
- [ ] Update experiments README

---

## Artifacts

| File | Description |
|------|-------------|
| `artifacts/pan2025_val_results.json` | Validation set metrics |
| `artifacts/hc3_results.json` | HC3 OOD metrics |
| `artifacts/all_results.json` | Combined results |
| `artifacts/lora_adapter/` | Saved LoRA adapter weights |

## Success Criteria

| Criterion | Target | Status |
|-----------|--------|--------|
| PAN2025 Val ROC-AUC | ≥ 0.98 | Pending |
| HC3 ROC-AUC | ≥ 0.60 | Pending |
| Trainable params | < 1% of full model | Pending |
