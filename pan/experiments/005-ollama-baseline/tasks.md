# Tasks: Zero-Shot LLM Detection via Ollama

## Summary

Evaluate the zero-shot AI text detection capability of locally-hosted LLMs via **Ollama**. No training is performed—models classify text as human or AI-generated using prompt-based inference only.

---

## Datasets

| # | Dataset | Domain | Samples | Purpose |
|---|---------|--------|---------|---------|
| 1 | **PAN2025 Validation** | Mixed AI generators | 500 sampled | In-distribution evaluation |
| 2 | **HC3-Wiki** | Wikipedia Q&A (ChatGPT) | 500 sampled | Out-of-distribution evaluation |

---

## Phase 1: Environment Setup ✅

- [x] Install Ollama on local machine (v0.13.0)
- [x] Download target LLM model(s):
  - [ ] `llama3` (not installed)
  - [ ] `mistral` (not installed)
  - [x] `qwen2:7b` (primary model)
  - [x] `gemma2:9b` (available backup)
- [x] Verify Ollama is running and accessible via Python client
- [x] Create `src/005-ollama-baseline/` directory structure

---

## Phase 2: Prompt Engineering ✅

- [x] Design zero-shot classification prompt:
  - [x] Clear instruction for binary classification (human vs AI)
  - [x] Request structured output ("HUMAN" or "AI")
  - [x] Optional: Confidence version and chain-of-thought version
- [x] Test prompt on a few examples manually
- [x] Document final prompt templates in `prompts.py`:
  - `default`: Simple binary classification
  - `confidence`: With 0-100 confidence score
  - `cot`: Chain-of-thought reasoning

---

## Phase 3: Inference Pipeline ✅

- [x] Create `infer.py`:
  - [x] Load dataset (JSONL format)
  - [x] Send each text to Ollama API
  - [x] Parse model response to extract prediction
  - [x] Handle edge cases (unparseable responses, timeouts)
  - [x] Save predictions to output file
- [x] Add configuration for:
  - [x] Model selection (`--model`)
  - [x] Temperature (default: 0 for determinism)
  - [x] Max tokens for response (50)
  - [x] Sample limiting (`--max-samples`)

---

## Phase 4: Evaluation on PAN2025 Validation ✅

- [x] Run inference on PAN2025 validation set (500 samples)
- [x] Compute all 5 metrics:
  - [x] ROC-AUC: **0.4606**
  - [x] Brier Score: **0.668**
  - [x] F1 Score: **0.251**
  - [x] C@1: **0.332**
  - [x] F0.5u: **0.391**
- [x] Save results to `artifacts/pan2025_val_results.json`
- [x] Document in-distribution performance

---

## Phase 5: Evaluation on HC3 ✅

- [x] Run inference on HC3 Wiki dataset (500 samples)
- [x] Compute all 5 metrics:
  - [x] ROC-AUC: **0.4162**
  - [x] Brier Score: **0.590**
  - [x] F1 Score: **0.528**
  - [x] C@1: **0.410**
  - [x] F0.5u: **0.467**
- [x] Save results to `artifacts/hc3_results.json`
- [x] Compare OOD performance against trained baselines

---

## Phase 6: Analysis & Documentation ✅

- [x] Create comparison table:

| Model | PAN2025 Val ROC-AUC | HC3 ROC-AUC |
|-------|---------------------|-------------|
| LR (001) | 0.9953 | 0.5520 |
| LightGBM (002) | 0.9980 | 0.3356 |
| DeBERTa (003) | 0.9948 | 0.5939 |
| **Ollama (005)** | **0.4606** | **0.4162** |

- [x] Analyze results:
  - [x] Does zero-shot approach beat random (>0.5)? **NO** - below random on both datasets
  - [x] How does OOD performance compare to trained models? **Worse than all baselines**
  - [x] Are there patterns in misclassifications? **Strong bias toward HUMAN predictions on PAN, AI predictions on HC3**
- [x] Fill out `results.md` with findings
- [x] Document any prompt iterations and their effects

---

## Phase 7: Optional Extensions

- [ ] Try multiple LLM models and compare performance
- [ ] Experiment with different prompt strategies:
  - [ ] Chain-of-thought prompting
  - [ ] Few-shot examples (if zero-shot underperforms)
- [ ] Analyze inference speed and resource usage
- [ ] Test on additional external datasets (RAID, TuringBench, etc.)

---

## Artifacts

| File | Description |
|------|-------------|
| `artifacts/pan2025_val_predictions.json` | Raw predictions for PAN2025 val |
| `artifacts/pan2025_val_results.json` | Metrics for PAN2025 validation |
| `artifacts/hc3_predictions.json` | Raw predictions for HC3 |
| `artifacts/hc3_results.json` | Metrics for HC3 external dataset |
| `artifacts/all_results.json` | Combined results summary |
| `src/005-ollama-baseline/infer.py` | Main inference script |
| `src/005-ollama-baseline/prompts.py` | Prompt templates |
| `src/005-ollama-baseline/evaluate.py` | Metrics computation |
| `src/005-ollama-baseline/run_experiment.py` | End-to-end runner |

---

## Success Criteria

| Metric | Target | Status |
|--------|--------|--------|
| PAN2025 Val ROC-AUC | ≥ 0.65 | ❌ 0.4606 |
| HC3 ROC-AUC | ≥ 0.65 | ❌ 0.4162 |
| Inference pipeline working | Yes | ✅ Complete |
| Results documented | Yes | ✅ Complete |

---

## Notes

- **Inference speed**: ~3.3s/sample on PAN2025, ~1.5s/sample on HC3 (shorter texts)
- **Token limits**: Truncated texts to 3000 chars to fit context window
- **Determinism**: Used `temperature=0` for reproducible results
- **Key finding**: Zero-shot Qwen2 7B performs **below random** on AI text detection
