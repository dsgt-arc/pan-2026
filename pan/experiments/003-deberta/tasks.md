# Implementation Plan: DeBERTa-v3 Fine-tuning (CPU, Long Documents)

## Summary

This experiment fine-tunes **DeBERTa-v3** for AI text detection. Key constraints:
1. **Long documents**: Texts up to **6,500 tokens** must be supported (well beyond the 512-token limit of standard BERT models).
2. **CPU-only training**: No GPU is available; training and inference will run on CPU.

To handle these constraints, we will implement a **chunking strategy** (splitting long documents into overlapping 512-token windows) and aggregate predictions. We will also optimize for CPU performance using gradient checkpointing and small batch sizes.

All code will be placed in `harshul/src/003-deberta/`.

---

## Detailed Tasks

### Phase 1: Project Setup

- [x] Create the directory `harshul/src/003-deberta/`.
- [x] Create `__init__.py`.
- [x] Verify dependencies (`torch`, `transformers`, `datasets`, `accelerate`, `sentencepiece`) are installed.
- [x] Copy reusable modules from experiment 001 (`data_loader.py`, `metrics.py`).

### Phase 2: Long Document Handling Strategy

- [x] Create `chunking.py` to handle documents longer than 512 tokens.
- [x] Implement overlapping sliding window chunking:
  - Chunk size: 512 tokens.
  - Stride: 256 tokens (50% overlap).
  - For a 6,500-token document, this yields ~25 chunks.
- [x] Implement aggregation strategies for chunk predictions:
  - **Mean pooling**: Average probabilities across all chunks.
  - (Optional) **Max pooling**: Take the maximum probability.
- [x] Add tests to verify chunking on edge cases (short texts, exactly 512 tokens, etc.).

### Phase 3: Dataset & Tokenization

- [x] Create `dataset.py` to wrap PAN2025 data in HuggingFace `Dataset` format.
- [x] Tokenize using `microsoft/deberta-v3-base` tokenizer.
- [x] For training: use standard 512-token truncation (long docs will be chunked at inference).
- [x] For inference: apply chunking and return all chunks per document.

### Phase 4: CPU-Optimized Training Configuration

- [x] Create `train.py` as the main entry point.
- [x] Configure HuggingFace `TrainingArguments` for CPU:
  - `no_cuda=True`
  - `per_device_train_batch_size=2` (very small due to memory)
  - `gradient_accumulation_steps=8` (effective batch size of 16)
  - `gradient_checkpointing=True` (trade compute for memory)
  - `fp16=False` (CPU does not support mixed precision)
  - `learning_rate=2e-5`
  - `num_train_epochs=3`
  - `warmup_steps=500`
- [x] Use `Trainer` API for simplicity.
- [x] Implement checkpoint saving to `artifacts/`.
- [x] Train model using Google Colab GPU.
- [x] Upload trained model to Hugging Face Hub.

### Phase 4.5: Model Retrieval from Hugging Face

**Model hosted at:** [`hersheys-baklava/deberta-pan2025`](https://huggingface.co/hersheys-baklava/deberta-pan2025)

- [x] Create `load_model.py` to pull the model from HuggingFace.
- [x] Implement model loading:
  ```python
  from transformers import AutoModelForSequenceClassification, AutoTokenizer
  
  model = AutoModelForSequenceClassification.from_pretrained("hersheys-baklava/deberta-pan2025")
  tokenizer = AutoTokenizer.from_pretrained("hersheys-baklava/deberta-pan2025")
  ```
- [x] Add option to cache model locally in `artifacts/model/` directory.
- [x] Test model loading and verify inference works.

### Phase 5: Internal Evaluation (PAN2025 Validation) ✅ COMPLETE

- [x] Create `evaluate.py`.
- [x] Load model from HuggingFace: `hersheys-baklava/deberta-pan2025` (with local caching)
- [x] Load PAN2025 validation data.
- [x] For each validation sample:
  - Chunk the text using sliding window (512 tokens, 256 stride).
  - Run inference on each chunk.
  - Aggregate predictions (mean pooling).
- [x] Compute metrics:
  - ROC-AUC: **0.9948**
  - Brier Score: 0.1106
  - C@1: 0.8654
  - F1 Score: 0.8834
  - F0.5u: 0.9498
- [x] Save metrics to `artifacts/val_metrics.json`.
- [ ] Generate confusion matrix and ROC curve plots.

### Phase 6: External Evaluation (HC3 - Out-of-Distribution) ✅ COMPLETE

- [x] Create `evaluate_external.py`.
- [x] Load HC3 dataset (external, out-of-distribution test).
- [x] Apply the same chunking + aggregation strategy to HC3 data.
- [x] Compute all metrics:
  - ROC-AUC: **0.5939** (improved from LR baseline 0.5520)
  - Brier Score: 0.4939
  - C@1: 0.5006
  - F1 Score: 0.0024 ⚠️ (threshold miscalibration)
  - F0.5u: 0.0059
- [x] Save metrics to `artifacts/external_metrics.json`.
- [x] Compare performance to in-distribution (PAN2025) results.
- [ ] Analyze failure cases and error patterns.

### Phase 7: Results Documentation ✅ COMPLETE

- [x] Update `harshul/experiments/003-deberta/results.md` with final metrics.
- [x] Create comparison table:
  | Model | PAN2025 ROC-AUC | HC3 ROC-AUC | Notes |
  |-------|-----------------|-------------|-------|
  | LR Baseline | 0.9953 | 0.5520 | Baseline |
  | LightGBM | 0.9980 | 0.3356 | Inverted on OOD |
  | DeBERTa | 0.9948 | 0.5939 | +4.2% OOD improvement |
- [x] Compare DeBERTa's OOD generalization to LR and LightGBM.
- [x] Document the impact of chunking strategy on performance.
- [x] Add training details (epochs, loss curves, training time).
- [ ] Update `harshul/README.md` experiments table.
- [x] Document lessons learned and recommendations for future work.

### Phase 8: Simplified Jupyter Notebook

- [x] Create `harshul/notebooks/003-deberta/` directory.
- [x] Create `deberta_training.ipynb` that mirrors the training loop:
  - Load and tokenize PAN2025 data.
  - Fine-tune DeBERTa-v3 with CPU-optimized settings.
  - Run evaluation on validation set.
- [x] Add markdown explanations for each step (educational for new readers).
- [x] Include visualization of training loss and validation metrics.
- [x] Ensure the notebook can be run end-to-end independently.

---

## File Structure

```
harshul/src/003-deberta/
├── __init__.py
├── chunking.py          # Sliding window chunking for long documents
├── dataset.py           # Tokenization and HF Dataset wrapping
├── train.py             # Fine-tuning script (CPU-optimized)
├── evaluate.py          # Evaluation on validation set
└── evaluate_external.py # Evaluation on HC3 external dataset

harshul/notebooks/003-deberta/
└── deberta_training.ipynb  # Simplified end-to-end notebook
```

---

## Long Document Strategy

For a document with `N` tokens (up to 6,500):

```
Chunk 1: tokens [0, 512)
Chunk 2: tokens [256, 768)
Chunk 3: tokens [512, 1024)
...
```

At inference, each chunk produces a probability `p_i`. The final prediction is:
```
P(AI) = mean(p_1, p_2, ..., p_k)
```

---

## CPU Training Considerations

| Setting | Value | Rationale |
|---------|-------|-----------|
| `no_cuda` | True | Force CPU execution |
| `batch_size` | 2 | Small to fit in RAM |
| `grad_accum` | 8 | Simulate larger effective batch |
| `grad_checkpoint` | True | Reduce memory at cost of speed |
| `epochs` | 3 | Standard for fine-tuning |

**Expected training time**: ~10-24 hours on a modern CPU (depends on cores).

---

## Success Criteria

| Metric | LR Baseline (HC3) | Target (DeBERTa) |
|--------|-------------------|------------------|
| ROC-AUC | 0.5520            | ≥ 0.70           |
| F1      | 0.6079            | ≥ 0.75           |

---

## Next Steps

1. ~~Implement chunking logic and verify with unit tests.~~ ✅
2. ~~Set up the tokenization pipeline.~~ ✅
3. ~~Train model (completed on Colab GPU).~~ ✅
4. ~~Upload model to Hugging Face Hub.~~ ✅
5. ~~Create `load_model.py` to pull model from HuggingFace.~~ ✅
6. ~~Run evaluation on PAN2025 validation set.~~ ✅ (ROC-AUC: 0.9948)
7. ~~Run evaluation on HC3 (out-of-distribution test).~~ ✅ (ROC-AUC: 0.5939)
8. **Generate plots** (confusion matrix, ROC curve). *(optional)*
9. ~~Document results in `results.md` and compare to baselines.~~ ✅
10. ~~Update README with final experiment summary.~~ ✅

## 🎉 Experiment Complete!

**Key Finding**: DeBERTa achieves +4.2% OOD improvement over LR baseline (0.5939 vs 0.5520 ROC-AUC on HC3), but does not meet the target of ≥0.70. Threshold calibration is needed for practical deployment.
