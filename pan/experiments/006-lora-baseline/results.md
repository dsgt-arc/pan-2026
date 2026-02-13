# Results: Qwen2.5-1.5B LoRA Fine-tuning for AI Text Detection

**Date Completed**: 2026-02-13
**Author**: harshul

## Summary

Qwen2.5-1.5B with LoRA adapters achieves near-perfect in-distribution performance (ROC-AUC: 0.9999) and **dramatically improves out-of-distribution generalization** compared to all previous models. HC3 ROC-AUC jumps from 0.5939 (DeBERTa, previous best) to 0.9982, and RAID improves from 0.5879 to 0.8152 — a step change that validates the hypothesis that larger pretrained language models with parameter-efficient fine-tuning generalize better to unseen distributions.

**Verdict**: Confirmed. Qwen LoRA substantially outperforms all prior baselines on both in-distribution and out-of-distribution benchmarks.

## Observations

### Full Cross-Reference: All Models × All Metrics × All Datasets

|         |          | PAN2025 |    HC3 |   RAID | Avg OOD |
|---------|----------|--------:|-------:|-------:|--------:|
| **LR (001)** | ROC-AUC | 0.9953 | 0.5517 | 0.5727 | 0.5622 |
|         | Brier    | 0.0282 | 0.2706 | 0.2675 | 0.2691 |
|         | F1       | 0.9772 | 0.6079 | 0.6334 | 0.6207 |
|         | C@1      | 0.9705 | 0.5220 | 0.5440 | 0.5330 |
|         | F0.5u    | 0.9732 | 0.5487 | 0.5638 | 0.5563 |
| **LightGBM (002)** | ROC-AUC | 0.9964 | 0.3356 | 0.3398 | 0.3377 |
|         | Brier    | 0.0199 | 0.4415 | 0.4507 | 0.4461 |
|         | F1       | 0.9791 | 0.5915 | 0.5889 | 0.5902 |
|         | C@1      | 0.9730 | 0.4555 | 0.4500 | 0.4528 |
|         | F0.5u    | 0.9779 | 0.5144 | 0.5090 | 0.5117 |
| **DeBERTa (003)** | ROC-AUC | 0.9948 | 0.5939 | 0.5879 | 0.5909 |
|         | Brier    | 0.1106 | 0.4939 | 0.4896 | 0.4918 |
|         | F1       | 0.8834 | 0.0024 | 0.0000 | 0.0012 |
|         | C@1      | 0.8654 | 0.5006 | 0.5060 | 0.5033 |
|         | F0.5u    | 0.9498 | 0.0059 | 0.0000 | 0.0030 |
| **Qwen LoRA (006)** | ROC-AUC | **0.9999** | **0.9982** | **0.8152** | **0.9067** |
|         | Brier    | **0.0019** | **0.0112** | **0.1491** | **0.0802** |
|         | F1       | **0.9981** | **0.9868** | **0.8182** | **0.9025** |
|         | C@1      | **0.9975** | **0.9869** | **0.8460** | **0.9165** |
|         | F0.5u    | **0.9984** | **0.9947** | **0.9176** | **0.9562** |

### Training Details

- **Model**: `Qwen/Qwen2.5-1.5B` (1.54B parameters)
- **Method**: LoRA (rank 16, alpha 32, dropout 0.1, target modules: q_proj, v_proj)
- **Training Platform**: Google Colab (NVIDIA L4 GPU)
- **Epochs**: 3
- **Batch Size**: 4 (with gradient accumulation of 4, effective batch 16)
- **Learning Rate**: 2e-5
- **Max Sequence Length**: 512 tokens (with sliding-window chunking, stride 256)
- **Trainable Parameters**: ~4.7M / 1.54B (0.3%)
- **Adapter hosted at**: [hersheys-baklava/qwen-lora-pan2026](https://huggingface.co/hersheys-baklava/qwen-lora-pan2026)

### Evaluation Details

- **Inference**: GPU (NVIDIA L4), bfloat16 precision
- **Runtime**: ~8 minutes for 7,273 total samples (PAN2025 val: 3,589 + HC3: 1,684 + RAID: 2,000)
- **Throughput**: ~10-27 samples/sec depending on document length
- **RAID subsampling**: 2,000 samples (1,000 human + 1,000 AI), streamed from HuggingFace

## Analysis

### Key Findings

1. **OOD generalization breakthrough**: The Qwen LoRA model achieves an average OOD ROC-AUC of 0.9067, representing a +0.3158 improvement over the previous best (DeBERTa at 0.5909). This is the first model in the project to achieve above-chance OOD performance on a meaningful scale.

2. **HC3 is near-solved**: ROC-AUC of 0.9982 on HC3 Wiki indicates the model can reliably distinguish human-written from ChatGPT-generated text, even on data it has never seen during training. This is a dramatic improvement from all prior models (which hovered around 0.55).

3. **RAID remains harder**: ROC-AUC of 0.8152 on RAID is strong but noticeably lower than HC3. RAID includes outputs from multiple generators (GPT-3.5, GPT-4, Llama, Mistral, etc.) across multiple domains, making it a more diverse and challenging benchmark. The gap suggests room for improvement on multi-generator detection.

4. **Calibration is excellent**: Unlike DeBERTa (which had near-zero F1 on HC3 due to threshold miscalibration), Qwen LoRA produces well-calibrated probabilities. F1 scores of 0.99 (HC3) and 0.82 (RAID) at the default 0.5 threshold confirm the model's predictions are directly usable without threshold tuning.

5. **Scaling works**: Moving from DeBERTa-v3-base (86M params) to Qwen2.5-1.5B (1.54B params) with LoRA yielded massive OOD gains while training only 0.3% of parameters. The pretrained knowledge in the larger model provides substantially better generalization.

### Why Qwen LoRA Succeeded Where Others Failed

Prior models (LR, LightGBM, DeBERTa) all achieved strong PAN2025 performance but failed on OOD data because they learned dataset-specific artifacts rather than genuine human-vs-AI writing patterns. Qwen2.5-1.5B succeeds because:

- **Richer pretraining**: 1.5B parameters pretrained on diverse internet text gives the model a deep understanding of natural language patterns, making it easier to identify subtle AI-generated text artifacts.
- **LoRA preserves pretrained knowledge**: By only updating a small adapter (0.3% of params), the model retains its broad language understanding while learning the binary classification task.
- **Sequence classification head**: The classification head on top of a frozen-ish LLM backbone acts as a lightweight discriminator that leverages the full representational capacity of the base model.

### Confounders

- RAID was subsampled to 2,000 samples (from 400K+). Results may vary with different subsamples, though the random seed was fixed for reproducibility.
- TuringBench, M4, and Ghostbuster were excluded from this evaluation due to dataset loading issues. Future work should include these for a more complete OOD picture.
- HC3 Wiki contains only ChatGPT-generated text. Performance on other generators (GPT-4, Claude, Llama) is captured partially by RAID but deserves dedicated analysis.

## Artifacts

| Type | Location |
|------|----------|
| Adapter | [hersheys-baklava/qwen-lora-pan2026](https://huggingface.co/hersheys-baklava/qwen-lora-pan2026) |
| Evaluation Results | `evaluation_results.json` (on HuggingFace repo) |
| Training Notebook | `notebooks/006-lora-baseline/lora_training.ipynb` |
| Evaluation Notebook | `notebooks/006-lora-baseline/lora_evaluation.ipynb` |

## Next Steps

- [ ] Evaluate on TuringBench, M4, and Ghostbuster once dataset loading is resolved
- [ ] Analyze per-generator performance on RAID (which AI models are hardest to detect?)
- [ ] Test with higher LoRA rank (32, 64) to see if more adapter capacity improves RAID
- [ ] Investigate training on mixed-domain data (PAN2025 + HC3 subset) for further OOD gains
- [ ] Compare with full fine-tuning of a smaller model (e.g., Qwen2.5-0.5B) to understand the scaling contribution vs. the LoRA contribution

**Leads to**: Qwen LoRA establishes a strong baseline for the PAN 2026 submission. Next experiments should focus on multi-domain training and evaluating on the remaining OOD benchmarks.