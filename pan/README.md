# Harshul's Workspace

PAN 2026 experiments and code.

## Structure

```
.
├── src/                   # Python code
├── experiments/           # Experiment tracking
│   └── NNN-name/
│       ├── proposal.md    # Hypothesis and method
│       ├── results.md     # Outcomes (after completion)
│       └── artifacts/     # Logs, checkpoints
├── notebooks              # Notebooks used in experiments
└── tests                  # Tests used in experiments
```

## Creating a New Experiment

1. Copy the template:
   ```bash
   cp -r experiments/000-template experiments/NNN-name
   ```

2. Edit `proposal.md` with your hypothesis

3. After completion, fill in `results.md`

## Current Experiments

| ID | Name | Status | Description |
|----|------|--------|-------------|
| 000 | template | - | Template for new experiments |
| 001 | Logistic Regression Baseline | **Completed** | Simple logistic regression model with TF-IDF features |
| 002 | LightGBM + Stylometric Features | **Completed** | Hybrid model with TF-IDF and hand-crafted linguistic features |
| 003 | DeBERTa-v3 Fine-tuning | **Completed** | Transformer-based model - ROC-AUC: 0.9948 (PAN2025), 0.5939 (HC3) |
| 004 | External Evaluation Benchmark | **Completed** | OOD evaluation: 3 models × 5 datasets. DeBERTa leads (avg ROC-AUC 0.579) |
| 005 | Zero-Shot Ollama Baseline | **Completed** | Zero-shot LLM detection via Qwen2 7B - ROC-AUC: 0.46 (PAN2025), 0.42 (HC3). Below random. |
| 006 | LoRA Baseline | **Completed** | Qwen2.5-1.5B LoRA fine-tuning - ROC-AUC: 0.9999 (PAN2025), 0.9982 (HC3), 0.8152 (RAID) |
| 007 | Ensemble Baseline | **Completed** | Ensembling LR+DeBERTa+Qwen LoRA does not beat Qwen solo. Rejected. |
| 008 | Homoglyph Augmentation | **Completed** | Homoglyph-only augmentation — marginal HC3 gain but RAID regression. Rejected. |
| 009 | LoRA Replication | **Completed** | Clean replication of 006. ROC-AUC: 0.9993 (PAN2025), 0.9977 (HC3), 0.9974 (RAID) |
| 010 | Obfuscation Augmentation | **Completed** | Homoglyph + synonym augmentation. ROC-AUC: 0.9995 (PAN2025), 0.9985 (HC3), 0.9964 (RAID) |