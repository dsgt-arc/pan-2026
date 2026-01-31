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
| 001 | Logistic Regression Baseline | Completed | Simple logistic regression model with TF-IDF features |
| 002 | LightGBM + Stylometric Features | Completed | Hybrid model with TF-IDF and hand-crafted linguistic features |
| 003 | DeBERTa-v3 Fine-tuning | **Completed** | Transformer-based model - ROC-AUC: 0.9948 (PAN2026), 0.5939 (HC3) |