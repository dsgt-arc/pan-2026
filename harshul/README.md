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
└── pyproject.toml         # Package dependencies
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