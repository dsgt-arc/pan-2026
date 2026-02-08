---
description: Design and carry out an experiment.
---

# Experiment Workflow

This workflow guides you through creating, executing, and documenting a complete ML experiment in the PAN 2026 project.

---

## Phase 0: Pre-Planning

Before creating an experiment, ensure you have:

1. **A clear research question or hypothesis** to test
2. **Baseline results** from previous experiments to compare against
3. **Available resources** (compute, data, time estimates)

---

## Phase 1: Create Experiment Folder

### 1.1 Determine Experiment Number

// turbo
```bash
# List existing experiments to find the next available number
ls -la experiments/
```

The next experiment should use the format `NNN-descriptive-name` where NNN is the next sequential number.

### 1.2 Copy the Template

```bash
cp -r experiments/000-template experiments/NNN-experiment-name
```

### 1.3 Create Source Directory

// turbo
```bash
mkdir -p src/NNN-experiment-name
touch src/NNN-experiment-name/__init__.py
```

### 1.4 Create Artifacts Directory

// turbo
```bash
mkdir -p experiments/NNN-experiment-name/artifacts
```

---

## Phase 2: Write the Proposal (`proposal.md`)

Fill out `experiments/NNN-experiment-name/proposal.md` with:

### Required Sections:

1. **Title & Metadata**
   - Status: `proposed` → `in-progress` → `completed`
   - Created date (YYYY-MM-DD format)
   - Author

2. **Hypothesis**
   - Specific, falsifiable statement
   - Include quantitative targets (e.g., "ROC-AUC > 0.70 on HC3")

3. **Background**
   - Why this experiment matters
   - Reference previous experiments with links
   - What gap or question this addresses

4. **Method**
   - **Approach**: High-level description
   - **Setup**: Table with Data, Compute, Dependencies, Code paths
   - **Procedure**: Numbered step-by-step plan
   - **Variables**: Independent, Dependent, Controlled

5. **Evaluation**
   - **Metrics**: Table of metrics with descriptions
   - **Baseline**: Comparison numbers from previous experiments
   - **Success Criteria**: Specific thresholds for confirm/reject

6. **Limitations**
   - Known constraints and assumptions

---

## Phase 3: Update `pyproject.toml`

Update `experiments/NNN-experiment-name/pyproject.toml`:

```toml
[project]
name = "experiment-NNN-experiment-name"
version = "0.1.0"
description = "Brief description of the experiment"
requires-python = ">=3.10"

# Experiment-specific dependencies (core deps inherited from root)
dependencies = [
    "package1",    # Comment explaining why
    "package2",    # Comment explaining why
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

---

## Phase 4: Create `tasks.md`

Create `experiments/NNN-experiment-name/tasks.md` with:

1. **Summary**: One-paragraph description of the experiment

2. **Datasets Table**: List all datasets with sample counts

3. **Phased Task Lists**:
   - Use checkboxes: `- [ ]` for pending, `- [x]` for complete
   - Group into logical phases (Setup, Implementation, Evaluation, Analysis)
   - Add ✅ emoji to phase headers when complete

4. **Artifacts Table**: Expected output files

5. **Success Criteria Table**: Track progress toward goals

### Example Structure:
```markdown
## Phase 1: Environment Setup
- [ ] Task 1
- [ ] Task 2

## Phase 2: Implementation
- [ ] Task 3
- [ ] Task 4
```

---

## Phase 5: Implement the Experiment

### 5.1 Write Code in `src/NNN-experiment-name/`

Typical files include:
- `__init__.py` - Package marker
- `infer.py` or `train.py` - Main execution script
- `evaluate.py` - Metrics computation
- `prompts.py` or `features.py` - Task-specific components
- `run_experiment.py` - End-to-end runner

### 5.2 Reuse Common Utilities

Import from previous experiments when possible:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "003-deberta"))
from metrics import get_all_metrics
```

### 5.3 Save Outputs to Artifacts

All outputs should go to `experiments/NNN-experiment-name/artifacts/`:
- `*_predictions.json` - Raw predictions
- `*_results.json` - Computed metrics
- `*.png` - Visualizations
- `*.md` - Generated tables

### 5.4 Update `tasks.md` as You Go

Mark tasks complete as you finish them:
```markdown
- [x] Create inference pipeline ✅
```

---

## Phase 6: Run the Experiment

### 6.1 Test on Small Sample First

Always validate with a small sample before full run:
```bash
python3 src/NNN-experiment-name/run_experiment.py --max-samples 10
```

### 6.2 Execute Full Experiment

```bash
python3 src/NNN-experiment-name/run_experiment.py
```

### 6.3 Monitor Progress

For long-running experiments, check status periodically and log any issues.

---

## Phase 7: Write Results (`results.md`)

Fill out `experiments/NNN-experiment-name/results.md` with:

### Required Sections:

1. **Overview**
   - What was evaluated
   - Total evaluations run

2. **Summary Table**
   - Key metric comparison across conditions
   - Highlight best results with **bold**

3. **Visualizations**
   - Link to charts/graphs in artifacts
   - Use relative paths: `![Alt](artifacts/chart.png)`

4. **Key Findings**
   - Numbered list of main observations
   - Include both successes and failures

5. **Recommendations**
   - Actionable next steps

6. **Artifacts Table**
   - List all generated files with descriptions

---

## Phase 8: Update Experiment Tracker

Update `experiments/README.md` or the main project README:

### Add to Experiments Table:

```markdown
| ID | Name | Status | Description |
|----|------|--------|-------------|
| NNN | Experiment Name | **Completed** | Brief description with key result |
```

### Update Status:

- `proposed` - Proposal written, not started
- `in-progress` - Currently running
- **Completed** - Finished with results
- `abandoned` - Stopped (document why in results.md)

---

## Phase 9: Final Cleanup

1. **Update `proposal.md` status** to `completed`

2. **Ensure all tasks in `tasks.md`** are marked complete

3. **Commit all changes** with descriptive message:
   ```bash
   git add experiments/NNN-experiment-name/ src/NNN-experiment-name/
   git commit -m "Complete experiment NNN: Brief description of findings"
   ```

4. **Archive large artifacts** if needed (checkpoints, raw data)

---

## Quick Reference: File Checklist

| File | Location | Purpose |
|------|----------|---------|
| `proposal.md` | `experiments/NNN/` | Hypothesis and method |
| `tasks.md` | `experiments/NNN/` | Task tracking |
| `results.md` | `experiments/NNN/` | Final outcomes |
| `pyproject.toml` | `experiments/NNN/` | Dependencies |
| `artifacts/` | `experiments/NNN/` | Output files |
| `*.py` | `src/NNN/` | Implementation code |

---

## Standard Metrics

For all experiments, compute these 5 metrics:

| Metric | Description | Goal |
|--------|-------------|------|
| ROC-AUC | Ranking ability (threshold-independent) | Higher is better |
| Brier Score | Probability calibration | Lower is better |
| F1 Score | Balanced precision/recall | Higher is better |
| C@1 | PAN metric with abstention | Higher is better |
| F0.5u | PAN metric penalizing false negatives | Higher is better |

---

## Tips

1. **Start with the hypothesis** - Everything else flows from this
2. **Compare against baselines** - Always include previous experiment results
3. **Document as you go** - Update tasks.md in real-time
4. **Save intermediate results** - In case of crashes or interruptions
5. **Be honest about failures** - Negative results are still valuable