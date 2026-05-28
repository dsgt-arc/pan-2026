---
description: Generate publication-ready figures/plots matching the matplotlib paper style.
---

# Matplotlib Paper-Style Plotting Workflow

This workflow guides you through generating high-quality, publication-ready figures and charts using Matplotlib and Seaborn, following the clean aesthetic of AI safety research papers (e.g., Betley, Slocum et al.).

Use this workflow whenever a plotting request is made, when results need to be visualized, or when designing paper figures.

---

## 🎨 Aesthetic Guidelines

- **No Titles**: Do not include titles on the figure itself. Explanatory text belongs in the caption or surrounding context.
- **Semantic Colors**: Colors represent meaning, not arbitrary sequencing. Never use default rainbow cycles.
  - ❤️ **Treatment**: `#E24B4A` (Red) — main experimental condition
  - 💚 **Control A**: `#1D9E75` (Teal) — first control
  - 💙 **Control B**: `#378ADD` (Blue) — second control
  - 🩶 **Baseline**: `#888780` (Gray) — baseline (e.g., zero-shot, no-finetune)
  - 💛 **Highlight**: `#BA7517` (Amber) — highlights, ablations, or special comparisons
- **Sizing & Layout**: Standard figure size is `5 x 3.5` inches. Use `constrained_layout=True` rather than `tight_layout()`.
- **Spines & Grids**: Remove top and right spines. Use subtle horizontal grid lines only (`alpha=0.4`, `linewidth=0.5`).
- **Legends**: Minimalist legends with `frameon=False` and clean, short labels.
- **Double Save**: Always save both a **PDF** (for LaTeX inclusion) and a **PNG** (for quick preview/sharing), both with `transparent=True` and `bbox_inches="tight"`.

---

## 🛠️ Implementation Steps

Whenever asked to create or update a plot, follow these steps:

### 1. Identify Data and Target Path
- Locate the source data (e.g., a results JSON/CSV file, dictionary, or raw description).
- Determine where the plot will be saved. Usually, this is in an `artifacts/` folder, a specific experiment's subfolder, or a temporary output path.

### 2. Set Up the Script / Code
Use the following standardized Python skeleton to load your data and build the figure:

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set standard publication-ready contexts
sns.set_style("whitegrid")
sns.set_context("paper")

# Define the semantic palette
PALETTE = {
    "treatment": "#E24B4A",   # red
    "control_a": "#1D9E75",   # teal
    "control_b": "#378ADD",   # blue
    "baseline":  "#888780",   # gray
    "highlight": "#BA7517",   # amber
}

# Create standard-sized figure
fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)

# ----------------- PLOT LOGIC HERE -----------------
# (Use one of the standard plot types below)
# ---------------------------------------------------

# Clean up layout and styling
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, linewidth=0.5, alpha=0.4)
ax.xaxis.grid(False)
ax.legend(frameon=False, fontsize=10)
ax.set_xlabel("X-Axis Label Here", fontsize=11)
ax.set_ylabel("Y-Axis Label Here", fontsize=11)
ax.tick_params(labelsize=10)

# Save in high resolution and vector formats
plt.savefig("figure.pdf", bbox_inches="tight", dpi=150, transparent=True)
plt.savefig("figure.png", bbox_inches="tight", dpi=150, transparent=True)
plt.close()
```

### 3. Choose the Right Plot Type

#### A. Line Plot with Error Bars (Most Common)
For plotting progress, epochs, samples, or parameters. Use bootstrapped confidence intervals for uncertainty.

*Bootstrapped CI Helper:*
```python
def bootstrap_ci(data, n_boot=1000, ci=95):
    boots = [np.mean(np.random.choice(data, len(data))) for _ in range(n_boot)]
    lo = np.percentile(boots, (100 - ci) / 2)
    hi = np.percentile(boots, 100 - (100 - ci) / 2)
    return np.mean(data), hi - np.mean(data)  # returns (mean, yerr)
```

*Plotting Code:*
```python
ax.errorbar(x, y, yerr=err, color=PALETTE["treatment"],
            linewidth=1.5, marker="o", markersize=5,
            capsize=4, elinewidth=1, alpha=0.8, label="treatment")
```

#### B. Bar Plot
For comparing final evaluation metrics across distinct models or settings.

```python
ax.bar(x, y, color=[PALETTE["baseline"], PALETTE["control_a"], PALETTE["treatment"]], width=0.6, alpha=0.85)
```

#### C. Scatter Plot with Jitter
For individual seed runs, data point clouds, or fine-grained per-sample results.

```python
jitter = np.random.normal(0, 0.05, size=len(y))
ax.scatter(x + jitter, y, color=PALETTE["treatment"], s=20, alpha=0.5, label="treatment")
```

---

## 🚀 Execution & Verification

1. **Run the script**: Run the python script to generate the files.
2. **Verify output files**: Check that both the `.pdf` and `.png` files are saved successfully in the target directory.
3. **Present the results**: Show the generated PNG and describe the key takeaways of the plot, pointing to the code block used to generate it.
