"""
Generate visualizations for external evaluation results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
ARTIFACTS_DIR = Path("/Users/tanish/Desktop/CLEF2026/pan-2026/harshul/experiments/004-external_evaluation/artifacts")

# Load results
with open(ARTIFACTS_DIR / "all_metrics.json") as f:
    results = json.load(f)

# Extract ROC-AUC data
datasets = list(results["results"].keys())
models = ["DeBERTa", "LR", "LightGBM"]

# Create ROC-AUC matrix
roc_auc_matrix = []
for dataset in datasets:
    row = []
    for model in models:
        if model in results["results"][dataset]:
            row.append(results["results"][dataset][model]["ROC-AUC"])
        else:
            row.append(0)
    roc_auc_matrix.append(row)

roc_auc_matrix = np.array(roc_auc_matrix)

# 1. Heatmap: ROC-AUC by model x dataset
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(roc_auc_matrix, cmap='RdYlGn', aspect='auto', vmin=0.3, vmax=0.7)

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("ROC-AUC", rotation=-90, va="bottom", fontsize=12)

# Labels
ax.set_xticks(np.arange(len(models)))
ax.set_yticks(np.arange(len(datasets)))
ax.set_xticklabels(models, fontsize=12, fontweight='bold')
ax.set_yticklabels([d.replace('_', ' ').title() for d in datasets], fontsize=11)

# Add text annotations
for i in range(len(datasets)):
    for j in range(len(models)):
        val = roc_auc_matrix[i, j]
        color = "white" if val < 0.45 else "black"
        text = ax.text(j, i, f"{val:.3f}", ha="center", va="center", 
                      color=color, fontsize=12, fontweight='bold')

ax.set_title("ROC-AUC: External Evaluation Benchmark", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(ARTIFACTS_DIR / "roc_heatmap.png", dpi=150, bbox_inches='tight')
print(f"✅ Saved: {ARTIFACTS_DIR / 'roc_heatmap.png'}")

# 2. Bar chart: Average OOD performance per model
fig, ax = plt.subplots(figsize=(8, 5))

avg_roc = roc_auc_matrix.mean(axis=0)
colors = ['#2ecc71', '#3498db', '#e74c3c']
bars = ax.bar(models, avg_roc, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels
for bar, val in zip(bars, avg_roc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{val:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

# Styling
ax.set_ylabel("Average ROC-AUC", fontsize=12, fontweight='bold')
ax.set_xlabel("Model", fontsize=12, fontweight='bold')
ax.set_title("Average OOD Performance by Model", fontsize=14, fontweight='bold', pad=15)
ax.set_ylim(0, 0.8)
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='Random (0.5)')
ax.legend(loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(ARTIFACTS_DIR / "avg_performance_bar.png", dpi=150, bbox_inches='tight')
print(f"✅ Saved: {ARTIFACTS_DIR / 'avg_performance_bar.png'}")

print("\n=== Visualizations Complete ===")
