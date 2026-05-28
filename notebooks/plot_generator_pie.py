import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from pathlib import Path

# ── Load data ────────────────────────────────────────────────────────────────
TRAIN_PATH = Path("../pan/src/pan-data/train.jsonl")

def load_jsonl(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return pd.DataFrame(data)

train_df = load_jsonl(TRAIN_PATH)
train_df["word_count"] = train_df["text"].str.split().str.len()

# ── Generator order & percentages (AI only, sorted by count desc) ─────────────
TOTAL_TRAIN = len(train_df)
ai_df = train_df[train_df["label"] == 1]

gen_counts = ai_df["model"].value_counts()
names  = gen_counts.index.tolist()          # sorted most → least
counts = gen_counts.values.tolist()
n      = len(names)
pcts   = [c / TOTAL_TRAIN * 100 for c in counts]

# Average word count per model (AI rows only), in same order as names
avg_wc = ai_df.groupby("model")["word_count"].mean()
avg_wc_ordered = [avg_wc[name] for name in names]

# ── Color palette ─────────────────────────────────────────────────────────────
cmap   = plt.colormaps["plasma"]
colors = [cmap(0.92 - 0.85 * i / (n - 1)) for i in range(n)]

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 13))
gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 3.5], wspace=0.3)
ax_left  = fig.add_subplot(gs[0])
ax_right = fig.add_subplot(gs[1])

fig.patch.set_facecolor("white")
for ax in (ax_left, ax_right):
    ax.set_facecolor("white")

# ── LEFT: vertical stacked bar ────────────────────────────────────────────────
BAR_W  = 0.5
bottom = 0.0

for i in range(n):
    pct = pcts[i]
    ax_left.bar(0, pct, bottom=bottom, width=BAR_W,
                color=colors[i], edgecolor="white", linewidth=1.0)

    seg_mid = bottom + pct / 2
    r, g, b = colors[i][:3]
    tc = "black" if (0.299*r + 0.587*g + 0.114*b) > 0.55 else "white"

    fs = max(8.0, min(12.0, pct * 2.2))
    ax_left.text(0, seg_mid, f"{names[i]}  —  {pct:.1f}%",
                 ha="center", va="center",
                 fontsize=fs, fontweight="bold", color=tc)
    bottom += pct

ax_left.set_xlim(-0.4, 0.4)
ax_left.set_ylim(0, bottom + 0.1)
ax_left.axis("off")
ax_left.set_title("Generator\nDistribution", fontsize=16,
                  fontweight="bold", color="black", pad=10)

# ── RIGHT: horizontal bar chart, avg word count ───────────────────────────────
# Reverse so top of right chart aligns with top of stacked bar (least → most)
names_r   = list(reversed(names))
colors_r  = list(reversed(colors))
avg_wc_r  = list(reversed(avg_wc_ordered))
y_pos     = np.arange(n)

bars = ax_right.barh(y_pos, avg_wc_r, color=colors_r,
                     edgecolor="white", linewidth=0.8, height=0.75)

# Value labels at end of each bar
for j, val in enumerate(avg_wc_r):
    ax_right.text(val + 8, j, f"{val:.0f}", va="center",
                  fontsize=10, color="#333333")

ax_right.set_yticks(y_pos)
ax_right.set_yticklabels(names_r, fontsize=11, fontweight="bold")
ax_right.set_xlabel("Average Word Count", fontsize=13, color="black", labelpad=8)
ax_right.tick_params(axis="x", colors="black", labelsize=11)
ax_right.tick_params(axis="y", colors="black")
ax_right.spines["top"].set_visible(False)
ax_right.spines["right"].set_visible(False)
ax_right.spines["left"].set_visible(False)
ax_right.set_title("Avg. Generation Length (Words)", fontsize=16,
                   fontweight="bold", color="black", pad=10)

# Light vertical grid
ax_right.xaxis.grid(True, color="#dddddd", linewidth=0.8, linestyle="--")
ax_right.set_axisbelow(True)

# ── Shared title ──────────────────────────────────────────────────────────────
fig.suptitle("AI Generator Distribution & Average Generation Length — PAN Training Data",
             fontsize=18, fontweight="bold", color="black", y=0.98)

plt.tight_layout()
plt.savefig("generator_combined.png", dpi=300, bbox_inches="tight", facecolor="white")
print("Saved generator_combined.png")
