"""Appendix Fig. A3, tier effect on writing style, as a dot plot in the paper's style.

Reads the Cohen's d table that plot_cl_variance.py saves
(output_plots/cl_features/cl_tier_effect_heatmap.csv: writers x features, d = high-fit
minus moderate-fit, averaged across jobs) and writes tier_effect_dots.png next to it:
one row per writing-style feature, one dot per writer in its model colour, a black
bar at the mean of the eleven writers. Semantic-fit features are left out, as in
the original heatmap, because their tier effect is near-definitional.

Run after plot_cl_variance.py (generate_all_paper_figures.py does this).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import paper_style as ps
from aggregate_plots import MODEL_DISPLAY, WRITER_COLORS
from plot_feature_regression import FEATURE_CATEGORY_COLORS

ps.apply()
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "output_plots", "cl_features")

GROUPS = {
    "Length & Structure":  ["Word Count", "Sentence Count", "Paragraph Count", "Comma Count"],
    "Language Complexity": ["Avg Word Len", "TTR", "Flesch Ease"],
    "Sentiment & Affect":  ["VADER", "VAD Valence", "VAD Arousal", "VAD Dominance"],
    "Emotions":            ["Joy", "Neutral", "Surprise", "Fear", "Anger", "Disgust", "Sadness"],
}
NICE = {"Avg Word Len": "Average word length", "TTR": "Type-token ratio",
        "Flesch Ease": "Flesch Reading Ease", "VADER": "VADER sentiment"}
WASH_AT = 0.15                       # rows whose mean |d| reaches this get a faint wash
WASH_POS, WASH_NEG = "#e6f1ef", "#f4ece3"   # teal / sand: a direction, not favoured/penalised
XLIM = (-0.65, 0.85)


def main():
    d = pd.read_csv(os.path.join(DATA, "cl_tier_effect_heatmap.csv"), index_col=0)
    key = {v: k for k, v in MODEL_DISPLAY.items()}
    d.index = [key.get(i, i) for i in d.index]
    writers = d.index.tolist()
    feats = [f for fs in GROUPS.values() for f in fs]
    fcol = {f: FEATURE_CATEGORY_COLORS[g] for g, fs in GROUPS.items() for f in fs}

    # features along x, d along y: a short full-width strip
    fig, ax = plt.subplots(figsize=(ps.FULL_WIDTH, 2.4))
    fig.subplots_adjust(left=0.08, right=0.995, top=0.97, bottom=0.37)
    for i, f in enumerate(feats):
        m = d[f].mean()
        if abs(m) >= WASH_AT:
            ax.axvspan(i - 0.45, i + 0.45, color=WASH_POS if m > 0 else WASH_NEG, lw=0, zorder=0)
        ax.plot([i, i], XLIM, color="#ececea", lw=0.5, zorder=1)
        for w in writers:
            ax.scatter(i, d.loc[w, f], s=11, color=WRITER_COLORS[w], lw=0.3, ec="white", zorder=3)
        ax.plot([i - 0.36, i + 0.36], [m, m], color=ps.INK, lw=1.4, zorder=4)
    ax.axhline(0, color=ps.INK_2, lw=0.6, zorder=2)
    for b in np.cumsum([len(v) for v in GROUPS.values()])[:-1]:
        ax.axvline(b - 0.5, color=ps.INK_MUTED, lw=0.4)

    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels([NICE.get(f, f) for f in feats], rotation=40, ha="right",
                       rotation_mode="anchor")
    for t, f in zip(ax.get_xticklabels(), feats):
        t.set_color(fcol[f])
    ax.set_xlim(-0.6, len(feats) - 0.4)
    ax.set_ylim(*XLIM)
    ax.tick_params(axis="x", length=0)
    for s in ("bottom", "top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_ylabel("Cohen's d\n(high-fit − moderate-fit)")
    ax.text(len(feats) - 0.45, XLIM[1], "higher for high-fit ↑", ha="right", va="top",
            fontsize=ps.FS_TINY, color=ps.INK_2)
    ax.text(len(feats) - 0.45, XLIM[0], "higher for moderate-fit ↓", ha="right", va="bottom",
            fontsize=ps.FS_TINY, color=ps.INK_2)

    handles = [plt.Line2D([], [], marker="o", ls="", color=WRITER_COLORS[w], ms=3.5,
                          label=MODEL_DISPLAY[w]) for w in writers]
    handles.append(plt.Line2D([], [], color=ps.INK, lw=1.4, label="Mean of 11 writers"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.0), ncol=6,
               frameon=False, fontsize=ps.FS_TINY, handletextpad=0.3, columnspacing=1.0)

    out = os.path.join(DATA, "tier_effect_dots.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("Saved", out)


if __name__ == "__main__":
    main()
