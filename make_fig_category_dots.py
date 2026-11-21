"""Appendix D per-writer, per-evaluator dot plots, redrawn in the paper's figure style.

Reads the CSVs that plot_feature_regression.plot_category_dotplot saves
(<name>_data_dots.csv, <name>_data_box.csv in preference_by_category/) and writes
cd_<name>.png next to them, for:

  semantic_fit_clonly   Cosine Similarity to Job Ad, cover letter only
  semantic_fit_cvcl     both cosine features, CV + cover letter
  language_complexity   Average Word Length, Type-Token Ratio, Flesch Reading Ease

Per feature block: left, one row per writer (sorted by the mean over evaluators),
one dot per evaluator (style x that evaluator's Ridge coefficient), the writer's own
evaluation drawn solid with a dark ring, the mean over evaluators as a short bar;
right, the distribution of the raw feature value in that writer's letters
(10th-90th percentile line, interquartile bar, white median).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import paper_style as ps
from plot_feature_regression import FEATURE_CATEGORY_COLORS as CAT, FEATURE_LABELS

ps.apply()
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "output_plots", "cl_features", "preference_by_category")
DISPLAY = {
    "gpt-4o-mini": "GPT-4o mini", "gpt-5-mini": "GPT-5 mini", "gpt-5": "GPT-5",
    "gemini-3-flash-preview": "Gemini 3 Flash", "gemini-3.5-flash": "Gemini 3.5 Flash",
    "claude-haiku-4-5": "Claude Haiku 4.5", "claude-4.6-sonnet": "Claude Sonnet 4.6",
    "deepseek-v3.2": "DeepSeek V3.2", "deepseek-v4-flash": "DeepSeek V4 Flash",
    "deepseek-r1-8b": "DeepSeek R1 8B", "llama3.1-8b": "Llama 3.1 8B",
}
FEAT_CAT = {"job_cosine_sim": "Job-Ad Fit", "cv_cosine_sim": "CV Consistency",
            "avg_word_length": "Language Complexity", "ttr": "Language Complexity",
            "flesch_reading_ease": "Language Complexity"}
ROW_H = 0.118     # inches per writer row
HEAD_H = 0.26     # inches for each feature's header


def draw(name):
    dots = pd.read_csv(os.path.join(DATA, f"{name}_data_dots.csv"))
    box = pd.read_csv(os.path.join(DATA, f"{name}_data_box.csv"))
    feats = list(dict.fromkeys(dots.feature))
    evaluators = list(dict.fromkeys(dots.evaluator))
    nw = dots.writer.nunique()
    vmax = dots.contribution.abs().max() * 1.08

    heights = [HEAD_H + nw * ROW_H] * len(feats)
    H = sum(heights) + 0.55 * (len(feats) - 1) + 0.47
    fig = plt.figure(figsize=(ps.FULL_WIDTH, H))
    # margins in inches: the bottom one holds the last x-label, the legend goes below it
    gs = fig.add_gridspec(len(feats), 2, width_ratios=[2.3, 1], height_ratios=heights,
                          wspace=0.06, hspace=0.55 * len(feats) / sum(heights),   # 0.55 in between blocks
                          bottom=0.42 / H, top=1 - 0.05 / H)
    for k, f in enumerate(feats):
        d = dots[dots.feature == f]
        order = d.groupby("writer").contribution.mean().sort_values(ascending=False).index.tolist()
        ax, axb = fig.add_subplot(gs[k, 0]), fig.add_subplot(gs[k, 1])
        for i, w in enumerate(order):
            ax.axhline(i, color="#eceae6", lw=0.4, zorder=0)
            axb.axhline(i, color="#eceae6", lw=0.4, zorder=0)
            r = d[d.writer == w]
            for _, e in r.iterrows():
                own = e.evaluator == w
                ax.scatter(e.contribution, i, s=13 if own else 7,
                           color=ps.MODEL_SHADES[e.evaluator], alpha=1 if own else 0.4,
                           ec=ps.INK if own else "none", lw=0.6 if own else 0, zorder=4 if own else 3)
            m = r.contribution.mean()
            ax.plot([m, m], [i - 0.32, i + 0.32], color=ps.INK, lw=0.9, zorder=5,
                    solid_capstyle="butt")
            v = box[(box.feature == f) & (box.writer == w)].value
            p10, q1, med, q3, p90 = np.percentile(v, [10, 25, 50, 75, 90])
            c = ps.MODEL_SHADES[w]
            axb.plot([p10, p90], [i, i], color=c, lw=0.6, alpha=0.7, solid_capstyle="butt")
            axb.plot([q1, q3], [i, i], color=c, lw=3.0, solid_capstyle="butt")
            axb.scatter(med, i, s=4, color="white", lw=0, zorder=4)
        ax.axvline(0, color=ps.INK_MUTED, lw=0.5, ls=(0, (2, 2)), zorder=1)
        ax.set_xlim(-vmax, vmax)
        for a in (ax, axb):
            a.set_ylim(nw - 0.5, -0.6)
            a.tick_params(axis="y", length=0)
            a.spines["left"].set_visible(False)
        ax.set_yticks(range(nw))
        ax.set_yticklabels([DISPLAY[w] for w in order])
        for t, w in zip(ax.get_yticklabels(), order):
            t.set_color(ps.MODEL_SHADES[w])
        axb.set_yticks([])
        ax.set_title(FEATURE_LABELS.get(f, f), loc="left", x=0, color=CAT[FEAT_CAT[f]],
                     fontweight="bold")
        last = k == len(feats) - 1
        ax.set_xlabel("Contribution to predicted score (style × evaluator's coefficient)"
                      if last else "")
        axb.set_xlabel(f"{FEATURE_LABELS.get(f, f)}, raw value", fontsize=ps.FS_TINY)
        axb.tick_params(axis="x", labelsize=ps.FS_TINY)
        if k == 0:
            axb.set_title("Value in the writer's letters", loc="left", x=0, color=ps.INK_2)

    models = [Line2D([], [], marker="o", ls="", ms=3.2, color=ps.MODEL_SHADES[m], label=DISPLAY[m])
              for m in evaluators]
    keys = [Line2D([], [], marker="o", ls="", ms=3.6, color=ps.INK_MUTED, mec=ps.INK, mew=0.6,
                   label="own evaluation"),
            Line2D([], [], marker="|", ls="", ms=5, mew=0.9, color=ps.INK, label="mean over evaluators")]
    fig.legend(handles=models + keys, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=6,
               frameon=False, handletextpad=0.2, columnspacing=0.9, fontsize=ps.FS_TINY,
               title="Evaluator", title_fontsize=ps.FS_TINY)
    out = os.path.join(DATA, f"cd_{name}.png")
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)


for n in ("semantic_fit_clonly", "semantic_fit_cvcl", "language_complexity"):
    draw(n)
