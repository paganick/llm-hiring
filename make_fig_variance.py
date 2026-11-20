"""Appendix D variance-decomposition figures, redrawn in the paper's figure style.

Reads the CSVs written by plot_nested_model_decomposition.py and
plot_model_explainability.py and writes, next to them:

  vd_nested.png             nested R^2 decomposition, three conditions
  vd_substance_writer.png   per evaluator: substance vs writer identity
  vd_substance_style.png    per evaluator: substance vs feature blocks
  vd_style_drilldown.png    per evaluator: style blocks split by feature

Category colours follow plot_feature_regression.FEATURE_CATEGORY_COLORS (as in
Figure 5); model names are in their model colours.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import paper_style as ps
from plot_feature_regression import FEATURE_CATEGORY_COLORS as CAT, FEATURE_LABELS

ps.apply()
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "output_plots", "cl_features")
DISPLAY = {
    "gpt-4o-mini": "GPT-4o mini", "gpt-5-mini": "GPT-5 mini", "gpt-5": "GPT-5",
    "gemini-3-flash-preview": "Gemini 3 Flash", "gemini-3.5-flash": "Gemini 3.5 Flash",
    "claude-haiku-4-5": "Claude Haiku 4.5", "claude-4.6-sonnet": "Claude Sonnet 4.6",
    "deepseek-v3.2": "DeepSeek V3.2", "deepseek-v4-flash": "DeepSeek V4 Flash",
}
MCOLOR = {v: ps.MODEL_SHADES[k] for k, v in DISPLAY.items()}
SUBSTANCE = "#8c8279"      # warm grey: the candidate's own merit
WRITER = "#c0504d"         # muted red: writer identity
CONDS = [("cl_evaluations", "Cover letter only"), ("cv_cl_evaluations", "CV + cover letter")]


def model_rows(ax, names):
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    for t, n in zip(ax.get_yticklabels(), names):
        t.set_color(MCOLOR.get(n, ps.INK))
    ax.set_ylim(len(names) - 0.5, -0.6)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def stacked(ax, df, cols, colors, total_col="delta_R2", pct=None, xmax=None):
    """Horizontal stacked bars; optional in-bar percentage labels for columns in pct."""
    names = list(df.index)
    left = np.zeros(len(df))
    for c, col in zip(cols, colors):
        v = df[c].values
        ax.barh(range(len(df)), v, left=left, height=0.62, color=col, lw=0)
        if pct and c in pct:
            for i, (l, x) in enumerate(zip(left, v)):
                share = x / df[total_col].iloc[i]
                if share >= 0.06:
                    ax.text(l + x / 2, i, f"{share:.0%}", ha="center", va="center",
                            fontsize=ps.FS_TINY - 0.5, color="white")
        left += v
    for i, tot in enumerate(df[total_col].values):
        ax.text(tot + 0.006 * (xmax or 1), i, f"{tot:.2f}", va="center",
                fontsize=ps.FS_TINY, color=ps.INK_2)
    model_rows(ax, names)
    if xmax:
        ax.set_xlim(0, xmax)


def two_panels(height, wspace=0.55):
    return plt.subplots(1, 2, figsize=(ps.FULL_WIDTH, height), gridspec_kw={"wspace": wspace})


def legend_below(fig, handles, labels, ncol, y=-0.02):
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
               handlelength=0.9, handleheight=0.9, columnspacing=1.2)


# ── nested decomposition ─────────────────────────────────────────────────────
nd = pd.read_csv(os.path.join(DATA, "nested_model_decomposition.csv"), index_col=0)
steps = ["Job", "+ Substance", "+ Evaluator", "+ Writer  [branch A]", "+ Text Features  [branch B]"]
labels = ["Job", "+ Substance", "+ Evaluator", "+ Writer identity (A)", "+ Text features (B)"]
colors = ["#b8b2ab", SUBSTANCE, ps.INK_2, WRITER, ps.PROVIDER_COLORS["OpenAI"]]
fig, axes = plt.subplots(1, 3, figsize=(ps.FULL_WIDTH, 1.75), sharey=True,
                         gridspec_kw={"wspace": 0.12})
for ax, cond in zip(axes, ["CV Only", "Cover Letter Only", "CV + Cover Letter"]):
    for i, (s, col) in enumerate(zip(steps, colors)):
        v = nd.loc[s, cond]
        if pd.isna(v):
            ax.text(0.02, i, "no letter in this condition", va="center",
                    fontsize=ps.FS_TINY, color=ps.INK_MUTED, style="italic")
            continue
        ax.barh(i, v, height=0.62, color=col, lw=0)
        ax.text(v + 0.015, i, f"{v:.3f}", va="center", fontsize=ps.FS_TINY, color=ps.INK_2)
    ax.axhline(2.5, color=ps.INK_MUTED, lw=0.5, ls=(0, (2, 2)))
    ax.set_xlim(0, 1.18); ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel("$R^2$")
    ax.set_title({"CV Only": "CV only", "Cover Letter Only": "Cover letter only",
                  "CV + Cover Letter": "CV + cover letter"}[cond], loc="left", x=0)
axes[0].set_yticks(range(len(labels))); axes[0].set_yticklabels(labels)
axes[0].set_ylim(len(labels) - 0.5, -0.6)
for ax in axes:
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
fig.savefig(os.path.join(DATA, "vd_nested.png")); plt.close(fig)

# ── substance vs writer identity ─────────────────────────────────────────────
fig, axes = two_panels(2.3)
for k, ((etype, title), ax) in enumerate(zip(CONDS, axes)):
    df = pd.read_csv(os.path.join(DATA, f"evaluator_substance_vs_writer_{etype}.csv"), index_col=0)
    stacked(ax, df, ["Substance", "Writer"], [SUBSTANCE, WRITER], pct=["Substance", "Writer"], xmax=0.72)
    ax.set_xlabel("$\\Delta R^2$ over baseline")
    ax.set_title(title, loc="left", x=0)
    ps.panel_label(ax, "ab"[k], x=-0.5, y=1.0)
legend_below(fig, [Patch(color=SUBSTANCE), Patch(color=WRITER)],
             ["Substance (own CV-only score)", "Writer identity"], 2)
fig.savefig(os.path.join(DATA, "vd_substance_writer.png")); plt.close(fig)

# ── substance vs feature blocks ──────────────────────────────────────────────
blocks = ["Job-Ad Fit", "CV Consistency", "Length & Structure", "Language Complexity",
          "Sentiment & Affect", "Emotions"]
fig, axes = two_panels(2.3)
for k, ((etype, title), ax) in enumerate(zip(CONDS, axes)):
    df = pd.read_csv(os.path.join(DATA, f"evaluator_substance_vs_style_{etype}.csv"), index_col=0)
    cols = ["Substance"] + [b for b in blocks if b in df.columns]
    stacked(ax, df, cols, [SUBSTANCE] + [CAT[b] for b in cols[1:]],
            pct=["Substance", "Job-Ad Fit", "CV Consistency"], xmax=0.82)
    ax.set_xlabel("$\\Delta R^2$ over baseline")
    ax.set_title(title, loc="left", x=0)
    ps.panel_label(ax, "ab"[k], x=-0.5, y=1.0)
legend_below(fig, [Patch(color=SUBSTANCE)] + [Patch(color=CAT[b]) for b in blocks],
             ["Substance"] + [b.replace("Job-Ad", "Job-ad").replace("CV Consistency", "CV fit")
                              for b in blocks], 4)
fig.savefig(os.path.join(DATA, "vd_substance_style.png")); plt.close(fig)

# ── style drill-down, one segment per feature ────────────────────────────────
GROUPS = {
    "Length & Structure": ["word_count", "sentence_count", "paragraph_count", "comma_count"],
    "Language Complexity": ["avg_word_length", "ttr", "flesch_reading_ease"],
    "Sentiment & Affect": ["vader_compound", "vad_valence", "vad_arousal", "vad_dominance"],
    "Emotions": ["emo_joy", "emo_neutral", "emo_surprise", "emo_fear", "emo_anger",
                 "emo_disgust", "emo_sadness"],
}


def shades(hex_color, n):
    """n tints of one category colour, light to dark."""
    rgb = np.array(mcolors.to_rgb(hex_color))
    return [tuple(rgb + (1 - rgb) * t) for t in np.linspace(0.65, 0.0, n)]


feat_color = {f: c for g, fs in GROUPS.items() for f, c in zip(fs, shades(CAT[g], len(fs)))}
fig, axes = two_panels(2.3)
for k, ((etype, title), ax) in enumerate(zip(CONDS, axes)):
    tot = pd.read_csv(os.path.join(DATA, f"evaluator_style_drilldown_{etype}_part0.csv"), index_col=0)
    parts = pd.read_csv(os.path.join(DATA, f"evaluator_style_drilldown_{etype}_part1.csv"), index_col=0)
    feats = [f for fs in GROUPS.values() for f in fs if f in parts.columns]
    df = parts.loc[tot.index, feats].copy()
    df["delta_R2"] = tot["delta_R2"]
    stacked(ax, df, feats, [feat_color[f] for f in feats], xmax=0.085)
    ax.set_xlabel("$\\Delta R^2$ from style features")
    ax.set_title(title, loc="left", x=0)
    ps.panel_label(ax, "ab"[k], x=-0.5, y=1.0)
# one legend column per category (matplotlib fills columns top to bottom), padded to equal length
rows_per_col = 1 + max(len(fs) for fs in GROUPS.values())
handles, labs = [], []
for g, fs in GROUPS.items():
    col = [(Patch(color="none"), g)] + [(Patch(color=feat_color[f]), FEATURE_LABELS.get(f, f)) for f in fs]
    col += [(Patch(color="none"), "")] * (rows_per_col - len(col))
    handles += [h for h, _ in col]; labs += [l for _, l in col]
fig.legend(handles, labs, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4,
           handlelength=0.9, handleheight=0.9, columnspacing=1.0, fontsize=ps.FS_TINY)
fig.savefig(os.path.join(DATA, "vd_style_drilldown.png")); plt.close(fig)
print("Saved vd_nested, vd_substance_writer, vd_substance_style, vd_style_drilldown")
