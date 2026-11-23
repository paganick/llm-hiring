"""Self-preference summary figure (main-text Fig. 6, and its CV + cover letter twin).

Four panels, all in points of the 1-10 score scale:

  a  preference received by feature: own style x AVERAGE evaluator weight
     (its row sum is the Mean Preference Received on the x axis of c)
  b  self-preference by feature: own style x (OWN weight - average weight)
     (its row sum is the Net Self-Preference on the y axis of c)
  c  Mean Preference Received vs Net Self-Preference (feature model)
  d  the same comparison from raw scores, without the feature model

a + b is a model's own letters under its own taste (the "absolute" self-evaluation).
Panels a and b share their columns, so the feature names are shown once, under b.

Reads the CSVs written by plot_feature_regression.py and aggregate_plots.py; writes
  output_plots/paper_plots/self_pref_summary.png       cover letter only (main text)
  output_plots/paper_plots/self_pref_summary_cvcl.png  CV + cover letter (appendix)
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from adjustText import adjust_text
import paper_style as ps

ps.apply()
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "output_plots", "cl_features")
OUT = os.path.join(HERE, "output_plots", "paper_plots")
os.makedirs(OUT, exist_ok=True)

DISPLAY = {
    "gpt-4o-mini": "GPT-4o mini", "gpt-5-mini": "GPT-5 mini", "gpt-5": "GPT-5",
    "gemini-3-flash-preview": "Gemini 3 Flash", "gemini-3.5-flash": "Gemini 3.5 Flash",
    "claude-haiku-4-5": "Claude Haiku 4.5", "claude-4.6-sonnet": "Claude Sonnet 4.6",
    "deepseek-v3.2": "DeepSeek V3.2", "deepseek-v4-flash": "DeepSeek V4 Flash",
}
KEY = {v: k for k, v in DISPLAY.items()}
FEATURES = {  # all nineteen cover-letter-only features, grouped as in Figure 5
    "job_cosine_sim": "Cosine sim. to job ad",
    "word_count": "Word count", "sentence_count": "Sentence count",
    "paragraph_count": "Paragraph count", "comma_count": "Comma count",
    "avg_word_length": "Avg. word length", "ttr": "Type-token ratio",
    "flesch_reading_ease": "Flesch reading ease",
    "vader_compound": "VADER sentiment", "vad_valence": "VAD valence",
    "vad_arousal": "VAD arousal", "vad_dominance": "VAD dominance",
    "emo_joy": "Joy", "emo_neutral": "Neutral", "emo_surprise": "Surprise",
    "emo_fear": "Fear", "emo_anger": "Anger", "emo_disgust": "Disgust",
    "emo_sadness": "Sadness",
}
FEATURES_CVCL = {"job_cosine_sim": FEATURES["job_cosine_sim"], "cv_cosine_sim": "Cosine sim. to CV",
                 **{k: v for k, v in FEATURES.items() if k != "job_cosine_sim"}}
ANTI = "#eeece6"  # shade only the anti-diagonal: liked but harsher on itself, or the reverse
VMAX_RECEIVED, VMAX_SELF = 0.5, 0.17   # separate colour scales for a and b (see caption)


def tiles(ax, mat, vmax, title, letter, show_names):
    mat = mat.copy()
    mat["_sum"] = mat.sum(axis=1)
    sig = mat.abs() >= 0.02
    sig["_sum"] = True
    feats = [c for c in mat.columns if c != "_sum"]
    names = FEATURES_CVCL
    ps.tile_matrix(ax, mat, ps.GAP_CMAP, -vmax, vmax, sig=sig, fmt="{:+.2f}", fs=3.5,
                   self_outline=False, row_names=DISPLAY, col_names={**names, "_sum": "Sum"},
                   row_colors={w: ps.MODEL_SHADES[w] for w in DISPLAY})
    ax.axvline(len(feats) - 0.5, color=ps.INK_2, lw=0.6)  # separate the sum column
    if show_names:
        ax.get_xticklabels()[-1].set_fontweight("bold")
    else:   # same columns as the panel below: feature names shown once
        ax.set_xticklabels([])
    ax.set_title(title, fontweight="normal", fontsize=ps.FS_SMALL)
    ps.panel_label(ax, letter, x=-0.13, y=1.08)
    return mat["_sum"]


def quadrant_panel(ax, pts, lims, off, xlabel, ylabel, letter, ring=()):
    """off: {display name: (dx, dy, ha)} hand-placed labels, or None for automatic placement."""
    X0, X1, Y0, Y1 = lims
    ax.axhline(0, color=ps.INK_MUTED, lw=0.5, ls=(0, (2, 2)), zorder=1)
    ax.axvline(0, color=ps.INK_MUTED, lw=0.5, ls=(0, (2, 2)), zorder=1)
    quads = [((0, 0), None, "liked by the panel,\nkinder to itself", "right", "top"),
             ((X0, 0), ANTI, "less liked,\nkinder to itself", "left", "top"),
             ((0, Y0), ANTI, "liked by the panel,\nharsher on itself", "right", "bottom"),
             ((X0, Y0), None, "less liked,\nharsher on itself", "left", "bottom")]
    for (qx, qy), fc, txt, ha, va in quads:
        if fc:
            ax.add_patch(Rectangle((qx, qy), (X1 if qx == 0 else 0) - qx,
                                   (Y1 if qy == 0 else 0) - qy, fc=fc, ec="none", zorder=0))
        ax.text(X1 - 0.01 if ha == "right" else X0 + 0.01,
                Y1 - 0.01 if va == "top" else Y0 + 0.01, txt, ha=ha, va=va,
                fontsize=ps.FS_TINY - 0.5, color=ps.INK_MUTED, linespacing=1.0)
    texts = []
    for w, (x, y) in pts.items():
        c = ps.MODEL_SHADES[w]
        if w not in ring:
            ax.scatter(x, y, s=16, color=c, ec="white", lw=0.3, zorder=3)
        if off is None:
            texts.append(ax.text(x, y, DISPLAY[w], fontsize=ps.FS_TINY, color=c))
            continue
        if DISPLAY[w] not in off:  # labelled together with the model it overlaps
            continue
        dx, dy, ha = off[DISPLAY[w]]
        ax.annotate(DISPLAY[w], (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", fontsize=ps.FS_TINY, color=c)
    ax.set_xlim(X0, X1)
    ax.set_ylim(Y0, Y1)
    if texts:
        xs, ys = zip(*pts.values())
        adjust_text(texts, x=list(xs), y=list(ys), ax=ax, expand=(1.3, 1.6),
                    arrowprops=dict(arrowstyle="-", color=ps.INK_MUTED, lw=0.3))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ps.panel_label(ax, letter, x=-0.16)


def limits(pts, pad=0.5):
    xs, ys = zip(*pts.values())
    def rng(v):
        lo, hi = min(min(v), 0), max(max(v), 0)
        p = (hi - lo) * pad
        return lo - p, hi + p
    return (*rng(xs), *rng(ys))


def make(cond):
    cl = cond == "cl"
    feats = FEATURES if cl else FEATURES_CVCL
    fig = plt.figure(figsize=(ps.FULL_WIDTH, 5.5))
    outer = fig.add_gridspec(2, 1, height_ratios=[2.05, 1.1], hspace=0.42)
    top = outer[0].subgridspec(2, 1, hspace=0.28)
    bot = outer[1].subgridspec(1, 2, wspace=0.32)
    ax_a, ax_b = fig.add_subplot(top[0]), fig.add_subplot(top[1])
    ax_c, ax_d = fig.add_subplot(bot[0]), fig.add_subplot(bot[1])

    # a, b: per-feature contributions, writer = evaluator diagonal vs the nine-evaluator average
    pw = pd.read_csv(os.path.join(DATA, f"pairwise_contributions_{'clonly' if cl else 'cvcl'}.csv"))
    avg = pw.groupby(["writer", "feature"]).contribution.mean()
    own = pw[pw.writer == pw.evaluator].set_index(["writer", "feature"]).contribution
    received = avg.reindex(own.index).rename("v").reset_index()
    relative = (own - avg.reindex(own.index)).rename("v").reset_index()
    pivot = lambda d: d[d.feature.isin(feats)].pivot(index="writer", columns="feature", values="v") \
        .loc[list(DISPLAY), list(feats)]
    rec_sum = tiles(ax_a, pivot(received), VMAX_RECEIVED,
                    "Preference received by feature: own style × average weight", "a", False)
    self_sum = tiles(ax_b, pivot(relative), VMAX_SELF,
                     "Self-preference by feature: own style × (own − average weight)", "b", True)

    # c: from the feature model; the two axes are exactly the row sums of a and b
    pts_c = {w: (rec_sum[w], self_sum[w]) for w in DISPLAY}
    off_c = {
        "GPT-4o mini": (-5, 0, "right"), "GPT-5 mini": (5, 0, "left"), "GPT-5": (0, -7, "center"),
        "Gemini 3 Flash": (-2, -7, "center"), "Gemini 3.5 Flash": (0, -15, "center"),
        "DeepSeek V4 Flash": (3, 7, "center"), "DeepSeek V3.2": (-9, 7, "center"),
        "Claude Sonnet 4.6": (3, 7, "right"), "Claude Haiku 4.5": (2, -7, "right")} if cl else {
        "GPT-4o mini": (4, 3, "left"), "GPT-5 mini": (4, -3, "left"), "GPT-5": (0, -8, "center"),
        "Gemini 3 Flash": (-3, -8, "right"), "Gemini 3.5 Flash": (4, 0, "left"),
        "DeepSeek V4 Flash": (0, 8, "center"), "DeepSeek V3.2": (0, 8, "center"),
        "Claude Sonnet 4.6": (0, 7, "center"), "Claude Haiku 4.5": (-5, -1, "right")}
    quadrant_panel(ax_c, pts_c, (-0.36, 0.34, -0.26, 0.16) if cl else limits(pts_c), off_c,
                   "Mean preference received\n(sum of panel a)",
                   "Net self-preference\n(sum of panel b)", "c")

    # d: the same comparison from raw scores, without the feature model
    gd = pd.read_csv(os.path.join(HERE, "output_plots", "paper_plots", "gap_dotplot_combined.csv"))
    gd = gd[gd.condition == ("cl_evaluations" if cl else "cv_cl_evaluations")]
    pts_d, raw = {}, []
    for w in gd[gd.self].writer:
        g = gd[gd.writer == w]
        own_gap, others = g[g.self].gap.iloc[0], g[~g.self].gap.mean()
        pts_d[w] = (others, own_gap - others)
        raw.append((w, own_gap - others, others))
    if cl:
        quadrant_panel(ax_d, pts_d, (-0.28, 0.52, -0.36, 0.31), {
            "GPT-4o mini": (-5, 0, "right"), "Gemini 3.5 Flash": (3, 7, "right"),
            "Gemini 3 Flash": (0, -7, "center"), "DeepSeek V3.2": (0, 7, "center"),
            "Claude Sonnet 4.6": (6, 3, "left"),
            "GPT-5 mini": (2, 7, "left"), "Claude Haiku 4.5": (6, -7, "center"), "GPT-5": (0, -7, "center")},
            "Mean gap from other evaluators\n(raw scores)", "Own gap minus others'\n(raw scores)", "d",
            ring=("deepseek-v4-flash", "claude-4.6-sonnet"))
        # the two overlapping models are drawn as one split dot, half in each colour
        xs, ys = pts_d["deepseek-v4-flash"]
        ax_d.plot(xs, ys, "o", ms=4.6, fillstyle="left", mfc=ps.MODEL_SHADES["claude-4.6-sonnet"],
                  mfcalt=ps.MODEL_SHADES["deepseek-v4-flash"], mec="white", mew=0.3, zorder=3)
        ax_d.annotate("DeepSeek V4 Flash", (xs, ys), xytext=(6, -5), textcoords="offset points",
                      ha="left", va="center", fontsize=ps.FS_TINY, color=ps.MODEL_SHADES["deepseek-v4-flash"])
    else:
        quadrant_panel(ax_d, pts_d, limits(pts_d), None,
                       "Mean gap from other evaluators\n(raw scores)", "Own gap minus others'\n(raw scores)", "d")

    suffix = "" if cl else "_cvcl"
    pd.DataFrame(raw, columns=["writer", "net_self_raw", "others_mean_gap"]).to_csv(
        os.path.join(OUT, f"self_pref_summary{suffix}_raw.csv"), index=False)
    pd.concat([pivot(received).assign(part="received"), pivot(relative).assign(part="self")]).to_csv(
        os.path.join(OUT, f"self_pref_summary{suffix}_by_feature.csv"))
    out = os.path.join(OUT, f"self_pref_summary{suffix}.png")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    make("cl")
    make("cvcl")
