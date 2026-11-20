"""Main-text summary figure for the self-preference paragraph of Section 2
(Results). Simplified from two appendix figures:

  a  per-feature Net Self-Preference (own minus average taste), top eight features
  b  self_preference_and_received   (cover-letter-only only, no arrows)

Reads the CSVs written by plot_feature_regression.py;
writes output_plots/paper_plots/self_pref_summary.png.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
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

fig = plt.figure(figsize=(ps.FULL_WIDTH, 4.3))
gs = fig.add_gridspec(2, 2, height_ratios=[0.85, 1.1], hspace=0.75, wspace=0.32)
axes = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

# ── a: per-feature self-preference, own taste RELATIVE to the average evaluator ─
# own style x (own coefficient - mean coefficient of the nine evaluators); summed
# over all features this is exactly the Net Self-Preference on the x axis of b.
ax = axes[0]
pw = pd.read_csv(os.path.join(DATA, "pairwise_contributions_clonly.csv"))
avg = pw.groupby(["writer", "feature"]).contribution.mean()
own = pw[pw.writer == pw.evaluator].set_index(["writer", "feature"]).contribution
rel = (own - avg.reindex(own.index)).rename("rel").reset_index()
rel = rel[rel.feature.isin(FEATURES)]
mat = rel.pivot(index="writer", columns="feature", values="rel").loc[list(DISPLAY), list(FEATURES)]
mat["_sum"] = mat.sum(axis=1)  # = Net Self-Preference (all nineteen features are shown)
sig = mat.abs() >= 0.02
sig["_sum"] = True
vmax = 0.17
ps.tile_matrix(ax, mat, ps.GAP_CMAP, -vmax, vmax, sig=sig,
               fmt="{:+.2f}", fs=3.5, self_outline=False,
               row_names=DISPLAY, col_names={**FEATURES, "_sum": "Sum"},
               row_colors={w: ps.MODEL_SHADES[w] for w in DISPLAY})
ax.axvline(len(FEATURES) - 0.5, color=ps.INK_2, lw=0.6)  # separate the sum column
ax.get_xticklabels()[-1].set_fontweight("bold")
ax.set_title("Self-preference by feature: own style × (own − average weight)",
             fontweight="normal", fontsize=ps.FS_SMALL)
ps.panel_label(ax, "a", x=-0.13, y=1.08)

# ── b, c: what the panel thinks of a writer vs how a model treats its own letters
from matplotlib.patches import Rectangle
ANTI = "#eeece6"  # shade only the anti-diagonal: liked but harsher on itself, or the reverse


def quadrant_panel(ax, pts, lims, off, xlabel, ylabel, letter, ring=()):
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
    for w, (x, y) in pts.items():
        c = ps.MODEL_SHADES[w]
        if w not in ring:
            ax.scatter(x, y, s=16, color=c, ec="white", lw=0.3, zorder=3)
        if DISPLAY[w] not in off:  # labelled together with the model it overlaps
            continue
        dx, dy, ha = off[DISPLAY[w]]
        ax.annotate(DISPLAY[w], (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", fontsize=ps.FS_TINY, color=c)
    ax.set_xlim(X0, X1)
    ax.set_ylim(Y0, Y1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ps.panel_label(ax, letter, x=-0.16)


# b: from the feature model (letter only)
sr = pd.read_csv(os.path.join(DATA, "self_preference_and_received.csv"), index_col=0)
sr = sr.dropna(subset=["net_self_preference_cl_only"])
# x: mean over the nine evaluators (not the sum), so both axes are in score points
pts_b = {KEY[n]: (r.total_preference_received_cl_only / 9, r.net_self_preference_cl_only)
         for n, r in sr.iterrows()}
quadrant_panel(axes[1], pts_b, (-0.36, 0.34, -0.26, 0.16), {
    "GPT-4o mini": (-5, 0, "right"), "GPT-5 mini": (5, 0, "left"), "GPT-5": (0, -7, "center"),
    "Gemini 3 Flash": (-2, -7, "center"), "Gemini 3.5 Flash": (0, -15, "center"),
    "DeepSeek V4 Flash": (3, 7, "center"), "DeepSeek V3.2": (-9, 7, "center"),
    "Claude Sonnet 4.6": (3, 7, "right"), "Claude Haiku 4.5": (2, -7, "right")},
    "Mean preference received\n(feature model)", "Net self-preference\n(sum of panel a)", "b")

# c: the same comparison from raw scores, without the feature model
gd = pd.read_csv(os.path.join(HERE, "output_plots", "paper_plots", "gap_dotplot_combined.csv"))
gd = gd[gd.condition == "cl_evaluations"]
pts_c, raw = {}, []
for w in gd[gd.self].writer:
    g = gd[gd.writer == w]
    own_gap, others = g[g.self].gap.iloc[0], g[~g.self].gap.mean()
    pts_c[w] = (others, own_gap - others)
    raw.append((w, own_gap - others, others))
quadrant_panel(axes[2], pts_c, (-0.28, 0.52, -0.36, 0.31), {
    "GPT-4o mini": (-5, 0, "right"), "Gemini 3.5 Flash": (3, 7, "right"),
    "Gemini 3 Flash": (0, -7, "center"), "DeepSeek V3.2": (0, 7, "center"),
    "Claude Sonnet 4.6": (6, 3, "left"),
    "GPT-5 mini": (2, 7, "left"), "Claude Haiku 4.5": (6, -7, "center"), "GPT-5": (0, -7, "center")},
    "Mean gap from other evaluators\n(raw scores)", "Own gap minus others'\n(raw scores)", "c",
    ring=("deepseek-v4-flash", "claude-4.6-sonnet"))
# the two overlapping models are drawn as one split dot, half in each colour
xs, ys = pts_c["deepseek-v4-flash"]
axes[2].plot(xs, ys, "o", ms=4.6, fillstyle="left", mfc=ps.MODEL_SHADES["claude-4.6-sonnet"],
             mfcalt=ps.MODEL_SHADES["deepseek-v4-flash"], mec="white", mew=0.3, zorder=3)
axes[2].annotate("DeepSeek V4 Flash", (xs, ys), xytext=(6, -5), textcoords="offset points",
                 ha="left", va="center", fontsize=ps.FS_TINY, color=ps.MODEL_SHADES["deepseek-v4-flash"])
pd.DataFrame(raw, columns=["writer", "net_self_raw", "others_mean_gap"]).to_csv(
    os.path.join(OUT, "self_pref_summary_raw.csv"), index=False)

out = os.path.join(OUT, "self_pref_summary.png")
fig.savefig(out)
print("wrote", out)
