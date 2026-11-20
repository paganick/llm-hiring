"""Appendix D self-evaluation figures, redrawn in the paper's figure style.

Reads the CSVs written by plot_feature_regression.py and writes, next to them:

  sp_absolute_clonly.png  own style x own Ridge coefficient, per feature, cover letter only
  sp_absolute_cvcl.png    the same with the CV present
  sp_received.png         Net Self-Preference vs Total Preference Received, both conditions

Markers: up/down triangle = the model's own letters sit above/below the
feature's mean; marker size = how far (standardised effect size).
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import paper_style as ps
from plot_feature_regression import FEATURE_CATEGORY_COLORS as CAT, FEATURE_LABELS, FEATURE_GROUPS_MAP

ps.apply()
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "output_plots", "cl_features")
DISPLAY = {
    "gpt-4o-mini": "GPT-4o mini", "gpt-5-mini": "GPT-5 mini", "gpt-5": "GPT-5",
    "gemini-3-flash-preview": "Gemini 3 Flash", "gemini-3.5-flash": "Gemini 3.5 Flash",
    "claude-haiku-4-5": "Claude Haiku 4.5", "claude-4.6-sonnet": "Claude Sonnet 4.6",
    "deepseek-v3.2": "DeepSeek V3.2", "deepseek-v4-flash": "DeepSeek V4 Flash",
}
KEY = {v: k for k, v in DISPLAY.items()}


def category(f):
    g = FEATURE_GROUPS_MAP.get(f, "")
    if g == "Semantic Fit":
        return "Job-Ad Fit" if f == "job_cosine_sim" else "CV Consistency"
    return g


def absolute(csv, out):
    d = pd.read_csv(os.path.join(DATA, csv))
    feats = d.sort_values("rank_from_top").drop_duplicates("feature")["feature"].tolist()
    feats = sorted(feats, key=lambda f: -d[d.feature == f].self_contribution.abs().max())
    fig, ax = plt.subplots(figsize=(ps.FULL_WIDTH * 0.78, 0.17 * len(feats) + 0.9))
    for i, f in enumerate(feats):
        ax.axhline(i, color="#eceae6", lw=0.4, zorder=0)
        for _, r in d[d.feature == f].iterrows():
            ax.scatter(r.self_contribution, i, marker="^" if r.above_pooled_mean else "v",
                       s=4 + 18 * min(r.effect_sd_units, 2.0), color=ps.MODEL_SHADES[r.writer],
                       lw=0, zorder=3)
    ax.axvline(0, color=ps.INK_MUTED, lw=0.5, ls=(0, (2, 2)), zorder=1)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels([FEATURE_LABELS.get(f, f) for f in feats])
    for t, f in zip(ax.get_yticklabels(), feats):
        t.set_color(CAT.get(category(f), ps.INK))
    ax.set_ylim(len(feats) - 0.5, -0.6)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Own letters under own taste (own style × own Ridge coefficient)")
    models = [Line2D([], [], marker="o", ls="", ms=3.5, color=ps.MODEL_SHADES[k], label=v)
              for k, v in DISPLAY.items()]
    shapes = [Line2D([], [], marker="^", ls="", ms=4, color=ps.INK_2, label="own mean above feature mean"),
              Line2D([], [], marker="v", ls="", ms=4, color=ps.INK_2, label="own mean below feature mean")]
    leg1 = fig.legend(handles=models, loc="upper left", bbox_to_anchor=(0.98, 0.95), frameon=False,
                      handletextpad=0.3)
    fig.legend(handles=shapes, loc="upper left", bbox_to_anchor=(0.98, 0.42), frameon=False,
               handletextpad=0.3, title="marker size = distance", title_fontsize=ps.FS_TINY)
    fig.add_artist(leg1)
    fig.savefig(os.path.join(DATA, out))
    plt.close(fig)


absolute("self_points_summary_ranked_data.csv", "sp_absolute_clonly.png")
absolute("self_points_summary_ranked_cvcl_data.csv", "sp_absolute_cvcl.png")

# ── Net Self-Preference vs Total Preference Received, arrow = adding the CV ──
sr = pd.read_csv(os.path.join(DATA, "self_preference_and_received.csv"), index_col=0)
sr = sr.dropna(subset=["net_self_preference_cl_only"])
fig, ax = plt.subplots(figsize=(ps.FULL_WIDTH * 0.72, 3.0))
ax.axhline(0, color=ps.INK_MUTED, lw=0.5, ls=(0, (2, 2)), zorder=1)
ax.axvline(0, color=ps.INK_MUTED, lw=0.5, ls=(0, (2, 2)), zorder=1)
OFF = {"GPT-4o mini": (4, 0, "left"), "GPT-5 mini": (-4, 0, "right"), "GPT-5": (-4, 0, "right"),
       "Gemini 3 Flash": (-4, -4, "right"), "Gemini 3.5 Flash": (4, 4, "left"),
       "DeepSeek V4 Flash": (4, -2, "left"), "DeepSeek V3.2": (4, 0, "left"),
       "Claude Sonnet 4.6": (4, 0, "left"), "Claude Haiku 4.5": (-4, 0, "right")}
for name, r in sr.iterrows():
    c = ps.MODEL_SHADES[KEY[name]]
    x0, y0 = r.net_self_preference_cl_only, r.total_preference_received_cl_only
    x1, y1 = r.net_self_preference_cv_cl, r.total_preference_received_cv_cl
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=c, lw=0.6, alpha=0.6,
                                shrinkA=2.5, shrinkB=2.5, mutation_scale=5))
    ax.scatter(x0, y0, s=16, marker="o", color=c, ec="white", lw=0.3, zorder=3)
    ax.scatter(x1, y1, s=12, marker="s", color=c, ec="white", lw=0.3, zorder=3)
    dx, dy, ha = OFF[name]
    ax.annotate(name, (x0, y0), xytext=(dx, dy), textcoords="offset points", ha=ha,
                va="center", fontsize=ps.FS_TINY, color=c)
xs = pd.concat([sr.net_self_preference_cl_only, sr.net_self_preference_cv_cl])
ax.set_xlim(xs.min() - 0.025, xs.max() + 0.03)   # room for the labels at both ends
ax.set_xlabel("Net Self-Preference (own minus average taste, summed)")
ax.set_ylabel("Total Preference Received (all nine evaluators)")
ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=4, color=ps.INK_2, label="Cover letter only"),
                   Line2D([], [], marker="s", ls="", ms=3.5, color=ps.INK_2, label="CV + cover letter")],
          loc="lower left", frameon=False, handletextpad=0.3)
fig.savefig(os.path.join(DATA, "sp_received.png"))
plt.close(fig)
print("Saved sp_absolute_clonly, sp_absolute_cvcl, sp_received")
