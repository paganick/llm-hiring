"""Evaluator-agreement heatmaps in the paper's tile style (as Figures 2-4).

Reads the correlation matrices that aggregate_plots.py saves and redraws them:

  agreement_quality.png  Appendix Fig. A4b: agreement on candidate quality,
                         one panel per condition (writer fixed for the letter
                         conditions, averaged over writers)
  agreement_writer.png   Appendix Fig. B5: agreement on the ranking of writers
                         for the same candidate, cover letter only vs CV + letter

Run after aggregate_plots.py (generate_all_paper_figures.py does this).
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import paper_style as ps
from aggregate_plots import MODEL_DISPLAY, WRITER_COLORS

ps.apply()
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "output_plots", "paper_plots")
CMAP = plt.get_cmap(ps.SEQUENTIAL_CMAP)
KEY = {v: k for k, v in MODEL_DISPLAY.items()}


def load(name):
    m = pd.read_csv(os.path.join(DATA, name), index_col=0)
    m.index = [KEY.get(i, i) for i in m.index]
    m.columns = [KEY.get(c, c) for c in m.columns]
    for w in m.index:            # self-correlation is 1 by definition: leave the cell empty
        if w in m.columns:
            m.loc[w, w] = float("nan")
    return m


def draw(ax, mat, title, vmin, vmax, letter, show_rows=True):
    ps.tile_matrix(ax, mat, CMAP, vmin, vmax, fmt="{:.2f}", fs=3.8, self_outline=False,
                   row_colors=WRITER_COLORS, col_colors=WRITER_COLORS,
                   row_names=MODEL_DISPLAY, col_names=MODEL_DISPLAY, dark_at=0.2)
    # tile_matrix picks text colour for a diverging map; on this one-hue map only the
    # darker tiles (upper ~40% of the range) need white text
    for t in ax.texts:
        try:
            v = float(t.get_text())
        except ValueError:
            continue
        dark = (v - vmin) / (vmax - vmin) > 0.6
        t.set_color("white" if dark else ps.INK)
    if not show_rows:
        ax.set_yticklabels([])
    ax.set_title(title, loc="left", x=0)
    if letter:   # A4b sits inside a figure that already labels it (b): no inner letters
        ps.panel_label(ax, letter, x=-0.42 if show_rows else -0.06, y=1.0)


def colourbar(fig, axes, vmin, vmax, label):
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin, vmax))
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label(label)
    cb.outline.set_visible(False)


# ── A4b: agreement on candidate quality, three conditions ────────────────────
q = [("merit_agreement_combined_cv_only.csv", "CV only"),
     ("merit_agreement_combined_cover_letter_only_writer_fixed.csv", "Cover letter only"),
     ("merit_agreement_combined_cv_cover_letter_writer_fixed.csv", "CV + cover letter")]
fig, axes = plt.subplots(1, 3, figsize=(ps.FULL_WIDTH, 2.35), gridspec_kw={"wspace": 0.08})
for k, ((f, t), ax) in enumerate(zip(q, axes)):
    draw(ax, load(f), t, 0.7, 1.0, "", show_rows=(k == 0))
colourbar(fig, axes, 0.7, 1.0, "Spearman $\\rho$")
fig.savefig(os.path.join(DATA, "agreement_quality.png"))
plt.close(fig)

# ── B5: agreement on the ranking of writers, two conditions ──────────────────
w = [("agreement_corr_spearman_STYLISTIC_cl_evaluations.csv", "Cover letter only"),
     ("agreement_corr_spearman_STYLISTIC_cv_cl_evaluations.csv", "CV + cover letter")]
fig, axes = plt.subplots(1, 2, figsize=(ps.FULL_WIDTH * 0.8, 2.7), gridspec_kw={"wspace": 0.08})
for k, ((f, t), ax) in enumerate(zip(w, axes)):
    draw(ax, load(f), t, 0.0, 1.0, "ab"[k], show_rows=(k == 0))
colourbar(fig, axes, 0.0, 1.0, "Spearman $\\rho$")
fig.savefig(os.path.join(DATA, "agreement_writer.png"))
plt.close(fig)
print("Saved agreement_quality.png and agreement_writer.png")
