"""paper_style.py: one visual style for every paper figure.

Figures are drawn at their printed size, so font sizes here are the sizes
that appear on the page (no print_pt rescaling needed). Conventions follow
Nature-journal artwork guidance: sans-serif type at 5-7 pt, thin lines,
sentence-case labels, bold lowercase panel letters, no in-figure headline
(the caption carries it).

Colour encodes provider, not individual model: five groups validated for
colour-vision deficiency (worst pair under protan/deutan/tritan simulation
dE_OKLab x100 = 8.8, normal vision 18.4). Locally hosted models share a
neutral dark grey, so the paper's API-vs-local contrast is visible at a glance.
Text always stays in neutral ink; colour sits on small swatches next to it.
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ── Sizes ────────────────────────────────────────────────────────────────────
CM = 1 / 2.54
TEXTWIDTH_IN = 17 * CM            # paper.tex: a4, 2 cm margins
FULL_WIDTH = 0.95 * TEXTWIDTH_IN  # \includegraphics[width=0.95\textwidth]
HALF_WIDTH = 0.48 * TEXTWIDTH_IN

FS_BASE, FS_SMALL, FS_TINY = 7.0, 6.0, 5.0

# ── Colours ──────────────────────────────────────────────────────────────────
INK, INK_2, INK_MUTED = "#1a1a19", "#52514e", "#9a9893"
PROVIDER_COLORS = {
    "OpenAI":         "#2a78d6",
    "Google":         "#1baf7a",
    "Anthropic":      "#eb6834",
    "DeepSeek":       "#c03a83",
    "Locally hosted": "#52514e",
}
MODEL_PROVIDER = {
    "gpt-4o-mini": "OpenAI", "gpt-5-mini": "OpenAI", "gpt-5": "OpenAI",
    "gemini-3-flash-preview": "Google", "gemini-3.5-flash": "Google",
    "claude-haiku-4-5": "Anthropic", "claude-4.6-sonnet": "Anthropic",
    "deepseek-v3.2": "DeepSeek", "deepseek-v4-flash": "DeepSeek",
    "deepseek-r1-8b": "Locally hosted", "llama3.1-8b": "Locally hosted",
}
MODEL_COLORS = {m: PROVIDER_COLORS[p] for m, p in MODEL_PROVIDER.items()}

# Per-model shades of the provider colour, for figures where several models of
# one provider sit side by side (bars, dots, coloured labels). All have >= 3:1
# contrast on white, so they are legible as label text too.
MODEL_SHADES = {
    "gpt-5": "#1b4f8f", "gpt-5-mini": "#2a78d6", "gpt-4o-mini": "#4f93e0",
    "gemini-3.5-flash": "#0e7a54", "gemini-3-flash-preview": "#17a06f",
    "claude-4.6-sonnet": "#b54613", "claude-haiku-4-5": "#e2622b",
    "deepseek-v3.2": "#8e2860", "deepseek-v4-flash": "#c03a83",
    "llama3.1-8b": "#3a3936", "deepseek-r1-8b": "#77756f",
}

# Diverging maps are chosen by MEANING, not by sign:
#   DIVERGING_CMAP       positive = favourable (favoured, wins, agreement, rewarded) -> blue
#   DIVERGING_CMAP_HARM  positive = harmful (net displacement of high-fit candidates) -> red
DIVERGING_CMAP = "RdBu"
DIVERGING_CMAP_HARM = "RdBu_r"
SEQUENTIAL_CMAP = "Blues"   # magnitude: one hue, light -> dark


def apply():
    """Set the global matplotlib style. Call once at import time in each plotting script."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FS_BASE, "axes.titlesize": FS_BASE, "axes.labelsize": FS_BASE,
        "xtick.labelsize": FS_SMALL, "ytick.labelsize": FS_SMALL,
        "legend.fontsize": FS_SMALL, "legend.frameon": False,
        "axes.titleweight": "bold", "axes.titlepad": 4,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": "#e6e5e1", "grid.linewidth": 0.4,
        "lines.linewidth": 1.0, "patch.linewidth": 0.5,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "figure.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,   # editable text in vector output
    })
    # No figure-wide headlines: the caption carries the title (Nature style).
    # Panel titles (ax.set_title) are kept. Turning suptitle into a no-op here
    # avoids editing ~25 call sites; savefig crops the space they reserved.
    from matplotlib.figure import Figure
    Figure.suptitle = lambda self, *args, **kwargs: None


def panel_label(ax, letter, x=-0.02, y=1.02):
    """Bold lowercase panel letter (Nature style) at the top-left of ax."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, fontweight="bold",
            ha="right", va="bottom")


def model_swatches(ax, models, axis="y", size=0.012, offset=0.004):
    """Neutral tick labels plus a small provider-coloured square beside each.

    For axis="y" the squares sit just left of the axes; for axis="x" just below.
    `size`/`offset` are in axes-fraction units of the long side.
    """
    n = len(models)
    for i, m in enumerate(models):
        if m not in MODEL_COLORS:
            continue
        c = MODEL_COLORS[m]
        if axis == "y":
            ax.add_patch(Rectangle((-offset - size, i + 0.5 - 0.22), size, 0.44,
                                   transform=mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData),
                                   color=c, clip_on=False, lw=0))
        else:
            ax.add_patch(Rectangle((i + 0.5 - 0.22, -offset - size), 0.44, size,
                                   transform=mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes),
                                   color=c, clip_on=False, lw=0))


def provider_legend(fig, loc="lower center", bbox=(0.5, -0.02), ncol=5):
    handles = [Rectangle((0, 0), 1, 1, color=c, lw=0) for c in PROVIDER_COLORS.values()]
    fig.legend(handles, PROVIDER_COLORS.keys(), loc=loc, bbox_to_anchor=bbox, ncol=ncol,
               handlelength=0.9, handleheight=0.9, columnspacing=1.2, fontsize=FS_SMALL)


# ── Heatmap ──────────────────────────────────────────────────────────────────
def fmt_num(v, decimals=2, strip_zero=True, sign=False):
    """Compact number: '.47', '−.06', '+1.2'; true minus sign; no '−.00'."""
    s = f"{v:+.{decimals}f}" if sign else f"{v:.{decimals}f}"
    if strip_zero:
        s = s.replace("0.", ".", 1) if abs(v) < 1 else s
    s = s.replace("-", "−")
    if s.lstrip("+−").strip(".0") == "":   # rounds to zero
        s = s.lstrip("+−")
    return s


def heatmap(ax, df, cmap, vmin, vmax, sig=None, fmt=fmt_num, display=None,
            mean_label="Avg.", mean_name="Mean", diag=True, cell_h=0.72,
            text_fs=4.8, sub=None, sub_fs=3.6, row_swatches=True, col_swatches=True,
            missing="—", mean_gap=0.35):
    """Draw df as a heatmap of separate cells (white gaps), the house style.

    sig      : DataFrame of bools like df; False -> value printed in muted grey.
    sub      : optional DataFrame of strings printed small under each value (e.g. '±1.9').
    display  : dict model-id -> display name for tick labels.
    A row/column labelled `mean_label` is set apart by a small gap and named `mean_name`.
    Self-evaluation cells (row == column) get a thin dark outline.
    """
    cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    display = display or {}
    rows, cols = list(df.index), list(df.columns)
    def ypos(i, r):
        return (i + (mean_gap if r == mean_label else 0)) * cell_h
    def xpos(j, c):
        return j + (mean_gap if c == mean_label else 0)
    for i, r in enumerate(rows):
        y = ypos(i, r)
        for j, c in enumerate(cols):
            x = xpos(j, c)
            val = df.loc[r, c]
            if val is None or (isinstance(val, float) and val != val):
                ax.add_patch(Rectangle((x, y), 1, cell_h, facecolor="#f1f0ec", edgecolor="white", lw=0.6))
                ax.text(x + .5, y + cell_h / 2, missing, ha="center", va="center", fontsize=text_fs, color=INK_MUTED)
                continue
            t = (val - vmin) / (vmax - vmin)
            ax.add_patch(Rectangle((x, y), 1, cell_h, facecolor=cmap(min(max(t, 0), 1)), edgecolor="white", lw=0.6))
            is_sig = True if sig is None else bool(sig.loc[r, c])
            strong = abs(t - 0.5) > 0.32
            col = "white" if strong else (INK if is_sig else INK_MUTED)
            yt = y + cell_h / 2 - (0.09 * cell_h if sub is not None else 0)
            ax.text(x + .5, yt, fmt(val), ha="center", va="center", fontsize=text_fs, color=col)
            if sub is not None and isinstance(sub.loc[r, c], str):
                ax.text(x + .5, y + cell_h * 0.78, sub.loc[r, c], ha="center", va="center",
                        fontsize=sub_fs, color=col)
            if diag and r == c and r != mean_label:
                ax.add_patch(Rectangle((x + .04, y + .03), .92, cell_h - .06, fill=False, ec=INK, lw=0.7))
    n_x = len(cols) + (mean_gap if mean_label in cols else 0)
    n_y = (len(rows) + (mean_gap if mean_label in rows else 0)) * cell_h
    ax.set_xlim(0, n_x); ax.set_ylim(n_y, 0)
    ax.set_xticks([xpos(j, c) + .5 for j, c in enumerate(cols)])
    ax.set_xticklabels([mean_name if c == mean_label else display.get(c, c) for c in cols],
                       rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks([ypos(i, r) + cell_h / 2 for i, r in enumerate(rows)])
    ax.set_yticklabels([mean_name if r == mean_label else display.get(r, r) for r in rows])
    ax.tick_params(length=0, pad=7)
    for s in ax.spines.values():
        s.set_visible(False)
    tr_y = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    tr_x = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    if row_swatches:
        for i, r in enumerate(rows):
            if r in MODEL_COLORS:
                ax.add_patch(Rectangle((-0.03, ypos(i, r) + cell_h * .25), 0.018, cell_h * .5,
                                       transform=tr_y, color=MODEL_COLORS[r], clip_on=False, lw=0))
    if col_swatches:
        for j, c in enumerate(cols):
            if c in MODEL_COLORS:
                ax.add_patch(Rectangle((xpos(j, c) + .28, -0.035), .44, 0.022,
                                       transform=tr_x, color=MODEL_COLORS[c], clip_on=False, lw=0))


def colorbar(fig, ax, cmap, vmin, vmax, label, low=None, high=None, **kw):
    """Thin colour bar; optional low/high end labels (e.g. 'penalised', 'favoured')."""
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    if isinstance(ax, mpl.axes.Axes) and "cax" not in kw:
        # slim bar attached to this one panel, so neighbouring panels keep their room
        kw["cax"] = ax.inset_axes([1.03, 0.08, 0.025, 0.84])
    else:
        kw = {"fraction": 0.025, "pad": 0.02, "aspect": 30, **kw}
    cb = fig.colorbar(sm, **({"ax": ax} if "cax" not in kw else {}), **kw)
    cb.outline.set_visible(False)
    cb.ax.tick_params(width=0.4, length=2, labelsize=FS_TINY + 0.5)
    # End meanings folded into the label, reading along the bar (low end first),
    # e.g. "penalised \u2190  Preference gap  \u2192 favoured": nothing to collide with.
    if low or high:
        label = f"{low} \u2190   {label}   \u2192 {high}"
    cb.set_label(label, fontsize=FS_TINY + 0.5)
    return cb


def sentence_case(s):
    """'Cosine Sim. to Job Ad' -> 'Cosine sim. to job ad'; keeps acronyms (VAD, VADER, CV)."""
    words = s.split(" ")
    out = []
    def fix(part, first):
        core = part.strip("().,")
        return part if (first or sum(ch.isupper() for ch in core) >= 2) else part.lower()
    for k, w in enumerate(words):
        out.append("-".join(fix(p, k == 0 and j == 0) for j, p in enumerate(w.split("-"))))
    return " ".join(out)


# ---------------------------------------------------------------------------
# 2026-11-18 redesign (Fig. 2 dot plot, tile matrices): red = favoured.
# ---------------------------------------------------------------------------
from matplotlib.colors import LinearSegmentedColormap as _LSC
from matplotlib.patches import FancyBboxPatch as _FBP

FAV, PEN = "#a8341f", "#2c5d8f"            # favoured / penalised ink
FAV_BG, PEN_BG = "#fbf1ec", "#eef3f8"      # matching background washes
# Soft diverging map, blue (low) -> warm white -> red (high).
GAP_CMAP = _LSC.from_list("gap", ["#2c5d8f", "#7fa8cc", "#d6e3ee", "#f7f7f5",
                                  "#f3d9cc", "#d9876a", "#a8341f"])


def tile_matrix(ax, df, cmap, vmin, vmax, sig=None, fmt="{:.2f}", fs=4.6,
                self_outline=True, row_colors=None, col_colors=None,
                row_names=None, col_names=None, dark_at=0.56):
    """Matrix as rounded tiles separated by white gaps (no grid lines).

    sig: boolean frame like df; True = bold ink, False = faded grey text.
    self_outline: outline cells where the row label equals the column label.
    row/col_colors: dict label -> colour for the tick labels.
    dark_at: |normalised value| above which text turns white.
    """
    nr, nc = df.shape
    for i, r in enumerate(df.index):
        for j, c in enumerate(df.columns):
            v = df.iloc[i, j]
            if pd.isna(v):
                continue
            t = (v - vmin) / (vmax - vmin)
            ax.add_patch(_FBP((j - 0.44, i - 0.40), 0.88, 0.80,
                              boxstyle="round,pad=0,rounding_size=0.12", fc=cmap(t), ec="none"))
            if self_outline and r == c:
                ax.add_patch(_FBP((j - 0.47, i - 0.43), 0.94, 0.86,
                                  boxstyle="round,pad=0,rounding_size=0.14", fc="none", ec=INK, lw=0.8))
            s = True if sig is None else bool(sig.iloc[i, j])
            dark = abs(t - 0.5) * 2 > dark_at
            txt = fmt.format(v)
            if txt in ("-0.00", "+0.00", "-0.0", "+0.0"):
                txt = txt[1:]
            ax.text(j, i, txt, ha="center", va="center", fontsize=fs,
                    color=("white" if dark else INK) if s else ("#e4e4e4" if dark else INK_MUTED),
                    fontweight="bold" if s else "normal")
    ax.set_xlim(-0.5, nc - 0.5)
    ax.set_ylim(nr - 0.5, -0.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_yticks(range(nr))
    ax.set_yticklabels([(row_names or {}).get(r, r) for r in df.index])
    ax.set_xticks(range(nc))
    ax.set_xticklabels([(col_names or {}).get(c, c) for c in df.columns],
                       rotation=40, ha="right", rotation_mode="anchor")
    for t, r in zip(ax.get_yticklabels(), df.index):
        t.set_color((row_colors or {}).get(r, INK))
    for t, c in zip(ax.get_xticklabels(), df.columns):
        t.set_color((col_colors or {}).get(c, INK))


def margin_bars(axb, values, cmap, vmin, vmax, ylim, fmt="{:.2f}", fs=4.6, label="Avg."):
    """Bar chart above a tile matrix (one bar per column), coloured by the same map."""
    x = np.arange(len(values))
    axb.bar(x, values, width=0.62, color=[cmap((v - vmin) / (vmax - vmin)) for v in values], lw=0)
    axb.axhline(0, color=INK_2, lw=0.5)
    pad = 0.06 * (ylim[1] - ylim[0])
    for j, v in enumerate(values):
        axb.text(j, v + (pad if v >= 0 else -pad), fmt.format(v), ha="center",
                 va="bottom" if v >= 0 else "top", fontsize=fs, color=INK_2)
    axb.set_ylim(*ylim)
    axb.spines["bottom"].set_visible(False)
    axb.spines["top"].set_visible(False)
    axb.spines["right"].set_visible(False)
    axb.tick_params(axis="x", length=0, labelbottom=False)
    axb.set_ylabel(label)


def side_bars(axr, values, colors, xlim, fmt="{:.1f}", fs=4.6, base=0.0, label=None):
    """Horizontal bars to the right of a tile matrix (one per row), drawn from `base`.

    Row i sits at y = i, matching tile_matrix's coordinates.
    """
    y = np.arange(len(values))
    axr.barh(y, np.asarray(values) - base, left=base, height=0.62, color=colors, lw=0)
    axr.axvline(base, color=INK_2, lw=0.5)
    pad = 0.03 * (xlim[1] - xlim[0])
    for yi, v in zip(y, values):
        axr.text(v + (pad if v >= base else -pad), yi, fmt.format(v), va="center",
                 ha="left" if v >= base else "right", fontsize=fs, color=INK)
    axr.set_xlim(*xlim)
    axr.set_ylim(len(values) - 0.5, -0.5)
    axr.tick_params(axis="y", left=False, labelleft=False)
    for side in ("left", "top", "right"):
        axr.spines[side].set_visible(False)
    if label:
        axr.set_xlabel(label)
