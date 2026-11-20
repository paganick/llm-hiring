"""Appendix D writer-vs-evaluator scatter pairs, in the paper's figure style.

Reads the CSVs that plot_model_explainability.py saves for each pair
(<name>_writer.csv, <name>_evaluator.csv) and redraws them at print size:
panel a = how each model writes (deviation from the writer average),
panel b = what each model rewards as an evaluator (Ridge coefficients).

Writes wve_<name>.png next to the inputs. Run after plot_model_explainability.py.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text
import paper_style as ps

ps.apply()
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "output_plots", "cl_features")
DISPLAY = {
    "gpt-4o-mini": "GPT-4o mini", "gpt-5-mini": "GPT-5 mini", "gpt-5": "GPT-5",
    "gemini-3-flash-preview": "Gemini 3 Flash", "gemini-3.5-flash": "Gemini 3.5 Flash",
    "claude-haiku-4-5": "Claude Haiku 4.5", "claude-4.6-sonnet": "Claude Sonnet 4.6",
    "deepseek-v3.2": "DeepSeek V3.2", "deepseek-v4-flash": "DeepSeek V4 Flash",
    "deepseek-r1-8b": "DeepSeek R1 8B", "llama3.1-8b": "Llama 3.1 8B",
}
COLOR = {v: ps.MODEL_SHADES[k] for k, v in DISPLAY.items()}

# name: (x feature, y feature, columns in the CSVs, diagonal on the evaluator panel)
PAIRS = {
    "cosine_cvcl": ("similarity to job ad", "similarity to CV",
                    ("echo_job", "echo_cv"), ("pref_job", "pref_cv"), True),
    "flesch_wordlen": ("Flesch reading ease", "average word length", ("x", "y"), ("x", "y"), False),
    "dominance_arousal": ("VAD dominance", "VAD arousal", ("x", "y"), ("x", "y"), False),
    "joy_neutral": ("Joy", "Neutral", ("x", "y"), ("x", "y"), False),
}


def panel(ax, df, xcol, ycol, xlabel, ylabel, title, letter, diagonal=False):
    for m, r in df.iterrows():
        ax.scatter(r[xcol], r[ycol], s=16, color=COLOR.get(m, ps.INK_MUTED),
                   ec="white", lw=0.3, zorder=3)
    if diagonal:
        lo = min(df[xcol].min(), df[ycol].min()); hi = max(df[xcol].max(), df[ycol].max())
        pad = (hi - lo) * 0.12
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=ps.INK_MUTED, lw=0.5,
                ls=(0, (1, 2)), zorder=1)
        ax.text(hi + pad, hi + pad, "y = x", ha="right", va="bottom", fontsize=ps.FS_TINY,
                color=ps.INK_MUTED)
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    else:
        ax.axhline(0, color=ps.INK_MUTED, lw=0.5, ls=(0, (2, 2)), zorder=1)
        ax.axvline(0, color=ps.INK_MUTED, lw=0.5, ls=(0, (2, 2)), zorder=1)
        for col, setter in ((xcol, ax.set_xlim), (ycol, ax.set_ylim)):
            lo, hi = df[col].min(), df[col].max()
            pad = (hi - lo) * 0.15
            setter(min(lo - pad, -pad * 0.3), max(hi + pad, pad * 0.3))
    texts = [ax.text(r[xcol], r[ycol], m, fontsize=ps.FS_TINY, color=COLOR.get(m, ps.INK))
             for m, r in df.iterrows()]
    adjust_text(texts, x=df[xcol].values, y=df[ycol].values, ax=ax,
                expand=(1.3, 1.6), arrowprops=dict(arrowstyle="-", color=ps.INK_MUTED, lw=0.3))
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", x=0)
    ps.panel_label(ax, letter, x=-0.16, y=1.0)


for name, (fx, fy, wcols, ecols, diag) in PAIRS.items():
    w = pd.read_csv(os.path.join(DATA, f"writer_vs_evaluator_{name}_writer.csv"), index_col=0)
    e = pd.read_csv(os.path.join(DATA, f"writer_vs_evaluator_{name}_evaluator.csv"), index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(ps.FULL_WIDTH, 2.7), gridspec_kw={"wspace": 0.32})
    panel(axes[0], w, *wcols, f"{fx[0].upper() + fx[1:]}, writer minus average",
          f"{fy[0].upper() + fy[1:]}, writer minus average", "As a writer", "a")
    panel(axes[1], e, *ecols, f"Reward for {fx} (Ridge coef.)",
          f"Reward for {fy} (Ridge coef.)", "As an evaluator", "b", diagonal=diag)
    fig.savefig(os.path.join(DATA, f"wve_{name}.png"))
    plt.close(fig)
    print("saved", f"wve_{name}.png")
