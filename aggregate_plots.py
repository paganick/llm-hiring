"""
aggregate_plots.py

Generates 4 fully-aggregated (All_CVs tier, all jobs combined) plots
and their companion CSV files in OUT_DIR:

  1. heatmap_gap_cl_evaluations.png / .csv
  2. heatmap_gap_cv_cl_evaluations.png / .csv
  3. win_matrix_UNBIASED_cv_cl_evaluations.png / .csv
  4. net_advantage_matrix_ALL_COMBINED.png / .csv
"""

import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
from scipy import stats

# ==========================
# CONFIGURATION
# ==========================
BASE_DIR = "./output_eval"

NO_GEMINI2 = "--no-gemini2" in sys.argv

if NO_GEMINI2:
    OUT_DIR         = "./output_plots/paper_plots_v3_no_gemini2"
    CACHE_PATH      = "./output_eval/master_df_no_gemini2.parquet"
    COMP_CACHE_PATH = "./output_eval/competitive_df_no_gemini2.parquet"
else:
    OUT_DIR         = "./output_plots/paper_plots_v3"
    CACHE_PATH      = "./output_eval/master_df.parquet"
    COMP_CACHE_PATH = "./output_eval/competitive_df.parquet"

TIER_NAME  = "All_CVs"
TIER_RANGE = (1, 50)

# ==========================
# PRINT-CONSISTENT FONT SIZING
# ==========================
# A matplotlib fontsize in points only prints at a consistent PHYSICAL size
# across figures if it's scaled by how much \includegraphics shrinks that
# figure's own canvas — a title at fontsize=20 on a 10in-wide canvas prints
# much bigger than the "same" fontsize=20 on a 26in-wide canvas, because the
# wider one gets shrunk more to fit the same \textwidth fraction. Two
# figures only render identical-looking text if they resolve to the same
# PRINTED point size, not the same source fontsize.
#
# paper.tex: a4paper, left=2cm, right=2cm -> textwidth = 21 - 2 - 2 = 17cm.
PAPER_TEXTWIDTH_IN = 17 / 2.54
INCLUDEGRAPHICS_FRAC = 0.95  # the width fraction used by ~all figures in the paper


def print_pt(target_pt, canvas_width_in, frac=INCLUDEGRAPHICS_FRAC):
    """Matplotlib source fontsize (pt) needed so that, once this figure's
    PNG (canvas_width_in inches wide) is placed via
    \\includegraphics[width=frac\\textwidth], `target_pt` is the physical
    point size it actually prints at. Always derive figure fontsizes this
    way (not raw rcParams["font.size"] + N) when the figure needs to match
    another figure's text size on the page."""
    printed_width_in = frac * PAPER_TEXTWIDTH_IN
    return target_pt * canvas_width_in / printed_width_in


# Canonical PRINTED point sizes shared by every figure in the paper, so
# "the title" (or "a tick label", "an annotation", ...) is the same
# physical size everywhere regardless of that figure's own canvas width.
# Pass these to print_pt(TARGET_X_PT, this_figure_canvas_width_in).
HEADLINE_PT      = 9.0   # a figure's single overall title / row header ("(a) ...")
PANEL_TITLE_PT   = 7.5   # a per-panel/subplot title inside a multi-panel figure
AXIS_LABEL_PT    = 7.0   # axis labels (xlabel/ylabel/cbar label)
TICK_PT_TARGET   = 6.0   # tick labels, general
DENSE_TICK_PT    = 6.0   # tick labels on dense grids (heatmap row/col model names)
ANNOTATION_PT    = 6.5   # value callouts (bar labels, scatter point labels, ...)
LEGEND_PT_TARGET = 6.5   # legend entries
DENSE_ANNOT_PT   = 4.3   # in-cell numbers on dense heatmaps (9x9 or bigger)

UNIQUE_EVALUATORS = [
    "gpt-4o-mini", "gpt-5-mini", "gpt-5",
    "gemini-3-flash-preview", "gemini-3.5-flash",
    "claude-haiku-4-5", "claude-4.6-sonnet",
    "deepseek-v3.2", "deepseek-v4-flash",
]

RAW_WRITERS = [
    "gpt-4o-mini", "gpt-5-mini", "gpt-5",
    "gemini-3-flash-preview", "gemini-3.5-flash",
    "claude-haiku-4-5", "claude-4.6-sonnet",
    "deepseek-v3.2", "deepseek-v4-flash", "deepseek-r1-8b", "llama3.1-8b",
]

MODEL_PAIRS = [f"{e}_{w}" for e in UNIQUE_EVALUATORS for w in RAW_WRITERS]

SORTED_WRITERS = [e for e in UNIQUE_EVALUATORS if e in RAW_WRITERS] + \
                 [w for w in RAW_WRITERS if w not in UNIQUE_EVALUATORS]

TITLE_MAP = {
    "cl_evaluations":    "Cover Letter Only",
    "cv_cl_evaluations": "CV + Cover Letter",
    "cv_only":           "CV Only",
}

WRITER_COLORS = {
    "gpt-4o-mini":            "#1565C0",  # OpenAI – dark blue
    "gpt-5-mini":             "#42A5F5",  # OpenAI – light blue
    "gpt-5":                  "#0D47A1",  # OpenAI – navy
    "gemini-3-flash-preview": "#66BB6A",  # Google – medium green
    "gemini-3.5-flash":       "#558B2F",  # Google – olive green
    "claude-haiku-4-5":       "#E65100",  # Anthropic – burnt orange
    "claude-4.6-sonnet":      "#BF360C",  # Anthropic – deep red-orange
    "deepseek-v3.2":          "#4A148C",  # DeepSeek – dark purple
    "deepseek-v4-flash":      "#7B1FA2",  # DeepSeek – medium purple
    "deepseek-r1-8b":         "#AB47BC",  # DeepSeek – light purple
    "llama3.1-8b":            "#5D4037",  # Meta – brown
}

MODEL_DISPLAY = {
    "gpt-4o-mini":            "GPT-4o mini",
    "gpt-5-mini":             "GPT-5 mini",
    "gpt-5":                  "GPT-5",
    "gemini-2.0-flash":       "Gemini 2.0 Flash",
    "gemini-3-flash-preview": "Gemini 3 Flash",
    "gemini-3.5-flash":       "Gemini 3.5 Flash",
    "claude-haiku-4-5":       "Claude Haiku 4.5",
    "claude-4.6-sonnet":      "Claude Sonnet 4.6",
    "deepseek-v3.2":          "DeepSeek V3.2",
    "deepseek-v4-flash":      "DeepSeek V4 Flash",
    "deepseek-r1-8b":         "DeepSeek R1 8B",
    "llama3.1-8b":            "Llama 3.1 8B",
}

DISPLAY_COLORS = {MODEL_DISPLAY[k]: v for k, v in WRITER_COLORS.items()}

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.max_open_warning": 0})

# ==========================
# SHARED HELPERS
# ==========================

def extract_first_number(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            m = re.search(r"(\d+(?:\.\d+)?)", f.read())
            return float(m.group(1)) if m else None
    except Exception as e:
        print(f"  ERROR reading {filepath}: {e}")
        return None


def detect_cv_count(path):
    max_cv = 0
    if not os.path.exists(path):
        return 0
    for root, _, files in os.walk(path):
        for fname in files:
            m = re.search(r"cv(\d+)", fname, re.IGNORECASE)
            if m:
                max_cv = max(max_cv, int(m.group(1)))
    return max_cv


def format_job_title(job_id, folder_name):
    clean = folder_name.replace(job_id, "").strip("_")
    clean = re.sub(r"_\d+$", "", clean).replace("_", " ")
    return f"{clean} ({job_id.replace('job_', '')})"


def parse_filename(filename, etype):
    """Extracts (evaluator, writer, cv_idx) from a score filename."""
    evaluator = next(
        (e for e in sorted(UNIQUE_EVALUATORS, key=len, reverse=True)
         if filename.startswith(e + "_")),
        None,
    )
    if not evaluator:
        return None, None, None

    if etype == "cv_only":
        writer = "CV_ONLY"
    else:
        remainder = filename[len(evaluator) + 1:]
        writer = next(
            (w for w in sorted(RAW_WRITERS, key=len, reverse=True)
             if remainder.startswith(w + "_") or remainder.startswith(w + ".")
             or remainder == w),
            None,
        )
        if not writer:
            return None, None, None

    m = re.search(r"cv(\d+)", filename, re.IGNORECASE)
    if not m:
        return None, None, None
    return evaluator, writer, int(m.group(1))


# ==========================
# DATA LOADING — Plots 1 & 2
# ==========================

def _scores_from_folder(folder):
    scores = {}
    if not os.path.exists(folder):
        return scores
    for fname in os.listdir(folder):
        if fname.endswith(".txt"):
            s = extract_first_number(os.path.join(folder, fname))
            if s is not None:
                scores[fname] = s
    return scores


def collect_job_data(version_folders, cv_count):
    etypes = ["cl_evaluations", "cv_cl_evaluations"]
    data = {
        et: {pair: {i: [] for i in range(1, cv_count + 2)} for pair in MODEL_PAIRS}
        for et in etypes
    }
    for _, _, v_path in version_folders:
        if not os.path.exists(v_path):
            continue
        for run_folder in sorted(os.listdir(v_path)):
            run_path = os.path.join(v_path, run_folder)
            if not os.path.isdir(run_path):
                continue
            for etype in etypes:
                for fname, score in _scores_from_folder(os.path.join(run_path, etype)).items():
                    for pair in MODEL_PAIRS:
                        if pair in fname:
                            m = re.search(r"cv(\d+)", fname, re.IGNORECASE)
                            if m:
                                idx = int(m.group(1))
                                if idx <= cv_count:
                                    data[etype][pair][idx].append(score)
    return data


def merge_jobs_data(all_jobs_data):
    etypes = ["cl_evaluations", "cv_cl_evaluations"]
    global_data = {et: {pair: {} for pair in MODEL_PAIRS} for et in etypes}
    for job_data in all_jobs_data.values():
        for et in etypes:
            for pair in MODEL_PAIRS:
                for cv_idx, scores in job_data[et][pair].items():
                    global_data[et][pair].setdefault(cv_idx, []).extend(scores)
    return global_data


# ==========================
# DATA LOADING
# ==========================

def build_master_df(base_dir, force=False):
    if not force and os.path.exists(CACHE_PATH):
        print(f"  Loading master dataframe from cache ({CACHE_PATH})")
        return pd.read_parquet(CACHE_PATH)

    print("  Building master dataframe from raw files (this takes a while)...")
    rows = []
    for job_folder in sorted(os.listdir(base_dir)):
        job_path = os.path.join(base_dir, job_folder)
        if not os.path.isdir(job_path):
            continue
        m = re.match(r"(job_\d+)", job_folder)
        if not m:
            continue
        job_id = m.group(1)
        job_title = format_job_title(job_id, job_folder)
        for run_folder in os.listdir(job_path):
            if not run_folder.startswith("run_"):
                continue
            m_run = re.search(r"run_(\d+)", run_folder)
            run_id = int(m_run.group(1)) if m_run else 0
            run_path = os.path.join(job_path, run_folder)
            for etype in ["cv_only", "cl_evaluations", "cv_cl_evaluations"]:
                eval_path = os.path.join(run_path, etype)
                if not os.path.exists(eval_path):
                    continue
                for fname in os.listdir(eval_path):
                    if not fname.endswith(".txt"):
                        continue
                    evaluator, writer, cv_idx = parse_filename(fname, etype)
                    if evaluator and writer and cv_idx:
                        score = extract_first_number(os.path.join(eval_path, fname))
                        if score is not None:
                            rows.append({
                                "Job_ID": job_id, "Job_Title": job_title,
                                "Run": run_id,
                                "Eval_Type": etype, "Evaluator": evaluator,
                                "Writer": writer, "CV_Idx": cv_idx, "Score": score,
                            })
    df = pd.DataFrame(rows)
    df.to_parquet(CACHE_PATH, index=False)
    print(f"  Cached master dataframe → {CACHE_PATH}")
    return df




def load_competitive_data(base_dir):
    if not os.path.exists(COMP_CACHE_PATH):
        print(f"  WARNING: {COMP_CACHE_PATH} not found — skipping competitive plots.")
        return pd.DataFrame()
    return pd.read_parquet(COMP_CACHE_PATH)


def calculate_leapfrog_cosine(df, evaluator, baseline_writer, target_writer):
    """Like calculate_leapfrog but incumbents/challengers are fixed by cosine-similarity
    rank (CV_Idx 1-25 = high-similarity incumbents, 26-50 = low-similarity challengers).

    The 4 replicate runs are averaged into a single per-(evaluator, job, CV_Idx)
    score BEFORE ranking/displacement is computed, matching the run-averaged score
    s_{e,j,c,w} used as the unit everywhere else in the paper. The 4 runs are not
    independent evidence about the displacement rate -- they are the same model
    answering the same prompt 4 times -- so returning one value per (job, run)
    here would inflate the effective N of any significance test built on this
    function's output by 4x. This returns at most one value per job."""
    eval_df = df[df["Evaluator"] == evaluator]
    results = []
    for job_id in eval_df["Job"].unique():
        env = eval_df[eval_df["Job"] == job_id]
        cv_cl = env[env["Type"] == "cv_cl_evaluations"]
        incumbent_cvs  = list(range(1, 26))
        challenger_cvs = list(range(26, 51))
        inc_raw  = cv_cl[(cv_cl["Writer"] == baseline_writer) & cv_cl["CV_Idx"].isin(incumbent_cvs)]
        chal_raw = cv_cl[(cv_cl["Writer"] == target_writer)  & cv_cl["CV_Idx"].isin(challenger_cvs)]
        if inc_raw.empty or chal_raw.empty:
            continue
        incumbents  = inc_raw.groupby("CV_Idx", as_index=False)["Score"].mean()
        challengers = chal_raw.groupby("CV_Idx", as_index=False)["Score"].mean()
        challengers["Type_Tag"] = "Challenger"
        pool = pd.concat([incumbents, challengers])
        pool = pool.sort_values(by=["Score", "CV_Idx"], ascending=[False, True])
        pool["New_Rank"] = range(1, len(pool) + 1)
        success = pool[(pool["Type_Tag"] == "Challenger") & (pool["New_Rank"] <= 25)]
        results.append(len(success) / len(challengers) * 100)
    return results


def calculate_leapfrog(df, evaluator, baseline_writer, target_writer):
    """The 4 replicate runs are averaged into a single per-(evaluator, job, CV_Idx)
    score BEFORE ranking/displacement is computed (both for the CV-only
    incumbent/challenger split and for the CV+CL scores), matching the
    run-averaged score s_{e,j,c,w} used as the unit everywhere else in the
    paper. The 4 runs are not independent evidence about the displacement
    rate -- they are the same model answering the same prompt 4 times -- so
    returning one value per (job, run) here would inflate the effective N of
    any significance test built on this function's output by 4x. This
    returns at most one value per job."""
    eval_df = df[df["Evaluator"] == evaluator]
    results = []
    for job_id in eval_df["Job"].unique():
        env = eval_df[eval_df["Job"] == job_id]
        cv_only = env[env["Type"] == "cv_only"]
        cv_only_avg = cv_only.groupby("CV_Idx", as_index=False)["Score"].mean()
        if len(cv_only_avg) != 50:
            continue
        cv_only_avg = cv_only_avg.sort_values(by=["Score", "CV_Idx"], ascending=[False, True])
        cv_only_avg["Baseline_Rank"] = range(1, 51)
        incumbent_cvs  = cv_only_avg[cv_only_avg["Baseline_Rank"] <= 25]["CV_Idx"].tolist()
        challenger_cvs = cv_only_avg[cv_only_avg["Baseline_Rank"] > 25]["CV_Idx"].tolist()
        cv_cl = env[env["Type"] == "cv_cl_evaluations"]
        inc_raw  = cv_cl[(cv_cl["Writer"] == baseline_writer) & cv_cl["CV_Idx"].isin(incumbent_cvs)]
        chal_raw = cv_cl[(cv_cl["Writer"] == target_writer)  & cv_cl["CV_Idx"].isin(challenger_cvs)]
        if inc_raw.empty or chal_raw.empty:
            continue
        incumbents  = inc_raw.groupby("CV_Idx", as_index=False)["Score"].mean()
        challengers = chal_raw.groupby("CV_Idx", as_index=False)["Score"].mean()
        challengers["Type_Tag"] = "Challenger"
        pool = pd.concat([incumbents, challengers])
        pool = pool.sort_values(by=["Score", "CV_Idx"], ascending=[False, True])
        pool["New_Rank"] = range(1, len(pool) + 1)
        success = pool[(pool["Type_Tag"] == "Challenger") & (pool["New_Rank"] <= 25)]
        results.append(len(success) / len(challengers) * 100)
    return results


def _gap_matrices_from_df(tier_df, etype):
    """Compute heatmap-gap matrices from master_df (same logic as the raw-file approach)."""
    sub = tier_df[tier_df["Eval_Type"] == etype]
    if sub.empty:
        return None, None, None, None

    # Mean score per (Evaluator, CV_Idx, Writer) across all jobs/runs
    pair_mean = (sub.groupby(["Evaluator", "CV_Idx", "Writer"])["Score"]
                   .mean().reset_index().rename(columns={"Score": "PairMean"}))
    # Mean score per (Evaluator, CV_Idx) across all writers/jobs/runs
    all_mean = (sub.groupby(["Evaluator", "CV_Idx"])["Score"]
                  .mean().reset_index().rename(columns={"Score": "AllMean"}))
    merged = pair_mean.merge(all_mean, on=["Evaluator", "CV_Idx"])
    merged["Gap"] = merged["PairMean"] - merged["AllMean"]

    data_rows, annot_rows = [], []
    for eval_name in UNIQUE_EVALUATORS:
        row_vals, row_annots = [], []
        for write_name in SORTED_WRITERS:
            gaps = merged.loc[
                (merged["Evaluator"] == eval_name) & (merged["Writer"] == write_name), "Gap"
            ].tolist()
            if gaps:
                mean_gap = np.mean(gaps)
                is_sig = False
                if len(gaps) > 1 and np.var(gaps) != 0:
                    try:
                        _, p = stats.ttest_1samp(gaps, 0)
                        is_sig = p < 0.05
                    except Exception:
                        pass
                row_vals.append(mean_gap)
                row_annots.append(f"{mean_gap:.2f}{'(*)' if is_sig else ''}")
            else:
                row_vals.append(0.0)
                row_annots.append("0.00")
        data_rows.append(row_vals)
        annot_rows.append(row_annots)

    df = pd.DataFrame(data_rows, index=UNIQUE_EVALUATORS, columns=SORTED_WRITERS)
    df_annot = pd.DataFrame(annot_rows, index=UNIQUE_EVALUATORS, columns=SORTED_WRITERS)
    col_means = df.mean(axis=0)
    ew  = [w for w in df.columns if w in UNIQUE_EVALUATORS]
    new = [w for w in df.columns if w not in UNIQUE_EVALUATORS]
    sc  = (sorted(ew,  key=lambda w: col_means[w], reverse=True)
           + sorted(new, key=lambda w: col_means[w], reverse=True))
    sr  = [w for w in sc if w in UNIQUE_EVALUATORS]
    df       = df.loc[sr, sc]
    df_annot = df_annot.loc[sr, sc]
    df.loc["Avg."]       = df.mean(axis=0)
    df_annot.loc["Avg."] = [f"{df.loc['Avg.', c]:.2f}" for c in df.columns]
    return df, df_annot, sr, sc


def plot_heatmap_gap(tier_df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    etypes = ["cl_evaluations", "cv_cl_evaluations"]

    built = {}
    for etype in etypes:
        result = _gap_matrices_from_df(tier_df, etype)
        if result[0] is None:
            continue
        built[etype] = result

    # Symmetric scale: use max absolute value across both etypes (non-Average rows only)
    max_abs = max(
        built[et][0].loc[built[et][2], :].abs().max().max()
        for et in etypes
    )
    max_abs = np.ceil(max_abs * 20) / 20  # round up to nearest 0.05
    shared_vmin, shared_vmax = -max_abs, max_abs

    # --- Second pass: plot using shared scale ---
    fs = plt.rcParams["font.size"] + 3
    for etype in etypes:
        df, df_annot, sorted_rows, sorted_cols = built[etype]
        df.to_csv(os.path.join(save_dir, f"heatmap_gap_{etype}.csv"))

        df_plot      = df.rename(index=MODEL_DISPLAY, columns=MODEL_DISPLAY)
        df_annot_plot = df_annot.rename(index=MODEL_DISPLAY, columns=MODEL_DISPLAY)

        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(
            df_plot, annot=df_annot_plot, fmt="", cmap="RdBu_r", center=0,
            vmin=shared_vmin, vmax=shared_vmax,
            linewidths=0.5, linecolor="gray",
            annot_kws={"size": fs},
            cbar_kws={"label": "Preference Gap (score units, 0–10 scale)"},
        )
        ax.axhline(len(sorted_rows), color="black", linewidth=2)
        # Thick border on diagonal cells (evaluator == writer → self-preference)
        for e in sorted_rows:
            if e in sorted_cols:
                r = list(sorted_rows).index(e)
                c = list(sorted_cols).index(e)
                ax.add_patch(plt.Rectangle(
                    (c, r), 1, 1, fill=False, edgecolor="black", linewidth=3
                ))
        plt.title(
            f"Relative Preference Gap | {TITLE_MAP[etype]}\n"
            "Columns sorted left → right by most preferred writer  |  (*) = p < 0.05  |  Bold border = self-evaluation",
            fontsize=fs + 5,
        )
        plt.xlabel("Writer Model", fontsize=fs + 4)
        plt.ylabel("Evaluator Model", fontsize=fs + 4)
        ax.tick_params(axis="y", labelsize=fs)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=fs, fontweight="bold")
        for tick in ax.get_xticklabels():
            tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=fs, fontweight="bold")
        for tick in ax.get_yticklabels():
            tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
        ax.collections[0].colorbar.ax.tick_params(labelsize=fs)
        ax.collections[0].colorbar.ax.yaxis.label.set_size(fs + 4)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"heatmap_gap_{etype}.png"))
        plt.close()
        print(f"  Saved heatmap_gap_{etype}.png")


def plot_heatmap_gap_combined(tier_df, save_dir):
    """Both heatmaps side by side, each with its own ordering, shared colorbar."""
    os.makedirs(save_dir, exist_ok=True)
    etypes = ["cl_evaluations", "cv_cl_evaluations"]

    built = {et: _gap_matrices_from_df(tier_df, et) for et in etypes}
    if any(v[0] is None for v in built.values()):
        return

    # Shared symmetric scale
    max_abs = max(
        built[et][0].loc[built[et][2], :].abs().max().max()
        for et in etypes
    )
    max_abs = np.ceil(max_abs * 20) / 20
    shared_vmin, shared_vmax = -max_abs, max_abs

    canvas_w = 24
    title_pt    = print_pt(PANEL_TITLE_PT, canvas_w)
    label_pt    = print_pt(AXIS_LABEL_PT, canvas_w)
    tick_pt     = print_pt(DENSE_TICK_PT, canvas_w)
    square_pt   = print_pt(7.5, canvas_w)
    annot_pt    = print_pt(5.5, canvas_w)
    headline_pt = print_pt(HEADLINE_PT, canvas_w)

    fig, axes = plt.subplots(1, 2, figsize=(canvas_w, 9), constrained_layout=True)
    fig.get_layout_engine().set(wspace=0.08)

    for ax_idx, (ax, etype) in enumerate(zip(axes, etypes)):
        df, df_annot, sorted_rows, sorted_cols = built[etype]
        df_plot       = df.rename(index=MODEL_DISPLAY, columns=MODEL_DISPLAY)
        df_annot_plot = df_annot.rename(index=MODEL_DISPLAY, columns=MODEL_DISPLAY)
        sns.heatmap(
            df_plot, annot=df_annot_plot, fmt="", ax=ax,
            cmap="RdBu_r", center=0, vmin=shared_vmin, vmax=shared_vmax,
            linewidths=0.5, linecolor="gray",
            annot_kws={"size": annot_pt},
            cbar=False,
        )
        # Significance shown as bold numbers (space here is too tight for an
        # asterisk to read cleanly) — this is the one deliberate exception to
        # "bold is title-only" elsewhere in the paper.
        for text in ax.texts:
            t = text.get_text()
            if "(*)" in t:
                text.set_text(t.replace("(*)", ""))
                text.set_fontweight("bold")
        # Separator before Average row
        ax.axhline(len(sorted_rows), color="black", linewidth=2)
        # Thick border on diagonal cells (self-evaluation) — positions are index-based
        for e in sorted_rows:
            if e in sorted_cols:
                r = list(sorted_rows).index(e)
                c = list(sorted_cols).index(e)
                ax.add_patch(plt.Rectangle(
                    (c, r), 1, 1, fill=False, edgecolor="black", linewidth=3
                ))
        ax.set_title(TITLE_MAP[etype], fontsize=title_pt, fontweight="bold", pad=10)
        ax.set_xlabel("Writer Model", fontsize=label_pt, labelpad=12)
        ax.set_ylabel("Evaluator Model" if ax_idx == 0 else "", fontsize=label_pt)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=tick_pt)
        for tick in ax.get_xticklabels():
            tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
        ytick_info = [(t.get_text(), DISPLAY_COLORS.get(t.get_text(), "black"))
                      for t in ax.get_yticklabels()]
        ax.set_yticklabels(
            ["Avg." if name == "Avg." else "■" for name, _ in ytick_info],
            rotation=0,
        )
        for tick, (name, color) in zip(ax.get_yticklabels(), ytick_info):
            tick.set_color("black" if tick.get_text() == "Avg." else color)
            tick.set_fontsize(tick_pt if tick.get_text() == "Avg." else square_pt)

    # Single shared colorbar to the right
    sm = plt.cm.ScalarMappable(
        cmap="RdBu_r", norm=plt.Normalize(vmin=shared_vmin, vmax=shared_vmax)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=1.0, pad=0.02)
    cbar.set_label("Relative Preference Gap, 0–10 scale", fontsize=label_pt)
    cbar.ax.tick_params(labelsize=tick_pt)

    fig.suptitle(
        "Relative Preference Gap  |  Bold = p < 0.05",
        fontsize=headline_pt, fontweight="bold",
    )
    plt.savefig(os.path.join(save_dir, "heatmap_gap_combined.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved heatmap_gap_combined.png")


def plot_win_matrix(df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    etype = "cv_cl_evaluations"
    writers = sorted([w for w in df["Writer"].unique() if w != "CV_ONLY"])
    n = len(writers)

    subset = df[df["Eval_Type"] == etype]
    if subset.empty:
        return

    win_matrix = np.zeros((n, n))
    match_count = np.zeros((n, n))
    for (job, cv, evaluator), group in subset.groupby(["Job_ID", "CV_Idx", "Evaluator"]):
        scores = dict(zip(group["Writer"], group["Score"]))
        for i, w1 in enumerate(writers):
            for j, w2 in enumerate(writers):
                if i == j or evaluator in (w1, w2):
                    continue
                s1, s2 = scores.get(w1), scores.get(w2)
                if s1 is not None and s2 is not None:
                    match_count[i][j] += 1
                    if s1 > s2:
                        win_matrix[i][j] += 1
                    elif s1 == s2:
                        win_matrix[i][j] += 0.5

    win_rate = np.full((n, n), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        win_rate = (win_matrix / match_count) * 100
    np.fill_diagonal(win_rate, np.nan)

    df_win = pd.DataFrame(win_rate, index=writers, columns=writers)
    df_win["Avg."] = df_win.mean(axis=1, skipna=True)
    # Sort model rows only, then append Avg. at the bottom
    df_win = df_win.sort_values("Avg.", ascending=False)
    sorted_models = list(df_win.index)
    df_win = df_win[sorted_models + ["Avg."]]
    avg_row = df_win.mean(axis=0, skipna=True)
    avg_row.name = "Avg."
    df_win = pd.concat([df_win, avg_row.to_frame().T])

    df_win.to_csv(os.path.join(save_dir, f"win_matrix_UNBIASED_{etype}.csv"))

    # Custom annotation: "%" suffix, "—" for NaN (diagonal)
    annot_df = df_win.copy().astype(object)
    for r in annot_df.index:
        for c in annot_df.columns:
            val = df_win.loc[r, c]
            annot_df.loc[r, c] = "—" if pd.isna(val) else f"{val:.0f}%"

    mask = pd.DataFrame(False, index=df_win.index, columns=df_win.columns)
    for w in writers:
        if w in df_win.index and w in df_win.columns:
            mask.loc[w, w] = True

    # Precompute diagonal positions (raw names) before renaming
    diag_pos = [(list(df_win.index).index(w), list(df_win.columns).index(w))
                for w in writers if w in df_win.index and w in df_win.columns]
    n_models = sum(1 for r in df_win.index if r != "Avg.")

    df_win   = df_win.rename(index=MODEL_DISPLAY, columns=MODEL_DISPLAY)
    annot_df = annot_df.rename(index=MODEL_DISPLAY, columns=MODEL_DISPLAY)
    mask     = mask.rename(index=MODEL_DISPLAY, columns=MODEL_DISPLAY)

    fs = plt.rcParams["font.size"] + 3
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        df_win, annot=annot_df, fmt="", cmap="RdBu_r", vmin=0, vmax=100, center=50,
        cbar_kws={"label": "Win Rate %"}, linewidths=0.5, linecolor="gray",
        mask=mask, ax=ax, annot_kws={"size": fs},
    )
    # Grey patches + "—" text for diagonal
    for ri, ci in diag_pos:
        ax.add_patch(plt.Rectangle((ci, ri), 1, 1, fill=True, color="lightgrey", lw=0))
        ax.text(ci + 0.5, ri + 0.5, "—", ha="center", va="center", fontsize=fs, color="grey")

    # Thick separator before Avg row/col
    ax.axhline(n_models, color="black", linewidth=2)
    ax.axvline(len(df_win.columns) - 1, color="black", linewidth=2)

    plt.title(f"Head-to-Head Win Rate | {TITLE_MAP[etype]}", fontsize=fs + 5)
    plt.xlabel("Opponent Writer Model", fontsize=fs + 4)
    plt.ylabel("Writer Model", fontsize=fs + 4)
    ax.tick_params(axis="y", labelsize=fs)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=fs, fontweight="bold")
    for tick in ax.get_xticklabels():
        tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=fs, fontweight="bold")
    for tick in ax.get_yticklabels():
        tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
    ax.collections[0].colorbar.ax.tick_params(labelsize=fs)
    ax.collections[0].colorbar.ax.yaxis.label.set_size(fs + 4)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"win_matrix_UNBIASED_{etype}.png"))
    plt.close()
    print(f"  Saved win_matrix_UNBIASED_{etype}.png")


def plot_net_advantage(delta_matrix, p_matrix, raw_matrix, writers, save_dir, std_matrix=None):
    """Pooled (across all 9 evaluators) Net Competitive Advantage heatmap.
    Style matches win_net_combined's right panel: RdBu_r centered at 0, grey
    em-dash on the control diagonal, color-coded square y-ticks, no bold, no
    inline '%' (only in the colorbar label). When std_matrix is given (the
    across-evaluator std of the net rate per cell -- see main()'s std_mat,
    not run-to-run noise), each off-diagonal cell shows "+X.X" with the
    "±Y.Y" on its own line underneath, one decimal each. The Avg.
    row/column (no std, a mean-of-means) gets a 50% larger font since it
    isn't competing with a second line of text."""
    os.makedirs(save_dir, exist_ok=True)
    N = len(writers)
    canvas_w = 16
    title_pt  = print_pt(PANEL_TITLE_PT, canvas_w)
    label_pt  = print_pt(AXIS_LABEL_PT, canvas_w)
    tick_pt   = print_pt(DENSE_TICK_PT, canvas_w)
    square_pt = print_pt(7.5, canvas_w)
    annot_pt  = print_pt(5.5, canvas_w)
    headline_pt = print_pt(HEADLINE_PT, canvas_w)

    ext_writers = list(writers) + ["Avg."]

    masked_delta = delta_matrix.copy()
    np.fill_diagonal(masked_delta, np.nan)
    row_means    = np.nanmean(masked_delta, axis=1)
    col_means    = np.nanmean(masked_delta, axis=0)
    overall_mean = np.nanmean(masked_delta)

    ext_delta = np.zeros((N + 1, N + 1))
    ext_delta[:N, :N] = delta_matrix
    ext_delta[:N, N]  = row_means
    ext_delta[N, :N]  = col_means
    ext_delta[N, N]   = overall_mean

    annot = []
    for i in range(N + 1):
        row = []
        for j in range(N + 1):
            val, sign = ext_delta[i, j], "+" if ext_delta[i, j] > 0 else ""
            if i == j and i < N:
                row.append("—")
            elif i < N and j < N and std_matrix is not None and not np.isnan(std_matrix[i, j]):
                row.append(f"{sign}{val:.1f}\n±{std_matrix[i, j]:.1f}")
            else:
                row.append(f"{sign}{val:.1f}")
        annot.append(row)
    annot = np.array(annot)

    df_m = pd.DataFrame(ext_delta, index=ext_writers, columns=ext_writers)
    sort_vals     = pd.Series(col_means, index=writers).sort_values(ascending=False)
    sorted_models = sort_vals.index.tolist()
    df_m = df_m.loc[sorted_models + ["Avg."], sorted_models + ["Avg."]]
    old_idx = {w: i for i, w in enumerate(writers)}
    new_ri  = [old_idx[w] for w in sorted_models] + [N]
    new_ci  = [old_idx[w] for w in sorted_models] + [N]
    annot   = annot[np.ix_(new_ri, new_ci)]

    df_m.to_csv(os.path.join(save_dir, "net_advantage_matrix_ALL_COMBINED.csv"))

    df_m_plot = df_m.rename(index=MODEL_DISPLAY, columns=MODEL_DISPLAY)

    fig, ax = plt.subplots(figsize=(canvas_w, 12.5), constrained_layout=True)
    sns.heatmap(df_m_plot, annot=annot, fmt="", ax=ax, cmap="RdBu_r", center=0,
                linewidths=0.5, linecolor="gray", cbar=False,
                annot_kws={"size": annot_pt})
    # seaborn adds cell text in row-major order matching df_m_plot, which is
    # already sorted so the Avg. row/column are always last (index N).
    n_cols = N + 1
    for idx, text in enumerate(ax.texts):
        if text.get_text() == "—":
            text.set_color("grey")
        row, col = idx // n_cols, idx % n_cols
        if row == N or col == N:
            text.set_fontsize(annot_pt * 1.5)
        else:
            text.set_fontsize(annot_pt * 1.25)
    ax.axhline(N, color="black", linewidth=2)
    ax.axvline(N, color="black", linewidth=2)
    ax.set_title("Net Competitive Advantage over Control", fontsize=title_pt, fontweight="bold", pad=10)
    ax.set_ylabel("High-Fit Group — Writer Model", fontsize=label_pt)
    ax.set_xlabel("Moderate-Fit Group — Writer Model", fontsize=label_pt, labelpad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=tick_pt)
    for tick in ax.get_xticklabels():
        tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
    ytick_info = [(t.get_text(), DISPLAY_COLORS.get(t.get_text(), "black")) for t in ax.get_yticklabels()]
    ax.set_yticklabels(["Avg." if name == "Avg." else "■" for name, _ in ytick_info], rotation=0)
    for tick, (name, color) in zip(ax.get_yticklabels(), ytick_info):
        tick.set_color("black" if tick.get_text() == "Avg." else color)
        tick.set_fontsize(tick_pt if tick.get_text() == "Avg." else square_pt)

    net_vmax = np.nanmax(np.abs(ext_delta[:-1, :-1]))
    sm = cm.ScalarMappable(cmap="RdBu_r", norm=mcolors.Normalize(vmin=-net_vmax, vmax=net_vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.8)
    cb.set_label("Net Advantage over Control, %", fontsize=label_pt)
    cb.ax.tick_params(labelsize=tick_pt)

    fig.suptitle("Net Competitive Advantage (Pooled Across Evaluators)",
                 fontsize=headline_pt, fontweight="bold")
    plt.savefig(os.path.join(save_dir, "net_advantage_matrix_ALL_COMBINED.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved net_advantage_matrix_ALL_COMBINED.png")


def plot_net_advantage_extreme_evaluators(comp_df, writers, evaluators, save_dir,
                                           fname="net_advantage_extreme_evaluators.png"):
    """Side-by-side Net Competitive Advantage heatmaps for INDIVIDUAL
    evaluators (self-evaluation triples included throughout, matching
    calculate_leapfrog's existing behavior for the pooled matrix -- only the
    separate win-rate analysis excludes self-evaluation), rather than pooled
    across all 9 as in net_advantage_matrix_ALL_COMBINED.png /
    win_net_combined.png. Appendix companion surfacing the worst-case
    single-evaluator numbers: the pooled matrix's headline (+7.3%, ~1.8/25)
    is a cross-evaluator mean, and individual evaluators can go well beyond
    it on specific pairings (GPT-4o mini and DeepSeek V3.2 both reach ~+13%,
    ~3.2/25, chosen here as the two most extreme by peak cell and by summed
    |net| across their whole matrix respectively).

    Style matches win_net_combined's right panel exactly (same cmap, grey
    em-dash control diagonal, color-coded square y-ticks, no inline '%', no
    bold): no significance marking here at all (dropped rather than
    replicating the reference's bold-for-p<0.05 convention), since the point
    of this figure is just the raw worst-case magnitude per evaluator."""
    os.makedirs(save_dir, exist_ok=True)
    N = len(writers)
    # Font sizes are computed from the ORIGINAL canvas_w (so print_pt keeps
    # them at the print size already set), but the actual figure is drawn
    # 15% wider (fig_canvas_w): since print_pt's target print size doesn't
    # depend on canvas width, widening only the figure -- not the font
    # calculation -- is what actually gives each cell more room relative to
    # its (unchanged) annotation text.
    canvas_w = 28
    fig_canvas_w = canvas_w * 1.15
    title_pt    = print_pt(PANEL_TITLE_PT, canvas_w)
    label_pt    = print_pt(AXIS_LABEL_PT, canvas_w)
    tick_pt     = print_pt(DENSE_TICK_PT, canvas_w)
    square_pt   = print_pt(7.5, canvas_w)
    annot_pt    = print_pt(5.5, canvas_w) * 0.85  # ~15% smaller than the reference's own in-cell size
    headline_pt = print_pt(HEADLINE_PT, canvas_w)

    fig, axes = plt.subplots(1, len(evaluators), figsize=(fig_canvas_w, 11))

    for ax, evaluator in zip(axes, evaluators):
        raw = np.full((N, N), np.nan)
        for i, b in enumerate(writers):
            for j, t in enumerate(writers):
                vals = calculate_leapfrog(comp_df, evaluator, b, t)
                raw[i, j] = np.mean(vals) if vals else np.nan
        rounded_raw = np.round(raw, 1)

        delta = np.full((N, N), np.nan)
        for i in range(N):
            ctrl_val = rounded_raw[i, i]
            for j in range(N):
                delta[i, j] = rounded_raw[i, j] - ctrl_val

        masked_delta = delta.copy()
        np.fill_diagonal(masked_delta, np.nan)
        row_means, col_means, overall_mean = (np.nanmean(masked_delta, axis=1),
                                               np.nanmean(masked_delta, axis=0),
                                               np.nanmean(masked_delta))
        ext_delta = np.zeros((N + 1, N + 1))
        ext_delta[:N, :N], ext_delta[:N, N], ext_delta[N, :N], ext_delta[N, N] = \
            delta, row_means, col_means, overall_mean

        annot = []
        for i in range(N + 1):
            row = []
            for j in range(N + 1):
                val, sign = ext_delta[i, j], "+" if ext_delta[i, j] > 0 else ""
                row.append("—" if (i == j and i < N) else f"{sign}{val:.1f}")
            annot.append(row)
        annot = np.array(annot)

        ext_writers = writers + ["Avg."]
        sorted_models = pd.Series(col_means, index=writers).sort_values(ascending=False).index.tolist()
        df_m = pd.DataFrame(ext_delta, index=ext_writers, columns=ext_writers)
        df_m = df_m.loc[sorted_models + ["Avg."], sorted_models + ["Avg."]]
        old_idx = {w: i for i, w in enumerate(writers)}
        order = [old_idx[w] for w in sorted_models] + [N]
        annot = annot[np.ix_(order, order)]
        df_m_plot = df_m.rename(index=MODEL_DISPLAY, columns=MODEL_DISPLAY)
        df_m.to_csv(os.path.join(save_dir, f"net_advantage_extreme_evaluators_{evaluator}.csv"))

        sns.heatmap(df_m_plot, annot=annot, fmt="", cmap="RdBu_r", center=0,
                    linewidths=0.5, linecolor="gray", cbar=False,
                    annot_kws={"size": annot_pt}, ax=ax)
        for text in ax.texts:
            if text.get_text() == "—":
                text.set_color("grey")
        ax.axhline(N, color="black", linewidth=2)
        ax.axvline(N, color="black", linewidth=2)
        ax.set_title(f"Evaluator: {MODEL_DISPLAY.get(evaluator, evaluator)}", fontsize=title_pt, fontweight="bold", pad=10)
        ax.set_ylabel("High-Fit Group — Writer Model", fontsize=label_pt)
        ax.set_xlabel("Moderate-Fit Group — Writer Model", fontsize=label_pt, labelpad=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=tick_pt)
        for tick in ax.get_xticklabels():
            tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
        ytick_info = [(t.get_text(), DISPLAY_COLORS.get(t.get_text(), "black")) for t in ax.get_yticklabels()]
        ax.set_yticklabels(["Avg." if name == "Avg." else "■" for name, _ in ytick_info], rotation=0)
        for tick, (name, color) in zip(ax.get_yticklabels(), ytick_info):
            tick.set_color("black" if tick.get_text() == "Avg." else color)
            tick.set_fontsize(tick_pt if tick.get_text() == "Avg." else square_pt)

        net_vmax = np.nanmax(np.abs(ext_delta[:-1, :-1]))
        sm = cm.ScalarMappable(cmap="RdBu_r", norm=mcolors.Normalize(vmin=-net_vmax, vmax=net_vmax))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.8)
        cb.set_label("Net Advantage over Control, %", fontsize=label_pt)
        cb.ax.tick_params(labelsize=tick_pt)

    fig.suptitle("Net Competitive Advantage for Individual Evaluators",
                 fontsize=headline_pt, fontweight="bold", y=1.02)
    plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")


def plot_win_net_combined(tier_df, delta_matrix, p_matrix, raw_matrix, writers_net, save_dir,
                          group_labels=("High-Fit (CV-only)", "Moderate-Fit (CV-only)"),
                          fname="win_net_combined.png", std_matrix=None):
    """Win-rate matrix (left) and net advantage matrix (right) side by side."""
    os.makedirs(save_dir, exist_ok=True)
    canvas_w = 28
    title_pt    = print_pt(PANEL_TITLE_PT, canvas_w)
    label_pt    = print_pt(AXIS_LABEL_PT, canvas_w)
    tick_pt     = print_pt(DENSE_TICK_PT, canvas_w)
    square_pt   = print_pt(7.5, canvas_w)
    annot_pt    = print_pt(5.5, canvas_w)
    headline_pt = print_pt(HEADLINE_PT, canvas_w)

    # ── Win matrix data ────────────────────────────────────────────────────────
    etype   = "cv_cl_evaluations"
    subset  = tier_df[tier_df["Eval_Type"] == etype]
    writers_win = sorted([w for w in subset["Writer"].unique() if w != "CV_ONLY"])
    n_win = len(writers_win)

    win_matrix  = np.zeros((n_win, n_win))
    match_count = np.zeros((n_win, n_win))
    for (job, cv, evaluator), group in subset.groupby(["Job_ID", "CV_Idx", "Evaluator"]):
        scores = dict(zip(group["Writer"], group["Score"]))
        for i, w1 in enumerate(writers_win):
            for j, w2 in enumerate(writers_win):
                if i == j or evaluator in (w1, w2):
                    continue
                s1, s2 = scores.get(w1), scores.get(w2)
                if s1 is not None and s2 is not None:
                    match_count[i][j] += 1
                    if s1 > s2:
                        win_matrix[i][j] += 1
                    elif s1 == s2:
                        win_matrix[i][j] += 0.5

    with np.errstate(divide="ignore", invalid="ignore"):
        win_rate = (win_matrix / match_count) * 100
    np.fill_diagonal(win_rate, np.nan)

    df_win = pd.DataFrame(win_rate, index=writers_win, columns=writers_win)
    df_win["Avg."] = df_win.mean(axis=1, skipna=True)
    df_win = df_win.sort_values("Avg.", ascending=False)
    sorted_win_models = list(df_win.index)
    df_win = df_win[sorted_win_models + ["Avg."]]
    avg_row_win = df_win.mean(axis=0, skipna=True)
    avg_row_win.name = "Avg."
    df_win = pd.concat([df_win, avg_row_win.to_frame().T])

    annot_win = df_win.copy().astype(object)
    for r in annot_win.index:
        for c in annot_win.columns:
            val = df_win.loc[r, c]
            annot_win.loc[r, c] = "—" if pd.isna(val) else f"{val:.0f}"

    mask_win = pd.DataFrame(False, index=df_win.index, columns=df_win.columns)
    for w in writers_win:
        if w in df_win.index and w in df_win.columns:
            mask_win.loc[w, w] = True

    # ── Net advantage data ─────────────────────────────────────────────────────
    N = len(writers_net)
    masked_delta = delta_matrix.copy()
    np.fill_diagonal(masked_delta, np.nan)
    row_means    = np.nanmean(masked_delta, axis=1)
    col_means    = np.nanmean(masked_delta, axis=0)
    overall_mean = np.nanmean(masked_delta)

    ext_delta = np.zeros((N + 1, N + 1))
    ext_delta[:N, :N] = delta_matrix
    ext_delta[:N, N]  = row_means
    ext_delta[N, :N]  = col_means
    ext_delta[N, N]   = overall_mean

    annot_net = []
    for i in range(N + 1):
        row = []
        for j in range(N + 1):
            val, sign = ext_delta[i, j], "+" if ext_delta[i, j] > 0 else ""
            if i == j and i < N:
                row.append("—")
            elif i < N and j < N and std_matrix is not None and not np.isnan(std_matrix[i, j]):
                row.append(f"{sign}{val:.1f}\n±{std_matrix[i, j]:.1f}")
            else:
                row.append(f"{sign}{val:.1f}")
        annot_net.append(row)
    annot_net = np.array(annot_net)

    ext_writers = list(writers_net) + ["Avg."]
    sort_vals     = pd.Series(col_means, index=writers_net).sort_values(ascending=False)
    sorted_net    = sort_vals.index.tolist()
    df_net        = pd.DataFrame(ext_delta, index=ext_writers, columns=ext_writers)
    df_net        = df_net.loc[sorted_net + ["Avg."], sorted_net + ["Avg."]]
    old_idx = {w: i for i, w in enumerate(writers_net)}
    new_ri  = [old_idx[w] for w in sorted_net] + [N]
    new_ci  = [old_idx[w] for w in sorted_net] + [N]
    annot_net = annot_net[np.ix_(new_ri, new_ci)]

    # Precompute diagonal positions (raw names) before renaming
    diag_win = [(list(df_win.index).index(w), list(df_win.columns).index(w))
                for w in writers_win if w in df_win.index and w in df_win.columns]
    n_win_models = len(sorted_win_models)

    df_win.to_csv(os.path.join(save_dir, f"{os.path.splitext(fname)[0]}_win_rate.csv"))
    df_net.to_csv(os.path.join(save_dir, f"{os.path.splitext(fname)[0]}_net_advantage.csv"))

    df_win   = df_win.rename(index=MODEL_DISPLAY, columns=MODEL_DISPLAY)
    annot_win = annot_win.rename(index=MODEL_DISPLAY, columns=MODEL_DISPLAY)
    mask_win  = mask_win.rename(index=MODEL_DISPLAY, columns=MODEL_DISPLAY)
    df_net    = df_net.rename(index=MODEL_DISPLAY, columns=MODEL_DISPLAY)

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 11), constrained_layout=True)
    fig.get_layout_engine().set(wspace=0.08)

    # Left: win matrix
    sns.heatmap(
        df_win, annot=annot_win, fmt="", ax=ax1,
        cmap="RdBu_r", vmin=0, vmax=100, center=50,
        linewidths=0.5, linecolor="gray",
        mask=mask_win, cbar=False,
        annot_kws={"size": annot_pt},
    )
    for ri, ci in diag_win:
        ax1.add_patch(plt.Rectangle((ci, ri), 1, 1, fill=True, color="lightgrey", lw=0))
        ax1.text(ci + 0.5, ri + 0.5, "—", ha="center", va="center", fontsize=tick_pt, color="grey")
    ax1.axhline(n_win_models, color="black", linewidth=2)
    ax1.axvline(len(df_win.columns) - 1, color="black", linewidth=2)
    ax1.set_title("Head-to-Head Win Rate", fontsize=title_pt, fontweight="bold", pad=10)
    ax1.set_xlabel("Opponent Writer Model", fontsize=label_pt, labelpad=12)
    ax1.set_ylabel("Writer Model", fontsize=label_pt)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=tick_pt)
    for tick in ax1.get_xticklabels():
        tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
    ytick_info1 = [(t.get_text(), DISPLAY_COLORS.get(t.get_text(), "black"))
                   for t in ax1.get_yticklabels()]
    ax1.set_yticklabels(
        ["Avg." if name == "Avg." else "■" for name, _ in ytick_info1],
        rotation=0,
    )
    for tick, (name, color) in zip(ax1.get_yticklabels(), ytick_info1):
        tick.set_color("black" if tick.get_text() == "Avg." else color)
        tick.set_fontsize(tick_pt if tick.get_text() == "Avg." else square_pt)
    sm1 = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(0, 100))
    sm1.set_array([])
    cb1 = fig.colorbar(sm1, ax=ax1, shrink=0.8)
    cb1.set_label("Win Rate, %", fontsize=label_pt)
    cb1.ax.tick_params(labelsize=tick_pt)

    # Right: net advantage
    # This panel occupies only about half of the figure's 28in canvas, but
    # cell text (now two lines, "+X.X" / "±Y.Y", since std_matrix was added)
    # needs sizing against its OWN width, not the full combined canvas --
    # otherwise the two-line annotations collide into their neighbors the
    # same way the fairness heatmap's did. net_annot_pt mirrors
    # plot_net_advantage's own canvas_w=16 (a close match to this panel's
    # actual rendered width).
    net_annot_pt = print_pt(5.5, canvas_w / 2)
    net_vmax = np.nanmax(np.abs(ext_delta[:-1, :-1]))  # ignore AVERAGE row/col for scale
    sns.heatmap(
        df_net, annot=annot_net, fmt="", ax=ax2,
        cmap="RdBu_r", center=0,
        linewidths=0.5, linecolor="gray",
        cbar=False,
        annot_kws={"size": net_annot_pt},
    )
    # seaborn adds cell text in row-major order matching df_net, which is
    # already sorted so the Avg. row/column are always last (index N).
    n_cols_net = N + 1
    for idx, text in enumerate(ax2.texts):
        if text.get_text() == "—":
            text.set_color("grey")
        row, col = idx // n_cols_net, idx % n_cols_net
        if row == N or col == N:
            text.set_fontsize(net_annot_pt * 1.5)
        else:
            text.set_fontsize(net_annot_pt * 1.25)
    ax2.axhline(N, color="black", linewidth=2)
    ax2.axvline(N, color="black", linewidth=2)
    ax2.set_title("Net Competitive Advantage over Control", fontsize=title_pt, fontweight="bold", pad=10)
    ax2.set_xlabel(f"{group_labels[1]} Group — Writer Model", fontsize=label_pt, labelpad=12)
    ax2.set_ylabel(f"{group_labels[0]} Group — Writer Model", fontsize=label_pt)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right", fontsize=tick_pt)
    for tick in ax2.get_xticklabels():
        tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
    ytick_info2 = [(t.get_text(), DISPLAY_COLORS.get(t.get_text(), "black"))
                   for t in ax2.get_yticklabels()]
    ax2.set_yticklabels(
        ["Avg." if name == "Avg." else "■" for name, _ in ytick_info2],
        rotation=0,
    )
    for tick, (name, color) in zip(ax2.get_yticklabels(), ytick_info2):
        tick.set_color("black" if tick.get_text() == "Avg." else color)
        tick.set_fontsize(tick_pt if tick.get_text() == "Avg." else square_pt)
    sm2 = plt.cm.ScalarMappable(
        cmap="RdBu_r",
        norm=plt.Normalize(vmin=-net_vmax, vmax=net_vmax),
    )
    sm2.set_array([])
    cb2 = fig.colorbar(sm2, ax=ax2, shrink=0.8)
    cb2.set_label("Net Advantage over Control, %", fontsize=label_pt)
    cb2.ax.tick_params(labelsize=tick_pt)

    fig.suptitle(
        "Writer Model Competitiveness — CV + Cover Letter",
        fontsize=headline_pt, fontweight="bold",
    )
    plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")


def plot_evaluator_divergence(df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    records = []
    for etype in ["cl_evaluations", "cv_cl_evaluations"]:
        subset = df[df["Eval_Type"] == etype].copy()
        if subset.empty:
            continue
        consensus = (
            subset.groupby(["Job_ID", "CV_Idx", "Writer"])["Score"]
            .mean().reset_index().rename(columns={"Score": "Consensus"})
        )
        subset = subset.merge(consensus, on=["Job_ID", "CV_Idx", "Writer"])
        subset["AbsDev"] = (subset["Score"] - subset["Consensus"]).abs()
        for evaluator, grp in subset.groupby("Evaluator"):
            records.append({"Evaluator": evaluator, "Eval_Type": TITLE_MAP[etype],
                             "Mean_AbsDev": grp["AbsDev"].mean()})

    results_df = pd.DataFrame(records)
    pivot = results_df.pivot(index="Evaluator", columns="Eval_Type", values="Mean_AbsDev")
    pivot["_sort"] = pivot.mean(axis=1)
    # Descending sort → lowest divergence at top of horizontal bar chart
    pivot = pivot.sort_values("_sort", ascending=True).drop(columns="_sort")
    mean_dev = results_df["Mean_AbsDev"].mean()
    pivot.to_csv(os.path.join(save_dir, "evaluator_divergence.csv"))

    fs = plt.rcParams["font.size"] + 3
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="barh", ax=ax, color=["#4393C3", "#D6604D"], edgecolor="white")
    ax.axvline(mean_dev, color="black", linestyle="--", linewidth=1.2,
               label=f"Overall mean ({mean_dev:.2f})")
    # Value labels at tip of each bar
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=4, fontsize=fs - 2)
    ax.set_title("Evaluator Divergence from Group Consensus\n(Lower = Closer to Average)",
                 fontsize=fs + 4)
    ax.set_xlabel("Avg. Deviation from Group Score", fontsize=fs)
    ax.set_ylabel("Evaluator Model", fontsize=fs)
    ax.tick_params(labelsize=fs)
    ax.legend(fontsize=fs)   # no legend title
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "evaluator_divergence.png"))
    plt.close()
    print("  Saved evaluator_divergence.png")


def plot_stacked_score(df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fs = plt.rcParams["font.size"] + 3

    for etype in ["cl_evaluations", "cv_cl_evaluations"]:
        subset = df[df["Eval_Type"] == etype].copy()
        if subset.empty:
            continue
        evaluators = subset["Evaluator"].unique()
        writers    = subset["Writer"].unique()

        records = []
        for evaluator in evaluators:
            eval_data = subset[subset["Evaluator"] == evaluator]
            for writer in writers:
                gaps = []
                for (job, cv), group in eval_data.groupby(["Job_ID", "CV_Idx"]):
                    if writer not in group["Writer"].values:
                        continue
                    gaps.append(group[group["Writer"] == writer]["Score"].mean()
                                - group["Score"].mean())
                if gaps:
                    records.append({"Evaluator": evaluator, "Writer": writer,
                                    "Gap": np.mean(gaps)})
        if not records:
            continue

        gap_df    = pd.DataFrame(records)
        total_abs = gap_df.groupby("Evaluator")["Gap"].apply(lambda x: x.abs().sum())
        evaluator_order = total_abs.sort_values().index.tolist()

        ordered_writers = [w for w in SORTED_WRITERS if w in writers]

        fig, ax = plt.subplots(figsize=(14, 8))
        for x_pos, evaluator in enumerate(evaluator_order):
            eval_gaps = gap_df[gap_df["Evaluator"] == evaluator].set_index("Writer")
            bottom = 0.0
            # Pass 1: positive contributions (fixed writer order)
            for writer in ordered_writers:
                if writer not in eval_gaps.index:
                    continue
                gap = eval_gaps.loc[writer, "Gap"]
                if gap < 0:
                    continue
                ax.bar(x_pos, gap, bottom=bottom, color=WRITER_COLORS[writer],
                       edgecolor="white", linewidth=0.5)
                bottom += gap
            # Pass 2: negative contributions on top (fixed writer order, abs height)
            for writer in ordered_writers:
                if writer not in eval_gaps.index:
                    continue
                gap = eval_gaps.loc[writer, "Gap"]
                if gap >= 0:
                    continue
                ax.bar(x_pos, abs(gap), bottom=bottom, color=WRITER_COLORS[writer],
                       hatch="///", edgecolor="black", linewidth=0.5, alpha=0.85)
                bottom += abs(gap)

        ax.set_xticks(range(len(evaluator_order)))
        ax.set_xticklabels([MODEL_DISPLAY.get(e, e) for e in evaluator_order],
                           rotation=45, ha="right", fontsize=fs)
        for tick, ev in zip(ax.get_xticklabels(), evaluator_order):
            tick.set_color(WRITER_COLORS.get(ev, "black"))
            tick.set_fontweight("bold")

        ax.set_xlabel("Evaluator Model", fontsize=fs + 4)
        ax.set_ylabel("Cumulative |Score Gap| from Evaluator Mean", fontsize=fs + 4)
        ax.set_title(
            f"Cumulative Evaluator Bias (Score-Based) | {TITLE_MAP[etype]}\n"
            "Sorted left → right: least to most biased  |  Solid = over-scored, Hatched = under-scored",
            fontsize=fs + 5, pad=12,
        )
        ax.tick_params(axis="y", labelsize=fs)

        legend_handles = [mpatches.Patch(color=WRITER_COLORS[w], label=MODEL_DISPLAY.get(w, w))
                          for w in ordered_writers]
        legend_handles.append(mpatches.Patch(facecolor="white", hatch="///",
                                             edgecolor="black",
                                             label="Under-scored by this evaluator"))
        leg = ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left",
                        fontsize=fs - 1, title="Writer Model", title_fontsize=fs + 4)

        plt.tight_layout(pad=1.5)
        plt.savefig(os.path.join(save_dir, f"fairness_stacked_{etype}.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved fairness_stacked_{etype}.png")


def plot_stacked_rank(df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fs = plt.rcParams["font.size"] + 3

    for etype in ["cl_evaluations", "cv_cl_evaluations"]:
        subset = df[df["Eval_Type"] == etype].copy()
        if subset.empty:
            continue
        evaluators = subset["Evaluator"].unique()
        writers    = subset["Writer"].unique()
        n_writers  = len(writers)
        midpoint   = (n_writers + 1) / 2

        records = []
        for evaluator in evaluators:
            eval_data = subset[subset["Evaluator"] == evaluator]
            for (job_id, cv_idx), group in eval_data.groupby(["Job_ID", "CV_Idx"]):
                writer_ranks = group.groupby("Writer")["Score"].mean().rank(
                    ascending=False, method="average")
                for writer in writers:
                    if writer not in writer_ranks.index:
                        continue
                    records.append({"Evaluator": evaluator, "Writer": writer,
                                    "Gap": midpoint - writer_ranks[writer]})
        if not records:
            continue

        gap_df    = pd.DataFrame(records).groupby(["Evaluator", "Writer"])["Gap"].mean().reset_index()
        total_abs = gap_df.groupby("Evaluator")["Gap"].apply(lambda x: x.abs().sum())
        evaluator_order = total_abs.sort_values().index.tolist()

        ordered_writers = [w for w in SORTED_WRITERS if w in writers]

        fig, ax = plt.subplots(figsize=(14, 8))
        for x_pos, evaluator in enumerate(evaluator_order):
            eval_gaps = gap_df[gap_df["Evaluator"] == evaluator].set_index("Writer")
            bottom = 0.0
            # Pass 1: positive contributions (fixed writer order)
            for writer in ordered_writers:
                if writer not in eval_gaps.index:
                    continue
                gap = eval_gaps.loc[writer, "Gap"]
                if gap < 0:
                    continue
                ax.bar(x_pos, gap, bottom=bottom, color=WRITER_COLORS[writer],
                       edgecolor="white", linewidth=0.5)
                bottom += gap
            # Pass 2: negative contributions on top (fixed writer order, abs height)
            for writer in ordered_writers:
                if writer not in eval_gaps.index:
                    continue
                gap = eval_gaps.loc[writer, "Gap"]
                if gap >= 0:
                    continue
                ax.bar(x_pos, abs(gap), bottom=bottom, color=WRITER_COLORS[writer],
                       hatch="///", edgecolor="black", linewidth=0.5, alpha=0.85)
                bottom += abs(gap)

        ax.set_xticks(range(len(evaluator_order)))
        ax.set_xticklabels([MODEL_DISPLAY.get(e, e) for e in evaluator_order],
                           rotation=45, ha="right", fontsize=fs)
        for tick, ev in zip(ax.get_xticklabels(), evaluator_order):
            tick.set_color(WRITER_COLORS.get(ev, "black"))
            tick.set_fontweight("bold")

        ax.set_xlabel("Evaluator Model", fontsize=fs + 4)
        ax.set_ylabel("Cumulative |Rank Deviation| from Midpoint", fontsize=fs + 4)
        ax.set_title(
            f"Cumulative Evaluator Bias (Rank-Based) | {TITLE_MAP[etype]}",
            fontsize=fs + 5, pad=12,
        )
        ax.tick_params(axis="y", labelsize=fs)

        legend_handles = [mpatches.Patch(color=WRITER_COLORS[w], label=MODEL_DISPLAY.get(w, w))
                          for w in ordered_writers]
        legend_handles.append(mpatches.Patch(facecolor="white", hatch="///",
                                             edgecolor="black",
                                             label="Under-ranked by this evaluator"))
        leg = ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left",
                        fontsize=fs - 1, title="Writer Model", title_fontsize=fs + 4)
        plt.tight_layout(pad=1.5)
        plt.savefig(os.path.join(save_dir, f"fairness_rank_stacked_{etype}.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved fairness_rank_stacked_{etype}.png")


def plot_fairness_heatmap_combined(df, save_dir):
    """Side-by-side rank-gap heatmap (CL Only | CV+CL), evaluators x writers,
    each with a marginal bar chart of that evaluator's cumulative |bias| and
    a single shared colorbar. Replaces the old stacked-bar version: with 9
    evaluators x 11 writers x 2 conditions, a stacked solid/hatched bar per
    evaluator became too dense to read the per-writer breakdown off of; the
    heatmap shows the full signed g_{e,w} matrix directly, and the bars keep
    the "how biased overall" summary legible via length rather than a second
    color scale (a shared color scale for both the signed values and the
    much-larger totals would wash out the signed cells to near-white)."""
    os.makedirs(save_dir, exist_ok=True)
    # Raw (not print_pt-derived) sizes, copied as-is from the approved mockup
    # (mockup_heatmap_v3.py -> heatmap_polished_v5.png) that this function
    # ports into the real pipeline. Two earlier attempts to re-derive these
    # through print_pt (first DENSE_TICK_PT, then DENSE_ANNOT_PT) each landed
    # on a different number than the mockup's own hand-tuned values and
    # reintroduced the same cell-overflow bug -- hardcoding the mockup's
    # actual numbers is what guarantees an exact match.
    canvas_w = 32
    FS_ANNOT, FS_TICK, FS_LABEL, FS_TITLE, FS_SUPTITLE = 17, 17, 19, 24, 28
    title_pt     = FS_TITLE
    label_pt     = FS_LABEL  # ax_heat's x/ylabel use label_pt + 1 directly, matching the mockup
    tick_pt      = FS_TICK
    annot_pt     = FS_ANNOT
    bar_annot_pt = FS_ANNOT + 1
    headline_pt  = FS_SUPTITLE
    etypes = ["cl_evaluations", "cv_cl_evaluations"]
    CMAP = "vlag"

    # Pre-compute gap data + orderings for both etypes first, so a GLOBAL
    # vmax across both conditions can drive one shared color scale/colorbar.
    panels = {}
    global_vmax = 0.0
    for etype in etypes:
        subset = df[df["Eval_Type"] == etype].copy()
        if subset.empty:
            continue
        writers = subset["Writer"].unique()
        n_writers = len(writers)
        midpoint = (n_writers + 1) / 2

        records = []
        for evaluator in subset["Evaluator"].unique():
            eval_data = subset[subset["Evaluator"] == evaluator]
            for (job_id, cv_idx), group in eval_data.groupby(["Job_ID", "CV_Idx"]):
                writer_ranks = group.groupby("Writer")["Score"].mean().rank(
                    ascending=False, method="average")
                for writer in writers:
                    if writer not in writer_ranks.index:
                        continue
                    records.append({"Evaluator": evaluator, "Writer": writer,
                                    "Gap": midpoint - writer_ranks[writer]})
        if not records:
            continue

        gap_df = pd.DataFrame(records).groupby(["Evaluator", "Writer"])["Gap"].mean().reset_index()
        writers_here = [w for w in SORTED_WRITERS if w in writers]
        evaluators_here = list(gap_df["Evaluator"].unique())
        pivot = gap_df.pivot(index="Evaluator", columns="Writer", values="Gap").reindex(
            index=evaluators_here, columns=writers_here)

        row_tot = pivot.abs().sum(axis=1).sort_values()
        row_order = row_tot.index.tolist()
        # Columns follow the SAME order as the rows for every writer that is
        # also an evaluator, so the self-preference diagonal renders as an
        # actual diagonal line. The two writer-only, locally hosted models
        # (no row counterpart) are appended at the end in roster order, so
        # they stay grouped there as in every other figure.
        col_order = [m for m in row_order if m in writers_here] + \
                    [w for w in writers_here if w not in row_order]
        mat = pivot.loc[row_order, col_order]
        panels[etype] = dict(mat=mat, row_tot=row_tot, row_order=row_order, col_order=col_order)
        global_vmax = max(global_vmax, np.nanmax(np.abs(mat.values)))

    if len(panels) < 2:
        print("  Skipping fairness heatmap — data missing for one condition.")
        return

    for etype, d in panels.items():
        d["mat"].to_csv(os.path.join(save_dir, f"fairness_rank_combined_{etype}.csv"))

    norm = mcolors.Normalize(vmin=-global_vmax, vmax=global_vmax)
    fig = plt.figure(figsize=(canvas_w, canvas_w * 9 / 32))
    outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.4)

    heat_axes, rowbar_axes = [], []
    for panel_idx, etype in enumerate(etypes):
        d = panels[etype]
        mat, row_order, col_order = d["mat"], d["row_order"], d["col_order"]
        row_tot = d["row_tot"]
        n_r, n_c = mat.shape

        inner = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[0, panel_idx], width_ratios=[n_c, 4.2], wspace=0.08,
        )
        ax_heat = fig.add_subplot(inner[0, 0])
        ax_rowbar = fig.add_subplot(inner[0, 1], sharey=ax_heat)
        heat_axes.append(ax_heat)
        rowbar_axes.append(ax_rowbar)

        row_labels = [MODEL_DISPLAY.get(e, e) for e in row_order]
        col_labels = [MODEL_DISPLAY.get(w, w) for w in col_order]

        # No square=True: ax_heat and ax_rowbar share their y-axis, and each
        # fills its own GridSpec cell's full height, so dropping the aspect
        # constraint is what guarantees they render at the same physical height.
        sns.heatmap(mat, cmap=CMAP, norm=norm, annot=True, fmt=".1f", cbar=False,
                    ax=ax_heat, linewidths=0.6, linecolor="white",
                    xticklabels=col_labels, yticklabels=row_labels,
                    annot_kws={"fontsize": annot_pt})

        # Self-preference diagonal: bold black border on any (evaluator,
        # writer) cell where the evaluator is scoring its own writing --
        # same marker convention as Figure 2's self-preference diagonal.
        for i, e in enumerate(row_order):
            if e in col_order:
                j = col_order.index(e)
                ax_heat.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                                 edgecolor="black", linewidth=3.2, zorder=5))

        ax_heat.set_xlabel("Writer model", fontsize=(label_pt + 1) * 1.3)
        ax_heat.set_ylabel("Evaluator model" if panel_idx == 0 else "", fontsize=(label_pt + 1) * 1.3)
        plt.setp(ax_heat.get_xticklabels(), rotation=45, ha="right", fontsize=tick_pt)
        plt.setp(ax_heat.get_yticklabels(), fontsize=tick_pt)
        # Color-code + bold the model-name tick labels, matching the
        # convention used for evaluator ticks elsewhere in this file.
        for tick, w in zip(ax_heat.get_xticklabels(), col_order):
            tick.set_color(WRITER_COLORS.get(w, "black"))
            tick.set_fontweight("bold")
        for tick, e in zip(ax_heat.get_yticklabels(), row_order):
            tick.set_color(WRITER_COLORS.get(e, "black"))
            tick.set_fontweight("bold")
        ax_heat.set_title(TITLE_MAP[etype], fontsize=title_pt, fontweight="bold", pad=10)

        # Row-marginal bars (evaluator Total |bias|), real x-axis with gridlines.
        y_pos = np.arange(n_r) + 0.5
        bar_vals = row_tot.loc[row_order].values
        bar_colors = [WRITER_COLORS.get(e, "#2C3E50") for e in row_order]
        ax_rowbar.barh(y_pos, bar_vals, color=bar_colors, height=0.62, zorder=3,
                       edgecolor="black", linewidth=0.4)
        for y, v in zip(y_pos, bar_vals):
            ax_rowbar.text(v + max(bar_vals) * 0.02, y, f"{v:.1f}", va="center",
                           fontsize=bar_annot_pt, fontweight="bold", color="#222222")
        ax_rowbar.set_ylim(ax_heat.get_ylim())
        ax_rowbar.set_xlim(0, row_tot.max() * 1.18)
        ax_rowbar.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax_rowbar.set_axisbelow(True)
        ax_rowbar.tick_params(axis="y", left=False, labelleft=False)
        ax_rowbar.tick_params(axis="x", labelsize=tick_pt - 1)
        for side in ("top", "right", "left"):
            ax_rowbar.spines[side].set_visible(False)
        ax_rowbar.set_xlabel("Cumulative $\\sum_w|\\bar g_{e,w}|$", fontsize=label_pt * 1.3)
        ax_rowbar.set_title("Total |bias|", fontsize=label_pt, fontweight="bold")

    # One shared colorbar, manually positioned just right of the second
    # panel's bar chart (close, not off at the figure edge).
    fig.canvas.draw()
    last_pos = rowbar_axes[-1].get_position()
    gap, width = 0.012, 0.014
    cax = fig.add_axes([last_pos.x1 + gap, last_pos.y0, width, last_pos.height])
    sm = cm.ScalarMappable(norm=norm, cmap=CMAP)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Rank gap $\\bar g_{e,w}$", fontsize=label_pt * 1.3)
    cb.ax.tick_params(labelsize=tick_pt - 1)

    fig.suptitle("Cumulative Evaluator Bias (Rank-Based), with Total |bias| per Evaluator",
                 fontsize=headline_pt, fontweight="bold", y=1.02)
    plt.savefig(os.path.join(save_dir, "fairness_rank_combined.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fairness_rank_combined.png")


# ==========================
# AGREEMENT CORRELATION
# ==========================

def plot_agreement_corr(tier_df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    # These PNGs are placed inside 0.48\textwidth minipages (side by side),
    # so \includegraphics[width=\textwidth] there really means 0.48 of the
    # page's textwidth — pass that as frac so print_pt scales correctly.
    canvas_w = 10
    frac = 0.48
    title_pt = print_pt(PANEL_TITLE_PT, canvas_w, frac)
    tick_pt  = print_pt(DENSE_TICK_PT, canvas_w, frac)
    annot_pt = print_pt(5.5, canvas_w, frac)
    label_pt = print_pt(AXIS_LABEL_PT, canvas_w, frac)

    def _build_corr(vecs, method):
        df = pd.DataFrame(vecs).dropna()
        return df.corr(method=method)

    def _draw(corr, title, fname, cbar_label, mask_diag=False):
        fig, ax = plt.subplots(figsize=(canvas_w, 8), constrained_layout=True)
        annot = corr.round(2).astype(str)
        diag_mask = pd.DataFrame(False, index=corr.index, columns=corr.columns)
        if mask_diag:
            for label in corr.index:
                if label in corr.columns:
                    diag_mask.loc[label, label] = True
        sns.heatmap(corr, annot=annot, fmt="", ax=ax,
                    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                    linewidths=0.5, linecolor="gray",
                    mask=diag_mask if mask_diag else None,
                    annot_kws={"size": annot_pt})
        if mask_diag:
            labels = list(corr.index)
            for i, label in enumerate(labels):
                if label in corr.columns:
                    ci = list(corr.columns).index(label)
                    ax.add_patch(plt.Rectangle((ci, i), 1, 1, fill=True,
                                               color="lightgrey", lw=0))
                    ax.text(ci + 0.5, i + 0.5, "—", ha="center", va="center",
                            fontsize=annot_pt, color="grey")
        ax.set_title(title, fontsize=title_pt, fontweight="bold", pad=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right",
                           fontsize=tick_pt)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                           fontsize=tick_pt)
        for tick in ax.get_xticklabels():
            tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
        for tick in ax.get_yticklabels():
            tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
        ax.set_xlabel("")
        ax.set_ylabel("")
        cbar = ax.collections[0].colorbar
        cbar.set_label(cbar_label, fontsize=label_pt)
        cbar.ax.tick_params(labelsize=tick_pt)
        plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches="tight")
        plt.close()
        corr.to_csv(os.path.join(save_dir, os.path.splitext(fname)[0] + ".csv"))
        print(f"  Saved {fname}")

    # CV-only: raw score agreement (Pearson) + within-group rank Spearman
    sub_cv = tier_df[tier_df["Eval_Type"] == "cv_only"].copy()
    if not sub_cv.empty:
        vecs_cv = {
            MODEL_DISPLAY.get(ev, ev): (
                sub_cv[sub_cv["Evaluator"] == ev]
                .groupby(["Job_ID", "CV_Idx"])["Score"].mean()
            )
            for ev in UNIQUE_EVALUATORS
        }
        _draw(_build_corr(vecs_cv, "pearson"),
              "Evaluator Agreement — CV Only\n(Pearson r of raw scores)",
              "agreement_corr_cv_only.png", "Pearson r")
        _draw(_build_corr(vecs_cv, "spearman"),
              "Evaluator Agreement — CV Only\n(Spearman ρ of raw scores)",
              "agreement_corr_spearman_cv_only.png", "Spearman ρ")

    # CL and CV+CL: style residual agreement + within-group rank-based Spearman
    for etype in ["cl_evaluations", "cv_cl_evaluations"]:
        sub = tier_df[tier_df["Eval_Type"] == etype].copy()
        if sub.empty:
            continue

        # Residual vectors (for Pearson and residual-Spearman)
        sub["Residual"] = sub["Score"] - sub.groupby(
            ["Evaluator", "Job_ID", "CV_Idx"])["Score"].transform("mean")
        vecs_res = {
            MODEL_DISPLAY.get(ev, ev): (
                sub[sub["Evaluator"] == ev]
                .groupby(["Job_ID", "CV_Idx", "Writer"])["Residual"].mean()
            )
            for ev in UNIQUE_EVALUATORS
        }

        _draw(_build_corr(vecs_res, "pearson"),
              f"Evaluator Style Agreement — {TITLE_MAP[etype]}\n"
              "(Pearson r of within-candidate score residuals)",
              f"agreement_corr_STYLE_{etype}.png", "Pearson r")
        _draw(_build_corr(vecs_res, "spearman"),
              f"Evaluator Style Agreement — {TITLE_MAP[etype]}\n"
              "(Spearman ρ of within-candidate score residuals)",
              f"agreement_corr_spearman_STYLE_{etype}.png", "Spearman ρ")

        # Stylistic Agreement: rank writers within each (Job, CV, Evaluator) group
        sub["Style_Rank"] = sub.groupby(
            ["Job_ID", "CV_Idx", "Evaluator"])["Score"].rank(method="average")
        pivot_style = sub.pivot_table(
            index=["Job_ID", "CV_Idx", "Writer"], columns="Evaluator", values="Style_Rank")
        # pivot_table sorts columns alphabetically by default; reorder to the
        # same canonical evaluator order used by every other heatmap in the
        # paper (UNIQUE_EVALUATORS), instead of leaving it alphabetical.
        pivot_style = pivot_style[[e for e in UNIQUE_EVALUATORS if e in pivot_style.columns]]
        pivot_style.columns = [MODEL_DISPLAY.get(c, c) for c in pivot_style.columns]
        corr_style = pivot_style.corr(method="spearman")
        _draw(corr_style,
              f"Stylistic Agreement — {TITLE_MAP[etype]}\n"
              "(Spearman ρ of within-candidate writer ranks)",
              f"agreement_corr_spearman_STYLISTIC_{etype}.png", "Spearman ρ",
              mask_diag=True)

        # Merit Agreement: Spearman on raw scores per (Job, Writer, CV) — analogous to CV-only
        vecs_merit = {
            MODEL_DISPLAY.get(ev, ev): (
                sub[sub["Evaluator"] == ev]
                .groupby(["Job_ID", "Writer", "CV_Idx"])["Score"].mean()
            )
            for ev in UNIQUE_EVALUATORS
        }
        _draw(_build_corr(vecs_merit, "spearman"),
              f"Merit Agreement — {TITLE_MAP[etype]}\n"
              "(Spearman ρ of raw scores, fixed writer)",
              f"agreement_corr_spearman_MERIT_{etype}.png", "Spearman ρ")


def plot_merit_agreement_combined(tier_df, save_dir):
    """Three-panel heatmap: candidate quality agreement across all conditions."""
    os.makedirs(save_dir, exist_ok=True)

    def _corr(vecs):
        return pd.DataFrame(vecs).dropna().corr(method="spearman")

    # CV-only: single ρ over (Job, CV) pairs
    sub_cv = tier_df[tier_df["Eval_Type"] == "cv_only"]
    vecs_cv = {
        MODEL_DISPLAY.get(ev, ev): (
            sub_cv[sub_cv["Evaluator"] == ev]
            .groupby(["Job_ID", "CV_Idx"])["Score"].mean()
        )
        for ev in UNIQUE_EVALUATORS
    }
    corr_cv = _corr(vecs_cv)

    # CL and CV+CL Merit: per-writer ρ over (Job, CV) pairs, then mean ± std
    merit_corrs = {}
    for etype in ["cl_evaluations", "cv_cl_evaluations"]:
        sub = tier_df[tier_df["Eval_Type"] == etype]
        if sub.empty:
            continue
        writers = sorted(sub["Writer"].unique())
        writer_corr_list = []
        for writer in writers:
            sub_w = sub[sub["Writer"] == writer]
            vecs_w = {
                MODEL_DISPLAY.get(ev, ev): (
                    sub_w[sub_w["Evaluator"] == ev]
                    .groupby(["Job_ID", "CV_Idx"])["Score"].mean()
                )
                for ev in UNIQUE_EVALUATORS
            }
            writer_corr_list.append(_corr(vecs_w))
        stacked = np.stack([c.values for c in writer_corr_list])
        mean_corr = pd.DataFrame(np.mean(stacked, axis=0),
                                 index=writer_corr_list[0].index,
                                 columns=writer_corr_list[0].columns)
        std_corr = pd.DataFrame(np.std(stacked, axis=0),
                                index=writer_corr_list[0].index,
                                columns=writer_corr_list[0].columns)
        merit_corrs[etype] = (mean_corr, std_corr)

    panels = [
        (corr_cv, None, "CV Only"),
        (*merit_corrs["cl_evaluations"],   "Cover Letter Only\n(writer fixed)") if "cl_evaluations" in merit_corrs else None,
        (*merit_corrs["cv_cl_evaluations"], "CV + Cover Letter\n(writer fixed)") if "cv_cl_evaluations" in merit_corrs else None,
    ]
    panels = [p for p in panels if p is not None]
    for corr, _std, title in panels:
        safe_title = re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_").lower()
        corr.to_csv(os.path.join(save_dir, f"merit_agreement_combined_{safe_title}.csv"))
    n = len(panels)
    canvas_w = 7 * n
    title_pt    = print_pt(PANEL_TITLE_PT, canvas_w)
    tick_pt     = print_pt(DENSE_TICK_PT, canvas_w)
    annot_pt    = print_pt(DENSE_ANNOT_PT, canvas_w)
    label_pt    = print_pt(AXIS_LABEL_PT, canvas_w)
    headline_pt = print_pt(HEADLINE_PT, canvas_w)

    fig, axes = plt.subplots(1, n, figsize=(canvas_w, 8), constrained_layout=True)
    fig.get_layout_engine().set(wspace=0.05)
    if n == 1:
        axes = [axes]

    for ax_idx, (ax, panel) in enumerate(zip(axes, panels)):
        corr, std, title = panel
        is_last = ax_idx == n - 1
        annot = corr.round(2).astype(str)
        sns.heatmap(corr, annot=annot, fmt="", ax=ax,
                    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                    linewidths=0.5, linecolor="gray",
                    annot_kws={"size": annot_pt}, cbar=False)
        ax.set_title(title, fontsize=title_pt, fontweight="bold", pad=10)
        ax.set_xlabel("", fontsize=0)

        # X-axis: colored model names on all panels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right",
                           fontsize=tick_pt)
        for tick in ax.get_xticklabels():
            tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))

        # Y-axis: colored model names only on leftmost panel
        if ax_idx == 0:
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                               fontsize=tick_pt)
            for tick in ax.get_yticklabels():
                tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
        else:
            ax.set_yticks([])
            ax.set_ylabel("")

    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.75, pad=0.02)
    cbar.set_label("Spearman ρ", fontsize=label_pt)
    cbar.ax.tick_params(labelsize=tick_pt)

    fig.suptitle("Candidate Quality Agreement across Conditions", fontsize=headline_pt, fontweight="bold")
    plt.savefig(os.path.join(save_dir, "merit_agreement_combined.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved merit_agreement_combined.png")


# ==========================
# STRICTNESS + AGREEMENT COMBINED
# ==========================

def _draw_strictness_left(ax_line, cv_sub, fs, variant):
    ax_line.cla()
    all_series = []
    all_stds = []
    for ev in UNIQUE_EVALUATORS:
        ev_data = cv_sub[cv_sub["Evaluator"] == ev].groupby("CV_Idx")["Score"].mean()
        if ev_data.empty:
            continue
        ev_std = cv_sub[cv_sub["Evaluator"] == ev].groupby("CV_Idx")["Score"].std().fillna(0)
        rolling = ev_data.rolling(window=3, min_periods=1).mean()
        rolling_std = ev_std.rolling(window=3, min_periods=1).mean()
        all_series.append(ev_data)
        all_stds.append(rolling_std)
        color = WRITER_COLORS[ev]
        ax_line.plot(
            rolling.index, rolling.values,
            label=MODEL_DISPLAY[ev],
            color=color,
            linewidth=2, alpha=0.85,
        )
    if all_series:
        all_rolling = pd.concat(all_series, axis=1).rolling(window=3, min_periods=1).mean()
        overall = all_rolling.mean(axis=1)
        ax_line.plot(overall.index, overall.values,
                     color="black", linewidth=3, label="Average", zorder=10)
        if variant == "avg_shade":
            overall_std = pd.concat(all_stds, axis=1).mean(axis=1)
            ax_line.fill_between(
                overall.index,
                overall.values - overall_std.values,
                overall.values + overall_std.values,
                color="black", alpha=0.12, label="Avg ± std",
            )
        elif variant == "envelope":
            ax_line.fill_between(
                overall.index,
                all_rolling.min(axis=1).values,
                all_rolling.max(axis=1).values,
                color="black", alpha=0.10, label="Model range",
            )
    ax_line.set_title("Evaluator Strictness by CV Rank", fontsize=fs + 5, pad=10)
    ax_line.set_xlabel("CV Rank", fontsize=fs + 4)
    ax_line.set_ylabel("Score", fontsize=fs + 4)
    ax_line.set_ylim(0, 10)
    ax_line.tick_params(axis="both", labelsize=fs)
    ax_line.grid(True, alpha=0.3)
    leg = ax_line.legend(fontsize=fs - 1, loc="upper right", framealpha=0.9)
    for text in leg.get_texts():
        text.set_color(DISPLAY_COLORS.get(text.get_text(), "black"))
        text.set_fontweight("bold")


def plot_strictness_agreement_combined(tier_df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fs = plt.rcParams["font.size"] + 5

    cv_sub = tier_df[tier_df["Eval_Type"] == "cv_only"]
    if cv_sub.empty:
        return

    # --- Right: evaluator agreement heatmap (shared across variants) ---
    vecs = {
        MODEL_DISPLAY.get(ev, ev): (
            cv_sub[cv_sub["Evaluator"] == ev]
            .groupby(["Job_ID", "CV_Idx"])["Score"].mean()
        )
        for ev in UNIQUE_EVALUATORS
    }
    corr_pearson  = pd.DataFrame(vecs).dropna().corr(method="pearson")
    corr_spearman = pd.DataFrame(vecs).dropna().corr(method="spearman")

    def _draw_heatmap(ax_heat, corr, cbar_label):
        sns.heatmap(corr, annot=corr.round(2).astype(str), fmt="", ax=ax_heat,
                    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                    linewidths=0.5, linecolor="gray",
                    annot_kws={"size": fs - 1})
        ax_heat.set_title("Evaluator Agreement", fontsize=fs + 5, pad=10)
        ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=45, ha="right",
                                fontsize=fs, fontweight="bold")
        ax_heat.set_yticklabels(ax_heat.get_yticklabels(), rotation=0,
                                fontsize=fs, fontweight="bold")
        for tick in ax_heat.get_xticklabels():
            tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
        for tick in ax_heat.get_yticklabels():
            tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
        ax_heat.set_xlabel("", fontsize=fs + 4)
        ax_heat.set_ylabel("", fontsize=fs + 4)
        cbar = ax_heat.collections[0].colorbar
        cbar.set_label(cbar_label, fontsize=fs + 4)
        cbar.ax.tick_params(labelsize=fs)

    # --- Pearson: line plot combined with heatmap ---
    fig, (ax_line, ax_heat) = plt.subplots(
        1, 2, figsize=(20, 7),
        gridspec_kw={"width_ratios": [1.5, 1]},
        constrained_layout=True,
    )
    _draw_strictness_left(ax_line, cv_sub, fs, "avg_shade")
    _draw_heatmap(ax_heat, corr_pearson, "Pearson r")
    fig.savefig(os.path.join(save_dir, "strictness_agreement_cv_only.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved strictness_agreement_cv_only.png")

    # --- Spearman: line plot combined with heatmap ---
    fig, (ax_line, ax_heat) = plt.subplots(
        1, 2, figsize=(20, 7),
        gridspec_kw={"width_ratios": [1.5, 1]},
        constrained_layout=True,
    )
    _draw_strictness_left(ax_line, cv_sub, fs, "avg_shade")
    _draw_heatmap(ax_heat, corr_spearman, "Spearman ρ")
    fig.savefig(os.path.join(save_dir, "strictness_agreement_spearman_cv_only.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved strictness_agreement_spearman_cv_only.png")

    # --- Line plot standalone ---
    fig, ax_line = plt.subplots(figsize=(12, 7), constrained_layout=True)
    _draw_strictness_left(ax_line, cv_sub, fs, "avg_shade")
    fig.savefig(os.path.join(save_dir, "strictness_lineplot.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved strictness_lineplot.png")

    # --- Box plot variant: High-Fit vs Moderate-Fit CV ranks ---
    all_ranks = sorted(cv_sub["CV_Idx"].unique())
    top25 = set(all_ranks[:25])
    bot25 = set(all_ranks[-25:])
    rows = []
    for ev in UNIQUE_EVALUATORS:
        ev_data = cv_sub[cv_sub["Evaluator"] == ev][["CV_Idx", "Score"]]
        for _, r in ev_data.iterrows():
            if r["CV_Idx"] in top25:
                rows.append({"Model": MODEL_DISPLAY[ev], "Tier": "High-Fit", "Score": r["Score"]})
            elif r["CV_Idx"] in bot25:
                rows.append({"Model": MODEL_DISPLAY[ev], "Tier": "Moderate-Fit", "Score": r["Score"]})
    box_df = pd.DataFrame(rows)

    fig, (ax_box, ax_heat) = plt.subplots(
        1, 2, figsize=(20, 7),
        gridspec_kw={"width_ratios": [1.5, 1]},
        constrained_layout=True,
    )
    model_order = [MODEL_DISPLAY[ev] for ev in UNIQUE_EVALUATORS if MODEL_DISPLAY[ev] in box_df["Model"].unique()]
    model_palette = {MODEL_DISPLAY[ev]: WRITER_COLORS[ev] for ev in UNIQUE_EVALUATORS}
    n_models = len(model_order)
    bar_width = 0.35 / n_models
    group_gap = 0.2
    tier_centers = []
    tier_means = {}
    for tier_idx, tier in enumerate(["High-Fit", "Moderate-Fit"]):
        x_center = tier_idx * (1 + group_gap)
        tier_centers.append(x_center)
        half_span = (n_models - 1) / 2 * (bar_width + 0.02) + bar_width / 2
        tier_means[tier] = (x_center - half_span, x_center + half_span,
                            box_df[box_df["Tier"] == tier]["Score"].mean())
        for model_idx, model in enumerate(model_order):
            subset = box_df[(box_df["Tier"] == tier) & (box_df["Model"] == model)]["Score"]
            if subset.empty:
                continue
            x_pos = x_center + (model_idx - (n_models - 1) / 2) * (bar_width + 0.02)
            ax_box.bar(x_pos, subset.mean(), width=bar_width,
                       color=model_palette[model], alpha=0.85,
                       yerr=subset.std(), error_kw={"capsize": 4, "linewidth": 1.5, "ecolor": "black"},
                       label=model if tier_idx == 0 else "_nolegend_")
    avg_line = None
    for tier, (x0, x1, mean_val) in tier_means.items():
        avg_line, = ax_box.plot([x0, x1], [mean_val, mean_val], color="black",
                                linestyle="dashed", linewidth=1.8, zorder=5)
        ax_box.text(x1 + 0.02, mean_val, f"{mean_val:.2f}", va="center",
                    ha="left", fontsize=fs - 1, fontweight="bold")
    ax_box.set_xticks(tier_centers)
    ax_box.set_xticklabels(["High-Fit", "Moderate-Fit"])
    ax_box.set_title("Evaluator Strictness: High-Fit vs Moderate-Fit CVs", fontsize=fs + 5, pad=10)
    ax_box.set_xlabel("", fontsize=fs + 4)
    ax_box.set_ylabel("Score", fontsize=fs + 4)
    ax_box.set_ylim(0, 10)
    ax_box.tick_params(axis="both", labelsize=fs)
    ax_box.grid(True, alpha=0.3, axis="y")
    for tick in ax_box.get_xticklabels():
        tick.set_fontweight("bold")
        tick.set_fontsize(fs + 2)
    handles, labels = ax_box.get_legend_handles_labels()
    if avg_line is not None:
        handles.append(avg_line)
        labels.append("Average")
    leg = ax_box.legend(handles, labels, fontsize=fs - 1, framealpha=0.9, loc="upper right")
    for text in leg.get_texts():
        text.set_color(DISPLAY_COLORS.get(text.get_text(), "black"))
        text.set_fontweight("bold")
    _draw_heatmap(ax_heat, corr_pearson, "Pearson r")
    fig.savefig(os.path.join(save_dir, "strictness_agreement_boxplot.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved strictness_agreement_boxplot.png")


def plot_strictness_by_condition(tier_df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    canvas_w = 24
    title_pt     = print_pt(PANEL_TITLE_PT, canvas_w)
    tick_pt      = print_pt(TICK_PT_TARGET, canvas_w)
    tier_label_pt = print_pt(5.3, canvas_w)
    annot_pt     = print_pt(ANNOTATION_PT, canvas_w)
    label_pt     = print_pt(AXIS_LABEL_PT, canvas_w)
    legend_pt    = print_pt(LEGEND_PT_TARGET, canvas_w)
    headline_pt  = print_pt(HEADLINE_PT, canvas_w)

    conditions = [
        ("cv_only",           "CV Only"),
        ("cl_evaluations",    "Cover Letter Only"),
        ("cv_cl_evaluations", "CV + Cover Letter"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(canvas_w, 7), sharey=True)
    model_palette = {MODEL_DISPLAY[ev]: WRITER_COLORS[ev] for ev in UNIQUE_EVALUATORS}

    legend_handles, legend_labels = [], []

    for col, (etype, title) in enumerate(conditions):
        ax       = axes[col]
        sub      = tier_df[tier_df["Eval_Type"] == etype]
        is_first = col == 0

        if sub.empty:
            ax.set_visible(False)
            continue

        # --- Bar chart ---
        all_ranks = sorted(sub["CV_Idx"].unique())
        top25 = set(all_ranks[:25])
        bot25 = set(all_ranks[-25:])

        rows = []
        for ev in UNIQUE_EVALUATORS:
            ev_data = sub[sub["Evaluator"] == ev][["CV_Idx", "Score"]]
            for _, r in ev_data.iterrows():
                if r["CV_Idx"] in top25:
                    rows.append({"Model": MODEL_DISPLAY[ev], "Tier": "High-Fit", "Score": r["Score"]})
                elif r["CV_Idx"] in bot25:
                    rows.append({"Model": MODEL_DISPLAY[ev], "Tier": "Moderate-Fit", "Score": r["Score"]})
        box_df = pd.DataFrame(rows)
        box_df.to_csv(os.path.join(save_dir, f"strictness_by_condition_{etype}.csv"), index=False)

        model_order = [MODEL_DISPLAY[ev] for ev in UNIQUE_EVALUATORS if MODEL_DISPLAY[ev] in box_df["Model"].unique()]
        n_models    = len(model_order)
        bar_width   = 0.6 / n_models
        group_gap   = 0.5
        tier_centers = []
        tier_means   = {}

        for tier_idx, tier in enumerate(["High-Fit", "Moderate-Fit"]):
            x_center = tier_idx * (1 + group_gap)
            tier_centers.append(x_center)
            half_span = (n_models - 1) / 2 * (bar_width + 0.08) + bar_width / 2
            tier_data = box_df[box_df["Tier"] == tier]["Score"]
            tier_means[tier] = (x_center - half_span, x_center + half_span,
                                tier_data.mean(), tier_data.std())
            for model_idx, model in enumerate(model_order):
                subset = box_df[(box_df["Tier"] == tier) & (box_df["Model"] == model)]["Score"]
                if subset.empty:
                    continue
                x_pos = x_center + (model_idx - (n_models - 1) / 2) * (bar_width + 0.08)
                m, s = subset.mean(), subset.std()
                bar = ax.bar(x_pos, m, width=bar_width,
                             color=model_palette[model], alpha=0.85,
                             yerr=s, error_kw={"capsize": 4, "linewidth": 1.5, "ecolor": "black"})
                if is_first and tier_idx == 0:
                    legend_handles.append(bar)
                    legend_labels.append(model)

        avg_line = None
        for tier, (x0, x1, mean_val, std_val) in tier_means.items():
            avg_line, = ax.plot([x0, x1], [mean_val, mean_val], color="black",
                                linestyle="dashed", linewidth=2.2, zorder=5)
            ax.text(x1 + 0.02, mean_val, f"{mean_val:.2f}", va="center",
                    ha="left", fontsize=annot_pt)

        if is_first and avg_line is not None:
            legend_handles.append(avg_line)
            legend_labels.append("Average")

        ax.set_xticks(tier_centers)
        ax.set_xticklabels(["High-Fit", "Moderate-Fit"])
        ax.set_title(title, fontsize=title_pt, fontweight="bold", pad=10)
        ax.set_ylabel("Score" if is_first else "", fontsize=label_pt)
        ax.set_ylim(0, 11.5)
        ax.tick_params(axis="both", labelsize=tick_pt)
        if not is_first:
            ax.tick_params(axis="y", labelleft=False)
        ax.grid(True, alpha=0.3, axis="y")
        for tick in ax.get_xticklabels():
            tick.set_fontsize(tier_label_pt)

    fig.suptitle("Scoring Behaviour Across Conditions", fontsize=headline_pt, fontweight="bold")
    leg = fig.legend(legend_handles, legend_labels, fontsize=legend_pt,
                      framealpha=0.9, loc="lower center", ncol=5,
                      bbox_to_anchor=(0.5, -0.14))
    for text in leg.get_texts():
        text.set_color(DISPLAY_COLORS.get(text.get_text(), "black"))

    fig.tight_layout(rect=[0, 0.1, 1, 0.93])
    fname = "strictness_by_condition.png"
    fig.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")


def plot_intra_run_variance(tier_df, save_dir):
    """Score std across the 4 independent runs — measures evaluator reproducibility."""
    os.makedirs(save_dir, exist_ok=True)
    fs = plt.rcParams["font.size"] + 5

    conditions = [
        ("cv_only",           "CV Only"),
        ("cl_evaluations",    "Cover Letter Only"),
        ("cv_cl_evaluations", "CV + Cover Letter"),
    ]
    cond_colors = ["#5C85C5", "#E07B4A", "#4CAF50"]

    def _run_std(sub, etype):
        grp = ["Job_ID", "CV_Idx", "Evaluator"] if etype == "cv_only" \
              else ["Job_ID", "CV_Idx", "Evaluator", "Writer"]
        return sub.groupby(grp)["Score"].std()

    # --- Figure 1: grouped bar chart per evaluator × condition ---
    n_ev   = len(UNIQUE_EVALUATORS)
    n_cond = len(conditions)
    bar_w  = 0.22
    x      = np.arange(n_ev)

    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    for ci, ((etype, label), color) in enumerate(zip(conditions, cond_colors)):
        sub  = tier_df[tier_df["Eval_Type"] == etype]
        stds = _run_std(sub, etype)
        means = [stds.xs(ev, level="Evaluator").mean() for ev in UNIQUE_EVALUATORS]
        offset = (ci - (n_cond - 1) / 2) * (bar_w + 0.03)
        bars = ax.bar(x + offset, means, width=bar_w, label=label,
                      color=color, alpha=0.85, edgecolor="white")
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=fs - 3)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY[ev] for ev in UNIQUE_EVALUATORS],
                       rotation=30, ha="right", fontsize=fs, fontweight="bold")
    for tick, ev in zip(ax.get_xticklabels(), UNIQUE_EVALUATORS):
        tick.set_color(DISPLAY_COLORS.get(MODEL_DISPLAY[ev], "black"))
    ax.set_ylabel("Mean score std across 4 runs", fontsize=fs + 2)
    ax.set_title("Scoring Reproducibility by Evaluator and Condition",
                 fontsize=fs + 5, pad=10)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.tick_params(axis="y", labelsize=fs)
    ax.grid(True, alpha=0.3, axis="y")
    leg = ax.legend(fontsize=fs, framealpha=0.9)
    for text in leg.get_texts():
        text.set_fontweight("bold")

    fig.savefig(os.path.join(save_dir, "intra_run_variance.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved intra_run_variance.png")

    # --- Figure 2: heatmaps (evaluator × writer) for CL-only and CV+CL ---
    cl_conditions = [("cl_evaluations", "Cover Letter Only"),
                     ("cv_cl_evaluations", "CV + Cover Letter")]

    fig, axes = plt.subplots(1, 2, figsize=(24, 7), constrained_layout=True)
    all_vals = []
    matrices = {}
    for etype, _ in cl_conditions:
        sub  = tier_df[tier_df["Eval_Type"] == etype]
        stds = _run_std(sub, etype)
        mat  = pd.DataFrame(index=[MODEL_DISPLAY[ev] for ev in UNIQUE_EVALUATORS],
                            columns=[MODEL_DISPLAY.get(w, w) for w in RAW_WRITERS],
                            dtype=float)
        for ev in UNIQUE_EVALUATORS:
            for w in RAW_WRITERS:
                try:
                    mat.loc[MODEL_DISPLAY[ev], MODEL_DISPLAY.get(w, w)] = \
                        stds.xs((ev, w), level=("Evaluator", "Writer")).mean()
                except KeyError:
                    mat.loc[MODEL_DISPLAY[ev], MODEL_DISPLAY.get(w, w)] = np.nan
        matrices[etype] = mat
        all_vals.extend(mat.values.flatten())

    vmin = np.nanmin(all_vals)
    vmax = np.nanmax(all_vals)
    n_writers = len(RAW_WRITERS)
    n_evals   = len(UNIQUE_EVALUATORS)
    ylord_cmap = plt.cm.YlOrRd

    for ax, (etype, title) in zip(axes, cl_conditions):
        mat = matrices[etype]

        # append avg column to the matrix (masked from main colormap)
        mat_full = mat.copy()
        mat_full["Avg."] = mat.mean(axis=1)

        # mask the avg column so the main heatmap leaves it white
        avg_mask = np.zeros(mat_full.shape, dtype=bool)
        avg_mask[:, -1] = True

        annot_full = mat_full.round(3).astype(str).replace("nan", "")

        sns.heatmap(mat_full, annot=annot_full, fmt="", ax=ax,
                    cmap="YlOrRd", vmin=vmin, vmax=vmax,
                    linewidths=0.5, linecolor="gray",
                    annot_kws={"size": fs - 1}, cbar=(ax is axes[-1]),
                    mask=avg_mask)

        # draw avg column cells with same YlOrRd colormap
        avg_vals_col = mat_full["Avg."].values
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        for i, val in enumerate(avg_vals_col):
            color = ylord_cmap(norm(val))
            ax.add_patch(mpatches.Rectangle(
                (n_writers, i), 1, 1,
                facecolor=color, edgecolor="gray", linewidth=0.5, zorder=2))
            # pick text color by luminance, matching seaborn's logic
            rgb = np.array(color[:3])
            rgb_lin = np.where(rgb <= 0.03928, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
            lum = rgb_lin @ [0.2126, 0.7152, 0.0722]
            txt_color = ".15" if lum > 0.408 else "w"
            ax.text(n_writers + 0.5, i + 0.5, f"{val:.3f}",
                    ha="center", va="center", fontsize=fs - 1,
                    color=txt_color, zorder=3)

        # thick separator line between writer columns and avg column
        ax.axvline(n_writers, color="black", lw=2.5, zorder=4)

        ax.set_title(title, fontsize=fs + 5, pad=10)
        ax.set_xlabel("Writer Model", fontsize=fs + 2)
        ax.set_ylabel("Evaluator Model" if ax is axes[0] else "", fontsize=fs + 2)

        # rebuild xtick labels including "Avg"
        xtick_labels = [MODEL_DISPLAY.get(w, w) for w in RAW_WRITERS] + ["Avg."]
        ax.set_xticks(np.arange(len(xtick_labels)) + 0.5)
        ax.set_xticklabels(xtick_labels, rotation=45, ha="right",
                           fontsize=fs, fontweight="bold")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                           fontsize=fs, fontweight="bold")
        for tick in ax.get_xticklabels():
            if tick.get_text() == "Avg.":
                tick.set_color("black")
            else:
                tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))
        for tick in ax.get_yticklabels():
            tick.set_color(DISPLAY_COLORS.get(tick.get_text(), "black"))

    cbar = axes[-1].collections[0].colorbar
    cbar.set_label("Mean std across 4 runs", fontsize=fs + 2)
    cbar.ax.tick_params(labelsize=fs)

    fig.suptitle("Scoring Reproducibility by Evaluator and Writer Model",
                 fontsize=fs + 5)
    fig.savefig(os.path.join(save_dir, "intra_run_variance_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved intra_run_variance_heatmap.png")


# ==========================
# MAIN
# ==========================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(BASE_DIR) and not os.path.exists(CACHE_PATH):
        print(f"ERROR: {BASE_DIR} does not exist.")
        return

    # Master dataframe (from parquet cache)
    print("\n=== Loading master dataframe ===")
    master_df = build_master_df(BASE_DIR)
    tier_df = pd.DataFrame()
    if not master_df.empty:
        start_cv, end_cv = TIER_RANGE
        tier_df = master_df[
            (master_df["CV_Idx"] >= start_cv) & (master_df["CV_Idx"] <= end_cv)
        ]

    # Net advantage (arena simulation)
    print("\n=== Loading data for net_advantage plot ===")
    comp_df = load_competitive_data(BASE_DIR)
    delta = p_mat = rounded_raw = writers = None
    if not comp_df.empty:
        writers = sorted(
            comp_df[comp_df["Type"] == "cv_cl_evaluations"]["Writer"].dropna().unique()
        )
        global_dist = {(b, t): [] for b in writers for t in writers}
        eval_dist = {(b, t): {} for b in writers for t in writers}
        for evaluator in UNIQUE_EVALUATORS:
            print(f"  Simulating arena for {evaluator}...")
            for b in writers:
                for t in writers:
                    vals = calculate_leapfrog(comp_df, evaluator, b, t)
                    global_dist[(b, t)].extend(vals)
                    if vals:
                        eval_dist[(b, t)][evaluator] = np.mean(vals)

        global_raw = np.zeros((len(writers), len(writers)))
        for i, b in enumerate(writers):
            for j, t in enumerate(writers):
                pcts = global_dist[(b, t)]
                global_raw[i, j] = np.mean(pcts) if pcts else np.nan

        rounded_raw = np.round(global_raw, 1)
        delta = np.zeros((len(writers), len(writers)))
        p_mat = np.full((len(writers), len(writers)), np.nan)
        std_mat = np.full((len(writers), len(writers)), np.nan)
        for i, b in enumerate(writers):
            ctrl_by_eval = eval_dist[(b, b)]
            ctrl_val  = rounded_raw[i, i]
            for j, t in enumerate(writers):
                tgt_by_eval = eval_dist[(b, t)]
                if not tgt_by_eval or not ctrl_by_eval:
                    delta[i, j] = np.nan
                    continue
                delta[i, j] = rounded_raw[i, j] - ctrl_val
                if i == j:
                    p_mat[i, j] = 1.0
                    continue
                # Significance is tested on the per-evaluator means (one
                # value per evaluator, already averaged over its runs and
                # jobs), not on the pooled per-job values used for the point
                # estimate above: evaluators are not exchangeable draws, so
                # pooling them into one sample inflates the effective N and
                # understates the p-value.
                common_evs = [e for e in UNIQUE_EVALUATORS
                              if e in tgt_by_eval and e in ctrl_by_eval]
                if len(common_evs) > 1:
                    tgt_arr  = np.array([tgt_by_eval[e] for e in common_evs])
                    ctrl_arr = np.array([ctrl_by_eval[e] for e in common_evs])
                    # Per-evaluator NET values (each evaluator's own target
                    # minus its own control), not the raw tgt_arr itself --
                    # std_mat is meant to show evaluator-to-evaluator spread
                    # in the net effect, same quantity delta[i, j] averages.
                    std_mat[i, j] = np.std(tgt_arr - ctrl_arr)
                    if np.array_equal(tgt_arr, ctrl_arr):
                        p_mat[i, j] = 1.0
                    else:
                        _, p = stats.ttest_rel(tgt_arr, ctrl_arr)
                        p_mat[i, j] = p

    # Generate all plots
    print(f"\n=== Generating plots → {OUT_DIR} ===")

    if not tier_df.empty:
        plot_heatmap_gap(tier_df, OUT_DIR)
        plot_heatmap_gap_combined(tier_df, OUT_DIR)
        plot_win_matrix(tier_df, OUT_DIR)
        plot_evaluator_divergence(tier_df, OUT_DIR)
        plot_stacked_score(tier_df, OUT_DIR)
        plot_stacked_rank(tier_df, OUT_DIR)
        plot_fairness_heatmap_combined(tier_df, OUT_DIR)
        plot_agreement_corr(tier_df, OUT_DIR)
        plot_merit_agreement_combined(tier_df, OUT_DIR)
        plot_strictness_agreement_combined(tier_df, OUT_DIR)
        plot_strictness_by_condition(tier_df, OUT_DIR)
        plot_intra_run_variance(tier_df, OUT_DIR)
    else:
        print("  Skipping master-df plots (no data).")

    if comp_df is not None and not comp_df.empty:
        plot_net_advantage(delta, p_mat, rounded_raw, writers, OUT_DIR, std_matrix=std_mat)
        if not tier_df.empty:
            plot_win_net_combined(tier_df, delta, p_mat, rounded_raw, writers, OUT_DIR, std_matrix=std_mat)

        # SI companion: the two most extreme INDIVIDUAL evaluators (pooled
        # matrix above averages across all 9), by peak cell (GPT-4o mini)
        # and by summed |net| across their whole matrix (DeepSeek V3.2).
        plot_net_advantage_extreme_evaluators(
            comp_df, writers, ("gpt-4o-mini", "deepseek-v3.2"), OUT_DIR)

        # Cosine-similarity split variant
        print("\n=== Simulating arena (cosine-similarity split) ===")
        global_dist_cos = {(b, t): [] for b in writers for t in writers}
        eval_dist_cos = {(b, t): {} for b in writers for t in writers}
        for evaluator in UNIQUE_EVALUATORS:
            print(f"  Simulating arena (cosine) for {evaluator}...")
            for b in writers:
                for t in writers:
                    vals = calculate_leapfrog_cosine(comp_df, evaluator, b, t)
                    global_dist_cos[(b, t)].extend(vals)
                    if vals:
                        eval_dist_cos[(b, t)][evaluator] = np.mean(vals)
        global_raw_cos = np.zeros((len(writers), len(writers)))
        for i, b in enumerate(writers):
            for j, t in enumerate(writers):
                pcts = global_dist_cos[(b, t)]
                global_raw_cos[i, j] = np.mean(pcts) if pcts else np.nan
        rounded_raw_cos = np.round(global_raw_cos, 1)
        delta_cos = np.zeros((len(writers), len(writers)))
        p_mat_cos = np.full((len(writers), len(writers)), np.nan)
        for i, b in enumerate(writers):
            ctrl_by_eval = eval_dist_cos[(b, b)]
            ctrl_val = rounded_raw_cos[i, i]
            for j, t in enumerate(writers):
                tgt_by_eval = eval_dist_cos[(b, t)]
                if not tgt_by_eval or not ctrl_by_eval:
                    delta_cos[i, j] = np.nan
                    continue
                delta_cos[i, j] = rounded_raw_cos[i, j] - ctrl_val
                if i == j:
                    p_mat_cos[i, j] = 1.0
                    continue
                # Same evaluator-level test as above: see comment there.
                common_evs = [e for e in UNIQUE_EVALUATORS
                              if e in tgt_by_eval and e in ctrl_by_eval]
                if len(common_evs) > 1:
                    tgt_arr  = np.array([tgt_by_eval[e] for e in common_evs])
                    ctrl_arr = np.array([ctrl_by_eval[e] for e in common_evs])
                    if np.array_equal(tgt_arr, ctrl_arr):
                        p_mat_cos[i, j] = 1.0
                    else:
                        _, p = stats.ttest_rel(tgt_arr, ctrl_arr)
                        p_mat_cos[i, j] = p
        if not tier_df.empty:
            plot_win_net_combined(
                tier_df, delta_cos, p_mat_cos, rounded_raw_cos, writers, OUT_DIR,
                group_labels=("High-Similarity", "Moderate-Similarity"),
                fname="win_net_combined_cosine.png",
            )
    else:
        print("  Skipping net_advantage / win_net_combined (no competitive data).")

    print(f"\nDone. Plots → {OUT_DIR}/")


if __name__ == "__main__":
    main()
