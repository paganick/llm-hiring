"""
generate_all_paper_figures.py — single entry point that regenerates every
figure referenced in paper/paper.tex, then reports how each regenerated file
compares to what's currently embedded in paper/figures/.

Why this exists: the figures are produced by six different scripts that
accumulated over the project's life (aggregate_plots.py plus five
specialised plot_*.py scripts added later for the feature/style analyses).
Nothing previously tied them together, so paper/figures/ could silently go
stale relative to the scripts/data that actually produce it (this happened
to win_net_combined.png — a June figure survived in the paper for months
after the writer roster was extended from 8 to 11 models). This script is
the fix: run it, read the report, and any figure that's stale shows up as
DIFFERS or MISSING instead of being discovered by hand months later.

Covers all figures except pipeline_diagram.pdf, which is not code-generated
(hand-drawn tikz in pipeline_diagram.tex, compiled and cropped manually —
see the note at the top of paper.tex).

What belongs in paper/figures/ is defined ENTIRELY by paper/paper.tex's own
\includegraphics calls, not by whatever already happens to be sitting in
that directory -- see report_and_sync()'s docstring for why that distinction
matters.

Usage:
    python generate_all_paper_figures.py              # regenerate + report only
    python generate_all_paper_figures.py --sync        # also copy changed/new
                                                         # NEEDED files into paper/figures/
    python generate_all_paper_figures.py --sync --prune  # --sync, and also delete
                                                         # any paper/figures/ file the
                                                         # paper doesn't cite

Run with the same interpreter used for the rest of this project's analysis
(needs pandas, numpy, matplotlib, seaborn, scipy, scikit-learn — e.g. the
`llm-bias-analysis` conda env: `conda run -n llm-bias-analysis python
generate_all_paper_figures.py`).
"""
import filecmp
import os
import re
import shutil
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# The manuscript lives next to the repo (../paper), outside git. Override
# with LLMHIRING_PAPER_DIR if it sits elsewhere.
PAPER_DIR = os.environ.get("LLMHIRING_PAPER_DIR",
                           os.path.join(PROJECT_ROOT, os.pardir, "paper"))
TEX_FIGURES_DIR = os.path.join(PAPER_DIR, "figures")
PAPER_TEX = os.path.join(PAPER_DIR, "paper.tex")


def find_needed_figures():
    """Figure basenames actually \\includegraphics'd by paper/paper.tex --
    the authoritative "what belongs in paper/figures/" list. Excludes
    pipeline_diagram, which isn't code-generated (see module docstring)."""
    with open(PAPER_TEX) as f:
        content = f.read()
    names = set(re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', content))
    names.discard("pipeline_diagram")
    return names

# Each script writes into the OUT_DIR it hardcodes internally (see the
# script itself for exactly which — they're not all the same directory).
# Order matters: later scripts import names from earlier ones.
SCRIPTS = [
    # --no-gemini2: gemini-2.0-flash was dropped from UNIQUE_EVALUATORS/RAW_WRITERS
    # (aggregate_plots.py) but the DEFAULT cache (master_df.parquet,
    # competitive_df.parquet) still contains its rows, and main() builds its
    # writers/evaluators lists from those raw caches' .unique() values rather
    # than filtering through the constants — so the plain default run lets
    # gemini-2.0-flash silently leak back in as an extra, undocumented 12th
    # writer. --no-gemini2 points at the caches that physically exclude it
    # (master_df_no_gemini2.parquet etc.), matching every other script here
    # (they all hardcode the *_no_gemini2 paths with no flag at all) and
    # matching the paper's stated 11-writer/9-evaluator roster.
    ("aggregate_plots.py", ["--no-gemini2"]),
    ("plot_feature_regression.py", []),
    ("plot_nested_model_decomposition.py", []),
    ("plot_model_explainability.py", []),
    ("plot_cl_features.py", []),
    ("plot_cl_variance.py", []),
]

# Where each script's outputs land, relative to PROJECT_ROOT.
OUTPUT_DIRS = [
    "output_plots/paper_plots_v3_no_gemini2",
    "output_plots/cl_features_no_gemini2",
]


def run_scripts():
    for script, extra_args in SCRIPTS:
        print(f"\n{'=' * 70}\nRunning {script} {' '.join(extra_args)}\n{'=' * 70}")
        result = subprocess.run(
            [sys.executable, script, *extra_args],
            cwd=PROJECT_ROOT,
        )
        if result.returncode != 0:
            print(f"\n!! {script} exited with code {result.returncode} — stopping.")
            sys.exit(result.returncode)


def find_generated_pngs():
    """Map figure basename (no extension) -> full path of the freshly generated file.

    Walks each OUT_DIR recursively — some figures (e.g. the
    preference_by_category/ dot-plots in plot_feature_regression.py) are
    written into subdirectories, not directly into OUT_DIR.
    """
    found = {}
    for rel_dir in OUTPUT_DIRS:
        abs_dir = os.path.join(PROJECT_ROOT, rel_dir)
        if not os.path.isdir(abs_dir):
            continue
        for root, _dirs, files in os.walk(abs_dir):
            for fname in files:
                if fname.endswith(".png"):
                    found[os.path.splitext(fname)[0]] = os.path.join(root, fname)
    return found


def report_and_sync(generated, do_sync, do_prune):
    """paper/paper.tex's \\includegraphics calls are the authority for what
    belongs in paper/figures/ -- NOT whatever already happens to be sitting
    there. (paper/figures/ used to accumulate ~80 orphaned figure variants
    this way: --sync copied anything "new" relative to the existing
    directory, so an unused figure, once added once, stayed synced forever.
    Comparing against the needed set instead of the existing directory is
    what prevents that from recurring.)"""
    needed = find_needed_figures()
    tex_figs = {
        os.path.splitext(f)[0]: os.path.join(TEX_FIGURES_DIR, f)
        for f in os.listdir(TEX_FIGURES_DIR)
        if f.endswith(".png")
    }

    print(f"\n{'=' * 70}\nCOMPARISON: generated output vs paper/figures/ "
          f"(scoped to the {len(needed)} figures paper/paper.tex actually cites)\n{'=' * 70}")

    same, differs, new_in_output, missing_entirely = [], [], [], []

    for name in sorted(needed):
        gen_path = generated.get(name)
        if gen_path is None:
            missing_entirely.append(name)
            continue
        embedded_path = tex_figs.get(name)
        if embedded_path is None:
            new_in_output.append(name)
        elif filecmp.cmp(gen_path, embedded_path, shallow=False):
            same.append(name)
        else:
            differs.append(name)

    unused_in_output = sorted(set(generated) - needed)
    unused_in_tex = sorted(set(tex_figs) - needed)

    print(f"\nUNCHANGED ({len(same)}):")
    for n in same:
        print(f"  = {n}")

    print(f"\nDIFFERS FROM EMBEDDED — paper/figures/ is stale ({len(differs)}):")
    for n in differs:
        print(f"  ! {n}")

    print(f"\nGENERATED BUT NOT YET IN paper/figures/ ({len(new_in_output)}):")
    for n in new_in_output:
        print(f"  + {n}")

    print(f"\nMISSING — cited by paper.tex but no script produces them ({len(missing_entirely)}):")
    for n in missing_entirely:
        print(f"  !! {n}")

    print(f"\nGENERATED BUT NOT CITED BY THE PAPER, never synced ({len(unused_in_output)}):")
    for n in unused_in_output:
        print(f"  . {n}")

    if unused_in_tex:
        print(f"\nIN paper/figures/ BUT NOT CITED BY THE PAPER ({len(unused_in_tex)})"
              f"{' -- removed' if do_prune else ' -- pass --prune to remove'}:")
        for n in unused_in_tex:
            print(f"  ? {n}.png")
            if do_prune:
                os.remove(tex_figs[n])

    if do_sync:
        to_copy = differs + new_in_output
        print(f"\nSyncing {len(to_copy)} file(s) into paper/figures/ ...")
        for n in to_copy:
            dest = os.path.join(TEX_FIGURES_DIR, n + ".png")
            shutil.copy2(generated[n], dest)
            print(f"  -> {n}.png")
    elif differs or new_in_output:
        print(
            "\nRe-run with --sync to copy the DIFFERS/new files above into "
            "paper/figures/. Nothing was copied this run."
        )

    if missing_entirely:
        print(f"\n!! {len(missing_entirely)} figure(s) the paper cites were not "
              f"produced by any script -- fix before submitting.")
        sys.exit(1)


def main():
    do_sync = "--sync" in sys.argv
    do_prune = "--prune" in sys.argv
    if "--skip-generation" not in sys.argv:
        run_scripts()
    generated = find_generated_pngs()
    report_and_sync(generated, do_sync, do_prune)


if __name__ == "__main__":
    main()
