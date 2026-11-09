# Regenerating the paper's figures

**Canonical entry point: `generate_all_paper_figures.py`.** Run this, not any
individual `plot_*.py` / `aggregate_plots.py` script directly, whenever you
need to regenerate figures for `../paper/paper.tex`.

```
conda run -n llm-bias-analysis python generate_all_paper_figures.py            # regenerate + report
conda run -n llm-bias-analysis python generate_all_paper_figures.py --sync     # also copy changed/new PNGs into ../paper/figures/
conda run -n llm-bias-analysis python generate_all_paper_figures.py --skip-generation --sync   # just re-diff/sync existing output_plots/, no rerun
```

It runs the six scripts that together produce every figure in the paper
(`aggregate_plots.py`, `plot_feature_regression.py`,
`plot_nested_model_decomposition.py`, `plot_model_explainability.py`,
`plot_cl_features.py`, `plot_cl_variance.py`, in that dependency order —
later ones import names from earlier ones), then diffs every PNG it
produced against what's currently embedded in `../paper/figures/` and reports
UNCHANGED / DIFFERS / new / orphaned.

Every regenerated `<name>.png` also gets a `<name>.csv` next to it with the
underlying numeric table(s) — see `output_plots/*/*.csv`. Use those instead
of re-deriving numbers from a script by hand.

## Pipeline / data map

| Script | Reads | Writes |
|---|---|---|
| `aggregate_plots.py` | `output_eval/master_df.parquet`, `output_eval/competitive_df.parquet` | `output_plots/paper_plots/` |
| `plot_feature_regression.py` | `output_eval/cl_features.parquet`, `output_eval/master_df.parquet` | `output_plots/cl_features/` |
| `plot_nested_model_decomposition.py` | same, + imports from `plot_feature_regression.py` | same dir |
| `plot_model_explainability.py` | same, + `output_eval/cv_job_similarity.parquet`, imports from `plot_nested_model_decomposition.py` | same dir |
| `plot_cl_features.py` | `output_eval/cl_features.parquet` | same dir |
| `plot_cl_variance.py` | `output_eval/cl_features.parquet` | same dir |

`../paper/figures/` is not a build output — it's the last manually-synced
snapshot embedded in the paper. It can silently drift from what these
scripts currently produce; that's what `generate_all_paper_figures.py`'s
comparison report catches.
