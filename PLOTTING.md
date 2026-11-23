# Regenerating the paper's figures

**Canonical entry point: `generate_all_paper_figures.py`.** Run this, not the
individual scripts, whenever you need to regenerate figures for
`../paper/paper.tex`.

```
conda run -n llm-hiring python generate_all_paper_figures.py            # regenerate + report
conda run -n llm-hiring python generate_all_paper_figures.py --sync     # also copy changed/new PNGs into ../paper/figures/
conda run -n llm-hiring python generate_all_paper_figures.py --skip-generation --sync   # just re-diff/sync existing output_plots/, no rerun
```

It runs the scripts below in dependency order (later ones import names from,
or read CSVs written by, earlier ones), then diffs every PNG it produced
against what's embedded in `../paper/figures/` and reports
UNCHANGED / DIFFERS / new / orphaned.

Every regenerated `<name>.png` also gets a `<name>.csv` (or `<name>_*.csv`)
next to it with the underlying numbers. Use those instead of re-deriving
numbers from a script by hand.

Three figures place their labels with `adjustText`, which moves them slightly
from run to run (`wve_tradeoffs`, `wve_cosine_cvcl`, `self_pref_summary_cvcl`).
They may be reported as DIFFERS even when the numbers behind them are
identical.

## Pipeline / data map

| Order | Script | Reads | Writes to |
|---|---|---|---|
| 1 | `aggregate_plots.py` | `output_eval/master_df.parquet`, `output_eval/competitive_df.parquet` | `output_plots/paper_plots/` |
| 2 | `make_fig_agreement.py` | CSVs from 1 | same dir |
| 3 | `plot_feature_regression.py` | `output_eval/cl_features.parquet`, `output_eval/master_df.parquet` | `output_plots/cl_features/` |
| 4 | `plot_nested_model_decomposition.py` | same, + imports from 3 | same dir |
| 5 | `plot_model_explainability.py` | same, + `output_eval/cv_job_similarity.parquet`, imports from 4 | same dir |
| 6 | `make_fig_writer_vs_evaluator.py` | CSVs from 5 | same dir |
| 7 | `make_fig_selfpref_summary.py` | CSVs from 1 and 3 | `output_plots/paper_plots/` |
| 8 | `plot_cl_features.py` | `output_eval/cl_features.parquet` | `output_plots/cl_features/` |
| 9 | `plot_cl_variance.py` | `output_eval/cl_features.parquet` | same dir |
| 10 | `make_fig_tier_effect.py` | CSV from 9 | same dir |

`paper_style.py` holds the shared style (fonts, model colours, tile matrices)
used by all of them.

## Which script draws which figure

| Paper | File | Script |
|---|---|---|
| Fig. 1 | `pipeline_diagram.pdf` | `make_fig1.py` (writes TikZ; compile with xelatex, see the script) |
| Fig. 2 | `gap_dotplot_combined` | `aggregate_plots.py` |
| Fig. 3 | `fairness_rank_combined` | `aggregate_plots.py` |
| Fig. 4 | `win_net_combined` | `aggregate_plots.py` |
| Fig. 5 | `common_preference_summary_ranked` | `plot_feature_regression.py` |
| Fig. 6 | `self_pref_summary` | `make_fig_selfpref_summary.py` |
| Fig. A1 | `cl_features_comprehensive` | `plot_cl_features.py` |
| Fig. A2 | `cl_embedding_homogeneity` | `plot_cl_variance.py` |
| Fig. A3 | `tier_effect_dots` | `make_fig_tier_effect.py` |
| Fig. A4 | `strictness_by_condition` | `aggregate_plots.py` |
| Fig. A5 | `agreement_quality` | `make_fig_agreement.py` |
| Fig. B6 | `agreement_writer` | `make_fig_agreement.py` |
| Fig. B7 | `heatmap_gap_combined` | `aggregate_plots.py` |
| Fig. B8 | `gap_by_tier` | `aggregate_plots.py` |
| Fig. C9 | `net_advantage_extreme_evaluators` | `aggregate_plots.py` |
| Fig. D10 | `wve_tradeoffs` | `make_fig_writer_vs_evaluator.py` |
| Fig. D11 | `common_preference_summary_ranked_cvcl` | `plot_feature_regression.py` |
| Fig. D12 | `wve_cosine_cvcl` | `make_fig_writer_vs_evaluator.py` |
| Fig. D13 | `self_pref_summary_cvcl` | `make_fig_selfpref_summary.py` |
| Table A1 | intra-run SD by evaluator and condition | `aggregate_plots.py` (`intra_run_variance*.csv`) |
| Table D3 | nested variance decomposition | `plot_nested_model_decomposition.py` (`nested_model_decomposition.csv`) |

Figure numbers follow the current manuscript; if figures are added or
removed, the file names stay the same.

`../paper/figures/` is not a build output: it's the last synced snapshot
embedded in the paper. It can drift from what these scripts produce; that's
what `generate_all_paper_figures.py`'s comparison report catches.
