# LLM writers × LLM evaluators: generative preference bias in AI-mediated hiring

Code for *"AI evaluators favor job applications written by certain AI models,
independent of candidate quality"* (Pagan, Ebling, Ionescu, Hannák).

Eleven writer LLMs (nine API-hosted, two locally hosted) write cover letters
for 50 candidates × 10 job postings. Nine API-hosted evaluator LLMs score every
candidate under three conditions (CV only, cover letter only, CV + cover
letter), four runs each: 414,000 evaluations in total.

## Quick start: reproduce the paper's figures

```bash
conda env create -f environment.yml
conda activate llm-hiring
python generate_all_paper_figures.py          # regenerate all figures + CSVs
```

This needs only the four parquet caches in `output_eval/` (tracked in this
repo). Figures and a same-name `.csv` of the numbers behind each one are
written to `output_plots/`. See `PLOTTING.md` for the pipeline order and the
data-file map. If the manuscript folder sits next to the repo
(`../paper`, or set `LLMHIRING_PAPER_DIR`), the script also diffs every figure
against what the paper embeds; `--sync` copies changed figures in.

## Layout

| Path | What |
|---|---|
| `get_data.py` | Selects the 10 jobs and 50 CVs per job from the Kaggle datasets (top 25 by CV–job cosine similarity, plus 25 starting at the first CV with similarity ≤ 0.4) |
| `cover_letter_generation.py` | Phase 1: writer models generate cover letters |
| `cover_letter_evaluation.py` | Phase 2, stage 1 (direct provider APIs): GPT-4o mini, GPT-5 mini, Claude Haiku 4.5 and Gemini 3 Flash score CV only and the original seven writers' letters. Kept as a record |
| `cover_letter_evaluation_gateway.py` | Phase 2, stage 2 (Vercel AI Gateway): every other score. Same prompts; Gemini 3.5 Flash with thinking level "low". **Use this one to rerun the evaluation**: it covers all nine evaluators and eleven writers |
| `build_score_caches.py` | Builds `master_df` and `competitive_df` from `evaluations.csv` |
| `compute_cv_job_similarity.py` | CV–job cosine similarity (all-MiniLM-L6-v2) → `output_eval/cv_job_similarity.parquet` |
| `extract_cl_features.py` | Writing-style features of every letter → `output_eval/cl_features.parquet` |
| `aggregate_plots.py` | Score-based analyses: preference gap, rank-gap bias, win rates, displacement, calibration |
| `plot_feature_regression.py` | Ridge regression of scores on writing-style features |
| `plot_nested_model_decomposition.py`, `plot_model_explainability.py` | Shapley R² decompositions |
| `plot_cl_features.py`, `plot_cl_variance.py` | Writer text-feature descriptives |
| `generate_all_paper_figures.py` | Runs the six analysis scripts in order (canonical entry point) |
| `output_eval/` | Parquet caches: all 414,000 scores (`master_df`), displacement input (`competitive_df`), letter features, CV–job similarity |

## Data

The raw inputs are not in this repository:

- **CVs and job descriptions**: public Kaggle datasets (Resume Dataset; job
  descriptions dataset), cited in the paper. Expected at `raw_datasets/`
  (`resume_data.csv`, `job_desc_data.csv`) and, after selection, `dataset/`.
- **NRC-VAD lexicon v2.1**: `raw_datasets/NRC-VAD-Lexicon-v2.1.txt`
  (download from the NRC website; research licence).
- **Generated cover letters** (`output_cl/`) and **raw evaluator outputs**:
  available from the corresponding author.

In the authors' working copy these sit in `../data/` and are linked into
the repo as `raw_datasets`, `dataset`, `output_cl` (gitignored).

## Notes

- Evaluator scores are parsed from `<final>…</final>` tags; the score per
  (job, candidate, writer, evaluator) is the mean over four runs.
- An early run also included Gemini 2.0 Flash as writer and evaluator. It was
  dropped because it is much older than the other models, and its outputs are
  not part of the released data or of any result.
- API keys are read from `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
  `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`. Local models run through Ollama.
