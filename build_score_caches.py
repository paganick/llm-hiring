"""Build the score caches used by the figure pipeline from evaluations.csv.

    python build_score_caches.py [--evaluations PATH] [--out DIR]

Writes master_df.parquet and competitive_df.parquet.
"""
import argparse
import os

import pandas as pd

EVAL_TYPES = {"cl_only": "cl_evaluations",
              "cv_cl": "cv_cl_evaluations",
              "cv_only": "cv_only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations",
                        default="../data/compressed_output_fixed/evaluations.csv")
    parser.add_argument("--out", default="output_eval")
    args = parser.parse_args()

    columns = ["job_id", "job_folder", "run", "eval_type", "evaluator_model",
               "generator_model", "cv", "score"]
    ev = pd.read_csv(args.evaluations, usecols=columns)

    master = pd.DataFrame({
        "Job_ID": ev["job_id"],
        "Job_Title": (ev["job_folder"].str.replace(r"^job_\d+_", "", regex=True)
                      .str.replace("_", " ")),
        "Eval_Type": ev["eval_type"].map(EVAL_TYPES),
        "Evaluator": ev["evaluator_model"],
        "Writer": ev["generator_model"].fillna("CV_ONLY"),
        "CV_Idx": ev["cv"],
        "Score": ev["score"],
        "Run": ev["run"],
    })
    competitive = (master[master["Eval_Type"] != "cl_evaluations"]
                   .rename(columns={"Job_ID": "Job", "Eval_Type": "Type"})
                   [["Job", "Run", "Evaluator", "Writer", "CV_Idx", "Score", "Type"]])

    os.makedirs(args.out, exist_ok=True)
    master.to_parquet(os.path.join(args.out, "master_df.parquet"), index=False)
    competitive.to_parquet(os.path.join(args.out, "competitive_df.parquet"), index=False)
    print(f"master: {len(master):,} rows; competitive: {len(competitive):,} rows → {args.out}")


if __name__ == "__main__":
    main()
