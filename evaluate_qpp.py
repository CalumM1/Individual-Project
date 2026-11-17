import argparse
import inspect
from functools import partial
from importlib import import_module

import pandas as pd
from scipy.stats import pearsonr, kendalltau, spearmanr

from first_pass_retrieval import DATASETS, RUNS_DIR, ROOT

QPP_RESULTS_DIR = ROOT / "results" / "qpp"

# trec fair counts every article in a wikiproject as relevant, so every query has thousands of
# relevant documents, far more than the 1000 retrieved. recall based metrics (AP, AP@100, R@1000)
# then mostly reflect how many relevant documents a query has rather than how well it was ranked,
# so only precision oriented metrics are used. datasets not listed here use every metric
PRECISION_METRICS = ["nDCG@10", "nDCG@100", "P@10", "RR"]
QPP_METRICS = {
    "trec_fair_2021_train": PRECISION_METRICS,
    "trec_fair_2022_train": PRECISION_METRICS,
}


def evaluate_qpp(preds, perq, system, metric):
    
    actual_values = perq.loc[perq["name"] == system, ["qid", metric]]
    df = actual_values.merge(preds, on="qid", how="inner")

    if len(df) < len(actual_values):
        print(f"warning: {len(actual_values) - len(df)} queries have no prediction")

    real, predicted = df[metric], df["prediction"]
    return {
        "pearson": pearsonr(real, predicted)[0],
        "kendall": kendalltau(real, predicted)[0],
        "spearman": spearmanr(real, predicted)[0],
        "n": len(df),
    }


def load_predictor(model):
    try:
        module_name, function_name = model.rsplit(":", 1)
    except ValueError as exc:
        raise ValueError("Model must be specified as module:function") from exc

    predictor = getattr(import_module(module_name), function_name)
    if not callable(predictor):
        raise TypeError(f"Selected model is not callable: {model}")
    return predictor


def main(model, results_file=None, training_queries=None):
    predictor = load_predictor(model)
    if training_queries is not None:
        predictor = partial(predictor, training_queries=training_queries)
    if results_file is None:
        results_file = {
            "word_count_predictor": "qpp_query_length_results",
            "significant_word_predictor": "qpp_significant_words_results",
            "syllable_term_predictor": "qpp_syllable_term_results",
        }.get(model.rsplit(":", 1)[-1], model.rsplit(":", 1)[-1])

    rows = []
    for dataset in DATASETS:
        run_dir = RUNS_DIR / dataset
        if not (run_dir / "perquery_metrics.parquet").exists():
            print(f"skipping {dataset}: no results yet")
            continue

        topics = pd.read_parquet(run_dir / "topics.parquet")
        perq = pd.read_parquet(run_dir / "perquery_metrics.parquet")
        # predictors that need collection statistics can ask for the run directory
        if "run_dir" in inspect.signature(predictor).parameters:
            preds = predictor(topics, run_dir=run_dir)
        else:
            preds = predictor(topics)

        metrics = QPP_METRICS.get(dataset, [c for c in perq.columns if c not in ("name", "qid")])
        for system in perq["name"].unique():
            for metric in metrics:
                res = evaluate_qpp(preds, perq, system, metric)
                rows.append({"dataset": dataset, "system": system, "metric": metric, **res})

    results = pd.DataFrame(rows)
    results.to_csv(QPP_RESULTS_DIR / f"{results_file}.csv", index=False)
    pd.set_option("display.width", 200)
    print(results.round(3).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a query performance prediction model.",
        epilog=(
            "Example: python evaluate_qpp.py "
            "predictors.qpp.word_count_predictor:word_count_predictor. "
            "Select a different function to run another model from the same module."
        ),
    )
    parser.add_argument(
        "model",
        help="Prediction model in module:function format",
    )
    parser.add_argument(
        "--results-file",
        help="Output CSV name (without .csv); defaults to the model name",
    )
    parser.add_argument(
        "--training-queries",
        help="Training dataset directory, for models that require training data",
    )
    args = parser.parse_args()
    main(args.model, args.results_file, args.training_queries)
