import pandas as pd
from scipy.stats import pearsonr, kendalltau, spearmanr

from first_pass_retrieval import DATASETS, RUNS_DIR, ROOT

QPP_RESULTS_DIR = ROOT / "qpp_results"


def evaluate_qpp(preds, perq, system, metric):
    
    gt = perq.loc[perq["name"] == system, ["qid", metric]]
    df = gt.merge(preds, on="qid", how="inner")

    if len(df) < len(gt):
        print(f"warning: {len(gt) - len(df)} queries have no prediction")

    y, p = df[metric], df["prediction"]
    return {
        "pearson": pearsonr(y, p)[0],
        "kendall": kendalltau(y, p)[0],
        "spearman": spearmanr(y, p)[0],
        "n": len(df),
    }


def main(predictor, results_file):
    rows = []
    for dataset in DATASETS:
        run_dir = RUNS_DIR / dataset
        if not (run_dir / "perquery_metrics.parquet").exists():
            print(f"skipping {dataset}: no results yet")
            continue

        topics = pd.read_parquet(run_dir / "topics.parquet")
        perq = pd.read_parquet(run_dir / "perquery_metrics.parquet")
        preds = predictor(topics)

        metrics = [c for c in perq.columns if c not in ("name", "qid")]
        for system in perq["name"].unique():
            for metric in metrics:
                res = evaluate_qpp(preds, perq, system, metric)
                rows.append({"dataset": dataset, "system": system, "metric": metric, **res})

    results = pd.DataFrame(rows)
    results.to_csv(QPP_RESULTS_DIR / f"{results_file}.csv", index=False)
    pd.set_option("display.width", 200)
    print(results.round(3).to_string(index=False))


if __name__ == "__main__":
    main()