import argparse

import pandas as pd

from evaluate_qep import actual_exposure, evaluate_qep, load_predictor
from first_pass_retrieval import DATASETS, LLM_DATASETS, RUNS_DIR, ROOT, WMODELS

QEP_TOPIC_RESULTS_DIR = ROOT / "results" / "qep_topics"


def main(model, results_file=None, folders=None):
    predictor = load_predictor(model)
    if results_file is None:
        results_file = model.rsplit(":", 1)[-1]

    QEP_TOPIC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for folder in folders or LLM_DATASETS:
        run_dir = RUNS_DIR / folder
        needed = ["topics.parquet", "groups.parquet", "term_stats.parquet",
                  *(f"{s}.run.parquet" for s in WMODELS)]
        missing = [f for f in needed if not (run_dir / f).exists()]
        if missing:
            print(f"skipping {folder}: missing {', '.join(missing)}")
            continue

        topics = pd.read_parquet(run_dir / "topics.parquet", columns=["qid", "topic_id", "topic_title"])
        groups = pd.read_parquet(run_dir / "groups.parquet")
        categories = DATASETS[LLM_DATASETS[folder]["base"]]["categories"]

        for category in categories:
            # predict once for every query in the folder, then score each topic's queries
            preds = predictor(run_dir, category)
            labels = groups.set_index("docno")[category]

            for system in WMODELS:
                run = pd.read_parquet(run_dir / f"{system}.run.parquet", columns=["qid", "docno", "rank"])
                actual = actual_exposure(run, labels)

                for (topic_id, title), topic in topics.groupby(["topic_id", "topic_title"]):
                    topic_actual = actual[actual["qid"].isin(topic["qid"])]
                    res = evaluate_qep(preds, topic_actual)
                    rows.append({"dataset": folder, "system": system, "category": category,
                                 "topic_id": topic_id, "topic_title": title, **res})

    results = pd.DataFrame(rows)
    results.to_csv(QEP_TOPIC_RESULTS_DIR / f"{results_file}.csv", index=False)
    pd.set_option("display.width", 200)
    print(results.round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a query exposure prediction model separately for each topic.",
        epilog="Example: python evaluate_qep_topics.py predictors.qep.gep:gep_predictor",
    )
    parser.add_argument("model", help="Prediction model in module:function format")
    parser.add_argument("--results-file", help="Output CSV name (without .csv); defaults to the model name")
    parser.add_argument("--folders", nargs="*", choices=list(LLM_DATASETS),
                        help="LLM run folders to evaluate (default: all)")
    args = parser.parse_args()
    main(args.model, args.results_file, args.folders)