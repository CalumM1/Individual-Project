import argparse
import re

import pandas as pd

from evaluate_qep import actual_exposure, evaluate_qep, load_predictor
from first_pass_retrieval import DATASETS, LLM_DATASETS, RUNS_DIR, ROOT, WMODELS

QEP_TOPIC_RESULTS_DIR = ROOT / "results" / "qep_topics"


def year_of(folder):
    return re.search(r"20\d\d", folder).group()


def main(model, folders=None):
    predictor = load_predictor(model)

    QEP_TOPIC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
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

        rows = []
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

        # one results file per year
        results = pd.DataFrame(rows)
        out = QEP_TOPIC_RESULTS_DIR / f"main_results_{year_of(folder)}.csv"
        results.to_csv(out, index=False)
        print(results.round(4).to_string(index=False))
        print(f"wrote {out}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a query exposure prediction model separately for each topic.",
        epilog="Example: python evaluate_qep_topics.py predictors.qep.gep:gep_predictor",
    )
    parser.add_argument("model", help="Prediction model in module:function format")
    parser.add_argument("--folders", nargs="*", choices=list(LLM_DATASETS),
                        help="LLM run folders to evaluate (default: all)")
    args = parser.parse_args()
    main(args.model, args.folders)