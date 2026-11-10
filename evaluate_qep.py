import math
from collections import Counter

import pandas as pd
from scipy.spatial.distance import jensenshannon

from first_pass_retrieval import DATASETS, RUNS_DIR, ROOT, WMODELS

QEP_RESULTS_DIR = ROOT / "results" / "qep"


def actual_exposure(run, labels, k=100):
    rows = []
    for qid, ranking in run.sort_values("rank").groupby("qid"):
        # sum the exposure each group gets from its positions in the top k
        exposure = Counter()
        for p, docno in enumerate(ranking["docno"].head(k), 1):
            for g in labels.get(docno, []):
                exposure[g] += 1 / math.log2(p + 1)

        # normalise the actual exposure into shares
        total = sum(exposure.values())
        rows += [(qid, g, v / total) for g, v in exposure.items()]
    return pd.DataFrame(rows, columns=["qid", "group", "actual"])


def evaluate_qep(preds, actual):
    queries = actual["qid"].unique()
    df = preds[preds["qid"].isin(queries)].merge(actual, on=["qid", "group"], how="left")
    df["actual"] = df["actual"].fillna(0.0)

    if df["qid"].nunique() < len(queries):
        print(f"warning: {len(queries) - df['qid'].nunique()} queries have no prediction")

    # measure the jensen shannon distance between predicted and actual shares
    jsd = df.groupby("qid").apply(lambda q: jensenshannon(q["prediction"], q["actual"]))
    return {"jsd": jsd.mean(), "jsd_std": jsd.std(), "n": len(jsd)}


def main(predictor, results_file):
    QEP_RESULTS_DIR.mkdir(exist_ok=True)
    rows = []
    for dataset, cfg in DATASETS.items():
        run_dir = RUNS_DIR / dataset
        if not (run_dir / "term_stats.parquet").exists():
            print(f"skipping {dataset}: no results yet")
            continue

        groups = pd.read_parquet(run_dir / "groups.parquet")

        for category in cfg["categories"]:
            preds = predictor(run_dir, category)
            labels = groups.set_index("docno")[category]

            for system in WMODELS:
                run = pd.read_parquet(run_dir / f"{system}.run.parquet", columns=["qid", "docno", "rank"])
                res = evaluate_qep(preds, actual_exposure(run, labels))
                rows.append({"dataset": dataset, "system": system, "category": category, **res})

    results = pd.DataFrame(rows)
    results.to_csv(QEP_RESULTS_DIR / f"{results_file}.csv", index=False)
    pd.set_option("display.width", 200)
    print(results.round(4).to_string(index=False))