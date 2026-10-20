from pathlib import Path

import pandas as pd
import pyterrier as pt
from pyterrier.measures import nDCG, AP, RR, P, R

ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "indexes"
RUNS_DIR = ROOT / "runs"

WMODELS = {"bm25": "BM25", "dirichlet": "DirichletLM", "tfidf": "TF_IDF"}


DATASETS = {
    "trec_fair2021_train": {
        "index": "trec-fair-2021.terrier",
        "dataset": "irds:trec-fair/2021/train",
        "variant": "text",
    },
    "trec_fair2022_train": {
        "index": "trec-fair-2022.terrier",
        "dataset": "irds:trec-fair/2022/train",
        "variant": "text",
    },
    "antique_test": {
        "index": "antique_train.terrier",
        "dataset": "irds:antique/test",
        "rel": 3,
    },
    # "beir_fiqa_test": {
    #     "index": "beir_fiqa_train.terrier",
    #     "dataset": "irds:beir/fiqa/test",
    # },
    # "beir_quora_test": {
    #     "index": "beir_quora_test.terrier",
    #     "dataset": "irds:beir/quora/test",
    # },
    "cord19_trec_covid": {
        "index": "cord19_trec_covid.terrier",
        "dataset": "irds:cord19/trec-covid",
        "variant": "title",
    },
    "msmarco_doc_dl2020": {
        "index": "msmarco-document_trec-dl-2020_judged.terrier",
        "dataset": "irds:msmarco-document/trec-dl-2020/judged",
    },
}


def run(name, index, dataset, rel=1, variant=None, depth=1000):
    out = RUNS_DIR / name
    out.mkdir(parents=True, exist_ok=True)

    ds = pt.get_dataset(dataset)
    topics = ds.get_topics(variant=variant)
    qrels = ds.get_qrels()

    topics = topics[topics["qid"].isin(qrels["qid"].unique())].copy()

    topics["query_raw"] = topics["query"]
    topics["query"] = (
        topics["query"]
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    empty = topics["query"] == ""
    if empty.any():
        print(f"[{name}] dropping {empty.sum()} queries that are empty after cleaning")
        topics = topics[~empty]

    idx = pt.IndexFactory.of(str(INDEX_DIR / index))

    runs = {}
    for sys_name, wmodel in WMODELS.items():
        run_path = out / f"{sys_name}.run.parquet"
        if run_path.exists():
            runs[sys_name] = pd.read_parquet(run_path)
        else:
            retr = pt.terrier.Retriever(idx, wmodel=wmodel, num_results=depth)
            res = retr.transform(topics)
            res.to_parquet(run_path)
            runs[sys_name] = res

    metrics = [
        nDCG @ 10,
        nDCG @ 100,
        AP(rel=rel),
        AP(rel=rel) @ 100,
        RR(rel=rel),
        P(rel=rel) @ 10,
        R(rel=rel) @ depth,
    ]

    perq = pt.Experiment(
        list(runs.values()),
        topics,
        qrels,
        eval_metrics=metrics,
        names=list(runs.keys()),
        perquery=True,
    )
    perq = perq.pivot_table(
        index=["name", "qid"], columns="measure", values="value"
    ).reset_index()

    full = pd.MultiIndex.from_product(
        [list(runs.keys()), topics["qid"]], names=["name", "qid"]
    )
    perq = perq.set_index(["name", "qid"]).reindex(full, fill_value=0.0).reset_index()
    perq.columns.name = None
    perq.to_parquet(out / "perquery_metrics.parquet")

    summary = perq.drop(columns="qid").groupby("name").mean()
    summary.to_csv(out / "summary_metrics.csv")
    print(f"\n[{name}] {len(topics)} queries\n{summary.round(4)}")

    topics.to_parquet(out / "topics.parquet")
    qrels.to_parquet(out / "qrels.parquet")

