from pathlib import Path

import ir_datasets
import numpy as np
import pandas as pd
import pyterrier as pt
from pyterrier.measures import nDCG, AP, RR, P, R

ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "indexes"
RUNS_DIR = ROOT / "runs"

WMODELS = {"bm25": "BM25", "dirichlet": "DirichletLM", "tfidf": "TF_IDF"}

GEP_K = 100

DATASETS = {
    "trec_fair_2021_train": {
        "index": "trec_fair_2021.terrier",
        "dataset": "irds:trec-fair/2021/train",
        "variant": "text",
        "categories": {"geo_location": "geographic_locations"},
    },
    "trec_fair_2022_train": {
        "index": "trec_fair_2022.terrier",
        "dataset": "irds:trec-fair/2022/train",
        "variant": "text",
        "categories": {
            "age_of_article": "creation_date_category",
            "age_of_topic": "years_category",
            "alphabetical": "first_letter_category",
            "popularity": "relative_pageviews_category",
        },
    },
}


def as_groups(value):
    values = value if isinstance(value, (list, tuple)) else [value]
    values = [str(v) for v in values if v not in (None, "")]
    return values or ["Unknown"]


def save_groups(path, dataset, categories):
    rows = {"docno": []} | {c: [] for c in categories}
    for doc in ir_datasets.load(dataset.removeprefix("irds:")).docs_iter():
        rows["docno"].append(doc.doc_id)
        for cat, field in categories.items():
            rows[cat].append(as_groups(getattr(doc, field)))
    groups = pd.DataFrame(rows)
    groups.to_parquet(path)
    return groups


def query_terms(topics):
    stemmer = pt.java.autoclass("org.terrier.terms.PorterStemmer")()
    return topics["query"].map(lambda q: [stemmer.stem(w) for w in q.lower().split()])


def save_term_stats(path, idx, terms, groups, categories, k=GEP_K):
    lex = idx.getLexicon()
    inv = idx.getInvertedIndex()
    meta = idx.getMetaIndex()
    n = idx.getCollectionStatistics().getNumberOfDocuments()

    docnos = [meta.getItem("docno", i) for i in range(n)]
    by_docid = groups.set_index("docno").reindex(docnos)

    masks = {}
    for cat in categories:
        labels = by_docid[cat].map(lambda v: v if isinstance(v, (list, np.ndarray)) else ["Unknown"])
        names = sorted({g for gs in labels for g in gs})
        bits = np.zeros(n, dtype=np.uint32)
        for i, name in enumerate(names):
            bits |= labels.map(lambda gs: name in gs).to_numpy(np.uint32) << i
        masks[cat] = (names, bits)

    rows = []
    for term in sorted({t for ts in terms for t in ts}):
        le = lex.getLexiconEntry(term)
        if le is None:
            continue
        ids, tfs = [], []
        for posting in inv.getPostings(le):
            ids.append(posting.getId())
            tfs.append(posting.getFrequency())
        ids, tfs = np.array(ids), np.array(tfs)
        for cat, (names, bits) in masks.items():
            for i, name in enumerate(names):
                in_group = ((bits[ids] >> i) & 1).astype(bool)
                top = np.sort(tfs[in_group])[::-1][:k]
                rows.append((term, le.getDocumentFrequency(), n, cat, name,
                             int(in_group.sum()), top.tolist()))

    stats = pd.DataFrame(rows, columns=["term", "df", "n_docs", "category", "group",
                                        "group_df", "top_tfs"])
    stats.to_parquet(path)


def run(name, index, dataset, categories, rel=1, variant=None, depth=1000):
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

    groups_path = out / "groups.parquet"
    if groups_path.exists():
        groups = pd.read_parquet(groups_path)
    else:
        print(f"[{name}] reading group labels from the corpus")
        groups = save_groups(groups_path, dataset, categories)

    if groups["docno"].duplicated().any():
        print(f"[{name}] dropping {groups['docno'].duplicated().sum()} duplicate docnos from groups")
        groups = groups.drop_duplicates("docno")
        groups.to_parquet(groups_path)

    topics["gep_terms"] = query_terms(topics)
    stats_path = out / "term_stats.parquet"
    if not stats_path.exists():
        print(f"[{name}] collecting term statistics for gep")
        save_term_stats(stats_path, idx, topics["gep_terms"], groups, categories)

    topics.to_parquet(out / "topics.parquet")
    qrels.to_parquet(out / "qrels.parquet")


if __name__ == "__main__":
    for name, cfg in DATASETS.items():
        run(name, **cfg)

