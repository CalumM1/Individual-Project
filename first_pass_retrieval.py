import argparse
import json
import shutil
from pathlib import Path

import ir_datasets
import numpy as np
import pandas as pd
import pyterrier as pt
from pyterrier.measures import nDCG, AP, RR, P, R

ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "indexes"
RUNS_DIR = ROOT / "runs"
QUERIES_DIR = ROOT / "queries"

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

# LLM query sets: kept separate from DATASETS so evaluate_qep.py does not pick them up
LLM_DATASETS = {
    "trec_fair_2021_llm": {"base": "trec_fair_2021_train",
                           "queries": "trec_fair_2021_train_llm_queries.json"},
    "trec_fair_2022_llm": {"base": "trec_fair_2022_train",
                           "queries": "trec_fair_2022_train_llm_queries.json"},
}

# files derived from the topics; deleted by --rebuild
DERIVED = ["bm25.run.parquet", "dirichlet.run.parquet", "tfidf.run.parquet",
           "perquery_metrics.parquet", "summary_metrics.csv", "term_stats.parquet",
           "query_stats.parquet", "query_group_stats.parquet", "topics.parquet", "qrels.parquet"]


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


def load_groups(out, dataset=None, categories=None, groups_from=None):
    # read the folder's groups.parquet, copying it from groups_from or building it from the
    # corpus if missing
    name = out.name
    path = out / "groups.parquet"
    if not path.exists():
        if groups_from is not None:
            print(f"[{name}] copying group labels from {groups_from.name}")
            shutil.copy(groups_from / "groups.parquet", path)
        else:
            print(f"[{name}] reading group labels from the corpus")
            save_groups(path, dataset, categories)
    groups = pd.read_parquet(path)
    if groups["docno"].duplicated().any():
        print(f"[{name}] dropping {groups['docno'].duplicated().sum()} duplicate docnos from groups")
        groups = groups.drop_duplicates("docno")
        groups.to_parquet(path)
    return groups


def clean_queries(topics, name):
    topics = topics.copy()
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
    return topics


def query_terms(topics):
    # match terrier's query pipeline: drop stopwords, then stem
    stopwords = pt.java.autoclass("org.terrier.terms.Stopwords")(None)
    stemmer = pt.java.autoclass("org.terrier.terms.PorterStemmer")()
    return topics["query"].map(
        lambda q: [stemmer.stem(w) for w in q.lower().split() if not stopwords.isStopword(w)]
    )


def irds_topics(dataset, variant=None):
    ds = pt.get_dataset(dataset)
    topics = ds.get_topics(variant=variant)
    qrels = ds.get_qrels()
    topics = topics[topics["qid"].isin(qrels["qid"].unique())].copy()
    return topics, qrels


def llm_topics(path):
    # one row per generated query, keeping the topic it was generated for
    rows = [{"qid": q["qid"], "query": q["query"], "topic_id": t["id"], "topic_title": t["title"]}
            for t in json.load(open(path)) for q in t["queries"]]
    return pd.DataFrame(rows)


def filter_llm_topics(topics, idx, n_per_topic=None, out=None):
    # drop queries with no indexed terms, queries whose stemmed terms repeat an earlier query of
    # the same topic, and (optionally) all but the first n_per_topic per topic
    lex = idx.getLexicon()
    known = topics["gep_terms"].map(lambda ts: tuple(sorted({t for t in ts if lex.getLexiconEntry(t) is not None})))
    reason = pd.Series("", index=topics.index)
    reason[known.map(len) == 0] = "no indexed terms"
    dup = pd.Series(list(zip(topics["topic_id"], known)), index=topics.index).duplicated()
    reason[(reason == "") & dup] = "duplicate after stemming"
    if n_per_topic:
        rank = (reason == "").groupby(topics["topic_id"]).cumsum()
        reason[(reason == "") & (rank > n_per_topic)] = "over per-topic limit"

    dropped = topics[reason != ""].assign(reason=reason[reason != ""])
    kept = topics[reason == ""].copy()
    kept["n_known_terms"] = known[reason == ""].map(len)
    if out is not None:
        dropped[["qid", "topic_id", "query_raw", "reason"]].to_csv(out / "dropped_queries.csv", index=False)
    print(f"[{out.name if out else 'llm'}] kept {len(kept)} queries, dropped {len(dropped)}: "
          f"{dropped['reason'].value_counts().to_dict()}")

    counts = kept.groupby("topic_id").size()
    if n_per_topic and (counts < n_per_topic).any():
        print(f"  WARNING: topics with fewer than {n_per_topic} queries: "
              f"{counts[counts < n_per_topic].to_dict()}")
    return kept


def group_masks(idx, groups, categories):
    # for each category, the sorted group names and a bitmask per document (indexed by terrier
    # docid) with bit i set when the document belongs to group i; documents missing from
    # groups count as "Unknown"
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
    return masks


def posting_ids(idx, terms):
    # terrier docids of the documents containing each query term
    lex = idx.getLexicon()
    inv = idx.getInvertedIndex()
    postings = {}
    for term in sorted({t for ts in terms for t in ts}):
        le = lex.getLexiconEntry(term)
        if le is None:
            continue
        postings[term] = np.fromiter((p.getId() for p in inv.getPostings(le)), dtype=np.int32)
    return postings


def matched_ids(postings, terms):
    # docids of the documents containing at least one of the terms
    ids = [postings[t] for t in set(terms) if t in postings]
    return np.unique(np.concatenate(ids)) if ids else np.array([], dtype=np.int32)


def save_term_stats(path, idx, terms, groups, categories, k=GEP_K):
    lex = idx.getLexicon()
    inv = idx.getInvertedIndex()
    coll = idx.getCollectionStatistics()
    n = coll.getNumberOfDocuments()
    # total tokens in the index (after stopword removal and stemming), for collection language models
    n_tokens = coll.getNumberOfTokens()

    # document lengths in tokens, to size each group's language model
    doi = idx.getDocumentIndex()
    doc_lens = np.fromiter((doi.getDocumentLength(i) for i in range(n)), dtype=np.int64, count=n)

    masks = {}
    for cat, (names, bits) in group_masks(idx, groups, categories).items():
        # documents and tokens in each group (a document can belong to several groups)
        sizes = [(int(m.sum()), int(doc_lens[m].sum()))
                 for m in (((bits >> i) & 1).astype(bool) for i in range(len(names)))]
        masks[cat] = (names, bits, sizes)

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
        for cat, (names, bits, sizes) in masks.items():
            for i, name in enumerate(names):
                in_group = ((bits[ids] >> i) & 1).astype(bool)
                top = np.sort(tfs[in_group])[::-1][:k]
                group_n_docs, group_n_tokens = sizes[i]
                rows.append((term, le.getDocumentFrequency(), le.getFrequency(), n, n_tokens,
                             cat, name, int(in_group.sum()), int(tfs[in_group].sum()),
                             group_n_docs, group_n_tokens, top.tolist()))

    stats = pd.DataFrame(rows, columns=["term", "df", "cf", "n_docs", "n_tokens", "category",
                                        "group", "group_df", "group_cf", "group_n_docs",
                                        "group_n_tokens", "top_tfs"])
    stats.to_parquet(path)


def save_query_stats(path, idx, topics):
    # n_matched: number of documents containing at least one query term, i.e. the size of the
    # union of the query terms' posting lists (needed for query scope)
    n = idx.getCollectionStatistics().getNumberOfDocuments()
    postings = posting_ids(idx, topics["gep_terms"])

    stats = pd.DataFrame({
        "qid": topics["qid"],
        "n_matched": topics["gep_terms"].map(lambda terms: matched_ids(postings, terms).size),
        "n_docs": n,
    })
    stats.to_parquet(path)


def save_query_group_stats(path, idx, topics, groups, categories):
    # n_matched per (query, category, group): number of the group's documents containing at
    # least one query term, alongside the group's size (needed for per-group query scope)
    postings = posting_ids(idx, topics["gep_terms"])
    masks = group_masks(idx, groups, categories)

    sizes = {cat: [int(((bits >> i) & 1).sum()) for i in range(len(names))]
             for cat, (names, bits) in masks.items()}

    rows = []
    for qid, terms in zip(topics["qid"], topics["gep_terms"]):
        ids = matched_ids(postings, terms)
        for cat, (names, bits) in masks.items():
            for i, name in enumerate(names):
                n_matched = int(((bits[ids] >> i) & 1).sum())
                rows.append((qid, cat, name, n_matched, sizes[cat][i]))

    stats = pd.DataFrame(rows, columns=["qid", "category", "group", "n_matched", "group_n_docs"])
    stats.to_parquet(path)


def build_stats(out, idx, topics, groups, categories):
    name = out.name
    stats_path = out / "term_stats.parquet"
    if not stats_path.exists():
        print(f"[{name}] collecting term statistics for gep")
        save_term_stats(stats_path, idx, topics["gep_terms"], groups, categories)

    query_stats_path = out / "query_stats.parquet"
    if not query_stats_path.exists():
        print(f"[{name}] counting documents that match each query")
        save_query_stats(query_stats_path, idx, topics)

    query_group_stats_path = out / "query_group_stats.parquet"
    if not query_group_stats_path.exists():
        print(f"[{name}] counting documents in each group that match each query")
        save_query_group_stats(query_group_stats_path, idx, topics, groups, categories)


def retrieve(out, idx, topics, depth=1000):
    runs = {}
    for sys_name, wmodel in WMODELS.items():
        run_path = out / f"{sys_name}.run.parquet"
        if run_path.exists():
            runs[sys_name] = pd.read_parquet(run_path)
        else:
            print(f"[{out.name}] {sys_name}: retrieving {len(topics)} queries")
            retr = pt.terrier.Retriever(idx, wmodel=wmodel, num_results=depth)
            res = retr.transform(topics[["qid", "query"]])
            res.to_parquet(run_path)
            runs[sys_name] = res
    return runs


def evaluate_runs(out, runs, topics, qrels, rel=1, depth=1000):
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
        topics[["qid", "query"]],
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
    print(f"\n[{out.name}] {len(topics)} queries\n{summary.round(4)}")


def check_topics(out, topics, rebuild=False):
    # make sure the folder's existing files were built for these queries
    path = out / "topics.parquet"
    if path.exists():
        old = pd.read_parquet(path, columns=["qid", "gep_terms"])
        same = (len(old) == len(topics)
                and (old["qid"].to_numpy() == topics["qid"].to_numpy()).all()
                and all(list(a) == list(b) for a, b in zip(old["gep_terms"], topics["gep_terms"])))
        if same:
            return
        if not rebuild:
            raise SystemExit(f"[{out.name}] the queries differ from those in {path}; files in this "
                             f"folder were built for the old queries. Rerun with --rebuild to "
                             f"replace them, or use a different folder name.")
        print(f"[{out.name}] queries changed: deleting derived files")
        for f in DERIVED:
            (out / f).unlink(missing_ok=True)
    topics.to_parquet(path)


def build_run_dir(name, index, topics, categories, dataset=None, qrels=None, rel=1,
                  depth=1000, groups_from=None, rebuild=False, prepare=None):
    """Build runs/<name> for any set of queries.

    topics: DataFrame with at least qid and query (raw text; cleaned here).
    qrels: optional; effectiveness metrics are computed only when given.
    groups_from: run folder to copy groups.parquet from instead of reading the corpus.
    prepare: optional fn(topics, idx, out) -> topics, applied after gep_terms are added
             and before anything is retrieved (used to filter LLM queries).
    """
    out = RUNS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    idx = pt.IndexFactory.of(str(INDEX_DIR / index))

    topics = clean_queries(topics, name)
    topics["gep_terms"] = query_terms(topics)
    if prepare is not None:
        topics = prepare(topics, idx, out)
    topics = topics.reset_index(drop=True)
    check_topics(out, topics, rebuild)

    runs = retrieve(out, idx, topics, depth)
    if qrels is not None:
        evaluate_runs(out, runs, topics, qrels, rel, depth)
        qrels.to_parquet(out / "qrels.parquet")

    groups = load_groups(out, dataset, categories, groups_from)
    build_stats(out, idx, topics, groups, categories)
    return out


def run(name, index, dataset, categories, rel=1, variant=None, depth=1000, rebuild=False):
    # a TREC dataset: topics and qrels from ir_datasets
    topics, qrels = irds_topics(dataset, variant)
    return build_run_dir(name, index, topics, categories, dataset=dataset, qrels=qrels,
                         rel=rel, depth=depth, rebuild=rebuild)


def run_llm(name, base, queries, n_per_topic=None, depth=1000, rebuild=False):
    # an LLM query set: topics from JSON, index and groups from the base dataset, no qrels
    cfg = DATASETS[base]
    topics = llm_topics(QUERIES_DIR / queries)
    return build_run_dir(
        name, cfg["index"], topics, cfg["categories"], dataset=cfg["dataset"], depth=depth,
        groups_from=RUNS_DIR / base, rebuild=rebuild,
        prepare=lambda t, idx, out: filter_llm_topics(t, idx, n_per_topic, out),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", choices=list(DATASETS),
                    help="TREC datasets to build (default: all, unless --llm is given)")
    ap.add_argument("--llm", nargs="*", choices=list(LLM_DATASETS) + ["all"],
                    help="LLM query sets to build")
    ap.add_argument("--n-per-topic", type=int, default=None,
                    help="LLM sets: keep the first N usable queries per topic (default: all)")
    ap.add_argument("--depth", type=int, default=1000)
    ap.add_argument("--rebuild", action="store_true",
                    help="replace a folder's files if its queries have changed")
    args = ap.parse_args()

    trec = args.datasets if args.datasets is not None else ([] if args.llm else list(DATASETS))
    llm = list(LLM_DATASETS) if args.llm == ["all"] else (args.llm or [])

    for name in trec:
        run(name, **DATASETS[name], depth=args.depth, rebuild=args.rebuild)
    for name in llm:
        run_llm(name, **LLM_DATASETS[name], n_per_topic=args.n_per_topic,
                depth=args.depth, rebuild=args.rebuild)


if __name__ == "__main__":
    main()