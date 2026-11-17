import math

import pandas as pd
import numpy as np

from predictors.qpp.term_info_distrobution import idf


def dist_informative_amount1(run_dir, category) -> pd.DataFrame:
    """
    Hypothesis: a group receives more exposure when the query's terms vary more in
    informativeness within that group's documents, measured per group as the ratio of the
    largest to the smallest query term idf, with idf computed over the group's documents
    and the ratios normalised into exposure shares across the groups
    """
    topics = pd.read_parquet(run_dir / "topics.parquet", columns=["qid", "gep_terms"])

    # document frequencies of the stemmed query terms within each group of this category
    stats = pd.read_parquet(run_dir / "term_stats.parquet", columns=["term", "category", "group", "group_df"])
    stats = stats[stats["category"] == category]
    group_df = {g: s.set_index("term")["group_df"] for g, s in stats.groupby("group")}

    # number of documents in each group (a document can belong to several groups)
    labels = pd.read_parquet(run_dir / "groups.parquet", columns=[category])[category]
    group_size = labels.explode().value_counts()

    def get_max_and_min_idf(terms, group):
        n = group_size.get(group, 0)
        df = group_df[group]
        max_idf = 0
        min_idf = math.inf

        for term in terms:
            # skip terms that do not appear in the group's documents
            if df.get(term, 0) == 0:
                continue
            term_idf = idf(n, df[term])
            max_idf = max(max_idf, term_idf)
            min_idf = min(min_idf, term_idf)

        return max_idf, min_idf

    def score(terms, group):
        max_idf, min_idf = get_max_and_min_idf(terms, group)
        # no query term appears in the group, so it is predicted no exposure
        if min_idf == math.inf:
            return 0.0
        return max_idf / min_idf

    rows = []
    for qid, terms in zip(topics["qid"], topics["gep_terms"]):
        scores = {g: score(terms, g) for g in group_df}
        total = sum(scores.values())
        for g, s in scores.items():
            # fall back to equal shares when no group contains any query term
            rows.append((qid, g, s / total if total > 0 else 1 / len(scores)))

    return pd.DataFrame(rows, columns=["qid", "group", "prediction"])


def dist_informative_amount2(run_dir, category) -> pd.DataFrame:
    """
    Hypothesis: a group receives more exposure when the query's terms vary more in
    informativeness within that group's documents, measured per group as the standard
    deviation of the query term idfs, with idf computed over the group's documents and the
    deviations normalised into exposure shares across the groups
    """
    topics = pd.read_parquet(run_dir / "topics.parquet", columns=["qid", "gep_terms"])
 
    # document frequencies of the stemmed query terms within each group of this category
    stats = pd.read_parquet(run_dir / "term_stats.parquet", columns=["term", "category", "group", "group_df"])
    stats = stats[stats["category"] == category]
    group_df = {g: s.set_index("term")["group_df"] for g, s in stats.groupby("group")}
 
    # number of documents in each group (a document can belong to several groups)
    labels = pd.read_parquet(run_dir / "groups.parquet", columns=[category])[category]
    group_size = labels.explode().value_counts()
 
    def score(terms, group):
        n = group_size.get(group, 0)
        df = group_df[group]
        # skip terms that do not appear in the group's documents
        idfs = [idf(n, df[term]) for term in terms if df.get(term, 0) > 0]
        if not idfs:
            return 0.0
        # population standard deviation, so a group matching a single term scores 0
        return float(np.std(idfs))
 
    rows = []
    for qid, terms in zip(topics["qid"], topics["gep_terms"]):
        scores = {g: score(terms, g) for g in group_df}
        total = sum(scores.values())
        for g, s in scores.items():
            # fall back to equal shares when every group scores 0
            rows.append((qid, g, s / total if total > 0 else 1 / len(scores)))
 
    return pd.DataFrame(rows, columns=["qid", "group", "prediction"])