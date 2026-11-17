import math
from collections import Counter

import pandas as pd


def query_clarity(run_dir, category) -> pd.DataFrame:
    """
    Hypothesis: a group receives more exposure when the query's language departs more from
    that group's language
    """
    topics = pd.read_parquet(run_dir / "topics.parquet", columns=["qid", "gep_terms"])

    # frequencies of the stemmed query terms within each group of this category,
    # and the size of each group in tokens
    stats = pd.read_parquet(
        run_dir / "term_stats.parquet",
        columns=["term", "category", "group", "group_cf", "group_n_tokens"],
    )
    stats = stats[stats["category"] == category]
    group_cf = {g: s.set_index("term")["group_cf"] for g, s in stats.groupby("group")}
    group_tokens = stats.groupby("group")["group_n_tokens"].first()

    def score(terms, group):
        n_tokens = group_tokens[group]
        cf = group_cf[group]
        # ql is the number of (non-stop) query terms, qtf the count of each one
        ql = len(terms)
        qtf = Counter(terms)

        clarity = 0.0
        for term, count in qtf.items():
            # skip terms that do not appear in the group's documents
            if cf.get(term, 0) == 0:
                continue
            p_q = count / ql
            p_group = cf[term] / n_tokens
            clarity += p_q * math.log2(p_q / p_group)

        # a group with no query terms scores 0, and shares cannot be negative
        return max(clarity, 0.0)

    rows = []
    for qid, terms in zip(topics["qid"], topics["gep_terms"]):
        scores = {g: score(terms, g) for g in group_cf}
        total = sum(scores.values())
        for g, s in scores.items():
            # fall back to equal shares when every group scores 0
            rows.append((qid, g, s / total if total > 0 else 1 / len(scores)))

    return pd.DataFrame(rows, columns=["qid", "group", "prediction"])