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


def query_clarity_inverse(run_dir, category, lam=0.9) -> pd.DataFrame:
    """
    Hypothesis: a group receives more exposure when the query's language is closer to that
    group's language, measured per group as 2^(-SCS_g), where SCS_g is the simplified
    clarity score of the query against the group's language model,
 
        SCS_g = sum_{w in Q} P_ml(w|Q) * log2(P_ml(w|Q) / P_g(w))
 
    P_g(w) is smoothed with the collection model so that every group has a finite score,
 
        P_g(w) = lam * tf_g / token_g + (1 - lam) * tf_coll / token_coll
 
    Normalising 2^(-SCS_g) across groups makes each share proportional to the geometric mean
    of the query term probabilities under the group's language model
    """
    topics = pd.read_parquet(run_dir / "topics.parquet", columns=["qid", "gep_terms"])
 
    # frequencies of the stemmed query terms within each group of this category and over the
    # whole collection, and the size in tokens of each group and of the collection
    stats = pd.read_parquet(
        run_dir / "term_stats.parquet",
        columns=["term", "cf", "n_tokens", "category", "group", "group_cf", "group_n_tokens"],
    )
    n_tokens = stats["n_tokens"].iloc[0]
    cf = stats.groupby("term")["cf"].first()
    stats = stats[stats["category"] == category]
    group_cf = {g: s.set_index("term")["group_cf"] for g, s in stats.groupby("group")}
    group_tokens = stats.groupby("group")["group_n_tokens"].first()
 
    def clarity(terms, group):
        g_cf = group_cf[group]
        g_tokens = group_tokens[group]
        ql = len(terms)
        qtf = Counter(terms)
 
        scs = 0.0
        matched = False
        for term, count in qtf.items():
            # skip terms that do not appear in the collection at all
            if cf.get(term, 0) == 0:
                continue
            p_q = count / ql
            p_group = g_cf.get(term, 0) / g_tokens if g_tokens > 0 else 0.0
            p_smoothed = lam * p_group + (1 - lam) * cf[term] / n_tokens
            scs += p_q * math.log2(p_q / p_smoothed)
            matched = True
 
        return scs if matched else None
 
    rows = []
    for qid, terms in zip(topics["qid"], topics["gep_terms"]):
        scores = {g: clarity(terms, g) for g in group_cf}
        if any(s is None for s in scores.values()):
            # no query term appears in the collection, so fall back to equal shares
            rows += [(qid, g, 1 / len(scores)) for g in scores]
            continue
 
        # 2^(-SCS) normalised across groups, shifted by the smallest SCS to avoid underflow
        best = min(scores.values())
        weights = {g: 2 ** (best - s) for g, s in scores.items()}
        total = sum(weights.values())
        rows += [(qid, g, w / total) for g, w in weights.items()]
 
    return pd.DataFrame(rows, columns=["qid", "group", "prediction"])