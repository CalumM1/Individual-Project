import math
 
import pandas as pd
 
 
def load(run_dir, category):
    # documents in each group matching each query 
    # and the size of each group
    stats = pd.read_parquet(
        run_dir / "query_group_stats.parquet",
        columns=["qid", "category", "group", "n_matched", "group_n_docs"],
    )
    return stats[stats["category"] == category]
 
 
def shares(stats, score):
    # normalise each query's group scores into exposure shares, falling back to equal shares
    # when every group scores 0
    rows = []
    for qid, q in stats.groupby("qid", sort=False):
        scores = {g: score(m, n) for g, m, n in zip(q["group"], q["n_matched"], q["group_n_docs"])}
        total = sum(scores.values())
        rows += [(qid, g, s / total if total > 0 else 1 / len(scores)) for g, s in scores.items()]
    return pd.DataFrame(rows, columns=["qid", "group", "prediction"])
 
 
def query_scope(run_dir, category) -> pd.DataFrame:
    """
    Hypothesis: a group receives more exposure when the query's terms match a smaller part of
    that group's documents
    """
    def score(n_matched, group_n_docs):
        # no query term appears in the group, so it is predicted no exposure
        if n_matched == 0:
            return 0.0
        return -math.log(n_matched / group_n_docs)
 
    return shares(load(run_dir, category), score)


def query_scope_inverse(run_dir, category) -> pd.DataFrame:
    """
    Hypothesis: a group receives more exposure when the query's terms match a larger part of
    that group's documents
    """
    def score(n_matched, group_n_docs):
        return n_matched / group_n_docs if group_n_docs > 0 else 0.0
 
    return shares(load(run_dir, category), score)