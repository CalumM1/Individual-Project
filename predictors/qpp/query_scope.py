import math

import pandas as pd


def query_scope(topics, run_dir) -> pd.DataFrame:
    """
    Hypothesis: queries whose terms match fewer documents are more specific and so more
    performant
    """
    # documents matching each query (union of its terms' posting lists) and the collection size
    stats = pd.read_parquet(run_dir / "query_stats.parquet", columns=["qid", "n_matched", "n_docs"])
    n = stats["n_docs"].iloc[0]
    n_matched = stats.set_index("qid")["n_matched"]

    def score(qid):
        matched = n_matched.get(qid, 0)
        # no query term appears in the collection, so the scope is undefined
        if matched == 0:
            return 0.0
        return -math.log(matched / n)

    return pd.DataFrame({
        "qid": topics["qid"],
        "prediction": topics["qid"].map(score),
    })