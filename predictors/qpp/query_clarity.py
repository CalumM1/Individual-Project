import math
from collections import Counter

import pandas as pd


def query_clarity(topics, run_dir) -> pd.DataFrame:
    """
    Hypothesis: queries whose language departs more from the collection's are
    less ambiguous and so more performant
    """
    # collection frequencies of the stemmed query terms and the collection size in tokens
    stats = pd.read_parquet(run_dir / "term_stats.parquet", columns=["term", "cf", "n_tokens"])
    n_tokens = stats["n_tokens"].iloc[0]
    cf = stats.groupby("term")["cf"].first()

    def score(terms):
        # ql is the number of (non-stop) query terms, qtf the count of each one
        ql = len(terms)
        qtf = Counter(terms)

        clarity = 0.0
        matched = False
        for term, count in qtf.items():
            # skip terms that do not appear in the collection
            if cf.get(term, 0) == 0:
                continue
            p_q = count / ql
            p_coll = cf[term] / n_tokens
            clarity += p_q * math.log2(p_q / p_coll)
            matched = True

        if not matched:
            return 0.0
        return clarity

    return pd.DataFrame({
        "qid": topics["qid"],
        "prediction": topics["gep_terms"].apply(score),
    })

