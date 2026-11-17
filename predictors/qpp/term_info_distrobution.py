import math

import numpy as np
import pandas as pd

def idf(n, nt):
    """
    #### n -> num docs
    #### nt -> num docs where query term appears
    """
    return math.log2((n + 0.5) / nt) / math.log2(n + 1)


def dist_informative_amount1(topics, run_dir) -> pd.DataFrame:
    """
    Hypothesis: queries whose terms vary more in informativeness are more performant,
    measured as the ratio of the largest to the smallest query term idf
    """
    # document frequencies of the stemmed query terms over the whole collection
    stats = pd.read_parquet(run_dir / "term_stats.parquet", columns=["term", "df", "n_docs"])
    n = stats["n_docs"].iloc[0]
    df = stats.groupby("term")["df"].first()

    def get_max_and_min_idf(terms):
        max_idf = 0
        min_idf = math.inf

        for term in terms:
            # skip terms that do not appear in the collection
            if df.get(term, 0) == 0:
                continue
            term_idf = idf(n, df[term])
            max_idf = max(max_idf, term_idf)
            min_idf = min(min_idf, term_idf)

        return max_idf, min_idf

    def score(terms):
        max_idf, min_idf = get_max_and_min_idf(terms)
        if min_idf == math.inf:
            return 1.0
        return max_idf / min_idf

    return pd.DataFrame({
        "qid": topics["qid"],
        "prediction": topics["gep_terms"].apply(score),
    })


def dist_informative_amount2(topics, run_dir) -> pd.DataFrame:
    """
    Hypothesis: queries whose terms vary more in informativeness are more performant,
    measured as the standard deviation of the query term idfs
    """
    # document frequencies of the stemmed query terms over the whole collection
    stats = pd.read_parquet(run_dir / "term_stats.parquet", columns=["term", "df", "n_docs"])
    n = stats["n_docs"].iloc[0]
    df = stats.groupby("term")["df"].first()

    def score(terms):
        # skip terms that do not appear in the collection
        idfs = [idf(n, df[term]) for term in terms if df.get(term, 0) > 0]
        if not idfs:
            return 0.0
        # population standard deviation, so a single term query scores 0
        return float(np.std(idfs))

    return pd.DataFrame({
        "qid": topics["qid"],
        "prediction": topics["gep_terms"].apply(score),
    })
