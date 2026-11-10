import math
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
import evaluate_qep


def idf(n, df):
    return math.log((n - df + 0.5) / (df + 0.5))


def gep_predictor(run_dir, category, k=100) -> pd.DataFrame:
    # load the query terms term statistics and group sizes
    topics = pd.read_parquet(run_dir / "topics.parquet")
    stats = pd.read_parquet(run_dir / "term_stats.parquet")
    stats = stats[stats["category"] == category]
    sizes = pd.read_parquet(run_dir / "groups.parquet", columns=[category])[category].explode().value_counts()
    groups = sorted(sizes.index)
    n = stats["n_docs"].iloc[0]
    df = stats.groupby("term")["df"].first()
    lookup = {(r.term, r.group): r for r in stats.itertuples()}

    rows = []
    for qid, query_terms in zip(topics["qid"], topics["gep_terms"]):
        q_tf = Counter(query_terms)
        terms = list(q_tf)

        # weight query terms with tf idf over the whole collection
        Q = np.array([q_tf[t] * idf(n, df.get(t, 0)) for t in terms])

        e = []
        for g in groups:
            # score each query term by the mean of the k largest tf idf values in the group
            S = []
            for t in terms:
                r = lookup.get((t, g))
                if r is None or r.group_df == 0:
                    S.append(0.0)
                    continue
                w = idf(sizes[g], r.group_df)
                scores = [tf * w for tf in r.top_tfs] + [0] * min(k, sizes[g] - r.group_df)
                S.append(np.mean(sorted(scores, reverse=True)[:k]))

            # compare the group vector with the query vector
            e.append(np.dot(S, Q))

        # normalise the group scores into exposure shares
        e = np.clip(e, 0, None)
        e = e / e.sum() if e.sum() > 0 else np.full(len(e), 1 / len(e))
        rows += [(qid, g, p) for g, p in zip(groups, e)]

    return pd.DataFrame(rows, columns=["qid", "group", "prediction"])


if __name__ == "__main__":
    evaluate_qep.main(gep_predictor, "gep_results")
