import math
from collections import Counter
import numpy as np

 
def idf(n, df):
    return math.log((n - df + 0.5) / (df + 0.5))
 
 
def predict_exposure(query, docs, doc_groups, k=100):
    # tokenise the query and every document
    q_tf = Counter(query.lower().split())
    terms = list(q_tf)
    doc_tf = {d: Counter(text.lower().split()) for d, text in docs.items()}
    groups = sorted({g for gs in doc_groups.values() for g in gs})
 
    # weight query terms with tf idf over the whole collection
    n = len(docs)
    Q = np.array([q_tf[t] * idf(n, sum(t in tf for tf in doc_tf.values())) for t in terms])
 
    e = []
    for g in groups:
        group_docs = [doc_tf[d] for d in docs if g in doc_groups.get(d, ())]
 
        # score each query term by the mean of the k largest tf idf values in the group
        S = []
        for t in terms:
            tfs = [tf[t] for tf in group_docs if tf[t] > 0]
            scores = [c * idf(len(group_docs), len(tfs)) for c in tfs] + [0] * k
            S.append(np.mean(sorted(scores, reverse=True)[:k]))
 
        # compare the group vector with the query vector
        e.append(np.dot(S, Q))
 
    # normalise the group scores into exposure shares
    e = np.clip(e, 0, None)
    return dict(zip(groups, e / e.sum()))
