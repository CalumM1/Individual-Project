import math
import numpy as np
from scipy.spatial.distance import jensenshannon


def evaluate(predicted, ranking, doc_groups, k=100):
    groups = sorted(predicted)

    # sum the exposure each group gets from its positions in the top k
    actual = dict.fromkeys(groups, 0.0)
    for p, d in enumerate(ranking[:k], 1):
        for g in doc_groups.get(d, ()):
            actual[g] += 1 / math.log2(p + 1)

    # normalise the actual exposure into shares
    total = sum(actual.values())
    actual = {g: v / total for g, v in actual.items()}

    # measure the jensen shannon distance between predicted and actual shares
    jsd = jensenshannon([predicted[g] for g in groups], [actual[g] for g in groups])
    return jsd, actual