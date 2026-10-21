import pandas as pd

def word_count_predictor(topics) -> pd.DataFrame:
    """
    Hypothesis: queries with more terms are more performant
    """
    return pd.DataFrame({
        "qid": topics["qid"],
        "prediction": topics["query"].str.split().str.len(),
    })