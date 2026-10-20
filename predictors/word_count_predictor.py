import pandas as pd


def word_count_predictor(topics):
    return pd.DataFrame({
        "qid": topics["qid"],
        "prediction": topics["query"].str.split().str.len(),
    })