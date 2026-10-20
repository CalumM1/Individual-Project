import pandas as pd


def query_length_predictor(topics):
    return pd.DataFrame({
        "qid": topics["qid"],
        "prediction": topics["query"].str.split().str.len(),
    })