import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent))
import evaluate_qpp


RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"

if len(sys.argv) != 2:
    raise ValueError("Please provide the name of training query directory")

TRAINING_QUERIES = sys.argv[1]

# trained on this one retriever and metric
TARGET_SYSTEM = "bm25"
TARGET_METRIC = "nDCG@10"


def count_syllables(word: str) -> int:
    """rough count based on vowels"""
    word = word.lower()
    n = len(re.findall(r"[aeiouy]+", word))
    if n > 1 and word.endswith("e") and not word.endswith(("le", "ee")):
        n -= 1
    return max(n, 1)


def extract_features(topics: pd.DataFrame) -> pd.DataFrame:
    words = topics["query"].str.split()
    return pd.DataFrame({
        "terms": words.str.len(),
        "syllables": words.apply(lambda ws: sum(count_syllables(w) for w in ws)),
    }, index=topics.index)


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    run_dir = RUNS_DIR / TRAINING_QUERIES
    train_topics = pd.read_parquet(run_dir / "topics.parquet")
    perq = pd.read_parquet(run_dir / "perquery_metrics.parquet")
    perq = perq[perq["name"] == TARGET_SYSTEM]

    y = perq.set_index("qid")[TARGET_METRIC].reindex(train_topics["qid"])
    if y.isna().any():
        raise ValueError(f"{y.isna().sum()} training queries have no {TARGET_METRIC} score")
    return train_topics, y


def syllable_term_predictor(topics: pd.DataFrame) -> pd.DataFrame:
    """
    Hypothesis: query performance can be predicted from a learned weighted
    combination of the number of terms and the number of syllables
    """
    train_topics, y_train = load_training_data()

    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(extract_features(train_topics), y_train.to_numpy())

    reg = model[-1]
    print(
        f"[syllable_term] trained on {TRAINING_QUERIES} ({len(train_topics)} queries), "
        f"target {TARGET_SYSTEM} {TARGET_METRIC}: "
        f"terms weight={reg.coef_[0]:.4f}, syllables weight={reg.coef_[1]:.4f}, "
        f"intercept={reg.intercept_:.4f} (standardised features)"
    )


    return pd.DataFrame({
        "qid": topics["qid"].values,
        "prediction": model.predict(extract_features(topics)),
    })


if __name__ == "__main__":
    evaluate_qpp.main(syllable_term_predictor, "qpp_syllable_term_results")