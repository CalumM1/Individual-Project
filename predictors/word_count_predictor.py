import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import evaluate_qpp

def word_count_predictor(topics) -> pd.DataFrame:
    """
    Hypothesis: queries with more terms are more performant
    """
    return pd.DataFrame({
        "qid": topics["qid"],
        "prediction": topics["query"].str.split().str.len(),
    })
    
    
if __name__ == "__main__":
    evaluate_qpp.main(word_count_predictor, "qpp_query_length_results")