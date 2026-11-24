import sys
from pathlib import Path

import pandas as pd

# folder holding main_results_<year>.csv; defaults to the evaluate_qep_topics output folder
results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/qep_topics")

for path in sorted(results_dir.glob("main_results_*.csv")):
    year = path.stem.removeprefix("main_results_")
    summary = (
        pd.read_csv(path)
        .groupby(["dataset", "topic_id", "topic_title"])["jsd"]
        .agg(mean_jsd="mean", median_jsd="median", n_rows="size")
        .reset_index()
        .sort_values("mean_jsd")
    )
    out = results_dir / f"aggregated_results_{year}.csv"
    summary.to_csv(out, index=False)
    print(f"wrote {out}")