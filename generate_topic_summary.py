import sys
from pathlib import Path

import pandas as pd

path = Path(sys.argv[1])
out = path.with_name(f"{path.stem}_per_topic.csv")

results = pd.read_csv(path)
summary = (
    results.groupby(["dataset", "topic_id", "topic_title"])["jsd"]
    .agg(mean_jsd="mean", median_jsd="median", n_rows="size")
    .reset_index()
    .sort_values(["dataset", "mean_jsd"])
)
summary.to_csv(out, index=False)
print(f"wrote {out}")