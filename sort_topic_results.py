import sys
from pathlib import Path

import pandas as pd

path = Path(sys.argv[1])
column = sys.argv[2] if len(sys.argv) > 2 else "jsd"

results = pd.read_csv(path)
if column not in results.columns:
    sys.exit(f"no column '{column}' in {path}; columns are: {', '.join(results.columns)}")

out = path.with_name(f"{path.stem}_sorted_{column}.csv")
results.sort_values(column).to_csv(out, index=False)
print(f"wrote {out} (sorted by {column})")