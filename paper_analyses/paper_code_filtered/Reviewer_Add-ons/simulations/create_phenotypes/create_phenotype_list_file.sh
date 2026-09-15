python - <<'PY'
import pandas as pd

y_file = "/path/to/your/project//simulations/matched_data_ids/Y_train_all_rep_1.pkl"
out_file = "/path/to/your/project//simulations/LR_results/phenotypes.txt"

with open(y_file, "rb") as f:
    y = pd.read_pickle(f)

cols = [c for c in y.columns if c not in {"FID", "IID", "#FID"}]

with open(out_file, "w") as f:
    for c in cols:
        f.write(c + "\n")

print(f"Wrote {len(cols)} phenotypes to {out_file}")
PY