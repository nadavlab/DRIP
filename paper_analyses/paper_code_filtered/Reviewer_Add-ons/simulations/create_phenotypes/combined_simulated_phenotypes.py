#!/usr/bin/env python3

import pandas as pd
from pathlib import Path

# --------------------
# Paths
# --------------------
PHENO_DIRS = [
    Path("/path/to/your/project//simulations/simulated_phenotypes_DRIP"),
    Path("/path/to/your/project//simulations/simulated_phenotypes_DRIP_heritability"),
    Path("/path/to/your/project//simulations/simulated_phenotypes_DRIP_GxG"),
]

Y_BASE_DIR = Path(
    "/path/to/your/project//our_model/PCA/Y_files"
)

OUT_DIR = Path(
    "/path/to/your/project//simulations/combined_DRIP_phenotypes_by_rep"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = [1, 2, 3, 4, 5]


# --------------------
# Helper functions
# --------------------
def clean_pheno_name(name):
    return (
        name.replace(".", "p")
            .replace("-", "m")
            .replace(" ", "_")
            .replace("=", "_")
            .replace(",", "_")
    )


def read_id_file(path):
    """
    Reads CSV/TXT/PKL files and returns a dataframe with FID/IID.
    Handles pandas DataFrame, Series, numpy arrays, and lists.
    """
    path = Path(path)

    if path.suffix == ".pkl":
        obj = pd.read_pickle(path)

        if isinstance(obj, pd.DataFrame):
            df = obj.copy()
        elif isinstance(obj, pd.Series):
            df = obj.reset_index()
        else:
            df = pd.DataFrame(obj)

    else:
        df = pd.read_csv(path, sep=None, engine="python")

    # Standard case
    if {"FID", "IID"}.issubset(df.columns):
        out = df[["FID", "IID"]].copy()

    elif "IID" in df.columns:
        out = pd.DataFrame({
            "FID": df["IID"],
            "IID": df["IID"]
        })

    else:
        # fallback: use index if meaningful
        if df.index.name in ["IID", "FID"]:
            ids = df.index
        else:
            ids = df.iloc[:, 0]

        out = pd.DataFrame({
            "FID": ids,
            "IID": ids
        })

    out["FID"] = out["FID"].astype(str)
    out["IID"] = out["IID"].astype(str)

    out = out.drop_duplicates(subset=["IID"])

    return out


def merge_phenotypes_for_rep(rep):
    rep_prefix = f"rep{rep}_"
    pheno_files = []

    for d in PHENO_DIRS:
        pheno_files.extend(sorted(d.glob(f"{rep_prefix}*.csv")))

    print(f"\nRep {rep}: found {len(pheno_files)} phenotype files")

    if len(pheno_files) == 0:
        raise ValueError(f"No phenotype files found for rep {rep}")

    merged = None
    used_names = set()

    for f in pheno_files:
        df = pd.read_csv(f)

        if not {"FID", "IID", "phenotype"}.issubset(df.columns):
            print(f"Skipping file with missing columns: {f}")
            continue

        pheno_name = clean_pheno_name(f.stem)

        original_name = pheno_name
        counter = 1
        while pheno_name in used_names:
            pheno_name = f"{original_name}_dup{counter}"
            counter += 1

        used_names.add(pheno_name)

        df = df[["FID", "IID", "phenotype"]].copy()
        df["FID"] = df["FID"].astype(str)
        df["IID"] = df["IID"].astype(str)

        df = df.rename(columns={"phenotype": pheno_name})

        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=["FID", "IID"], how="outer")

    if merged is None:
        raise ValueError(f"No valid phenotype files found for rep {rep}")

    return merged


# --------------------
# Main
# --------------------
summary_rows = []

for rep in RUNS:
    print(f"\n================ Rep {rep} ================")

    all_pheno = merge_phenotypes_for_rep(rep)

    test_id_file = (
        Y_BASE_DIR
        / f"rep{rep}"
        / "Hypertension_Y_test_1k_chunks_no_missing.pkl"
    )

    if not test_id_file.exists():
        raise FileNotFoundError(f"Missing test ID file: {test_id_file}")

    test_ids = read_id_file(test_id_file)
    test_iids = set(test_ids["IID"])

    all_pheno["FID"] = all_pheno["FID"].astype(str)
    all_pheno["IID"] = all_pheno["IID"].astype(str)

    pheno_iids = set(all_pheno["IID"])

    missing_test_iids = test_iids - pheno_iids
    test_iids_existing = test_iids & pheno_iids
    train_iids = pheno_iids - test_iids

    print(f"Phenotype individuals: {len(pheno_iids)}")
    print(f"Test IDs in test file: {len(test_iids)}")
    print(f"Test IDs found in phenotype file: {len(test_iids_existing)}")
    print(f"Test IDs missing from phenotype file: {len(missing_test_iids)}")
    print(f"Train IDs inferred as all non-test phenotype IDs: {len(train_iids)}")

    if len(missing_test_iids) > 0:
        print("Examples missing test IDs:", list(missing_test_iids)[:10])

    train_df = all_pheno[all_pheno["IID"].isin(train_iids)].copy()
    test_df = all_pheno[all_pheno["IID"].isin(test_iids_existing)].copy()


    # optional: order test file according to original test IDs
    test_order = test_ids[test_ids["IID"].isin(test_iids_existing)].copy()
    test_df = test_order.merge(
    test_df.drop(columns=["FID"], errors="ignore"),
    on="IID",
    how="inner")
    
    # ensure FID exists and is first column
    if "FID" not in test_df.columns:
        test_df.insert(0, "FID", test_df["IID"])
    else:
        cols = ["FID", "IID"] + [c for c in test_df.columns if c not in ["FID", "IID"]]
        test_df = test_df[cols]

    out_train = OUT_DIR / f"rep{rep}_train_phenotypes.txt"
    out_test = OUT_DIR / f"rep{rep}_test_phenotypes.txt"

    train_df.to_csv(out_train, sep="\t", index=False)
    test_df.to_csv(out_test, sep="\t", index=False)

    print(f"Saved train: {out_train}")
    print(f"Saved test:  {out_test}")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape:  {test_df.shape}")

    summary_rows.append({
        "rep": rep,
        "n_pheno_individuals": len(pheno_iids),
        "n_test_ids_in_file": len(test_iids),
        "n_test_ids_found": len(test_iids_existing),
        "n_test_ids_missing_from_phenotypes": len(missing_test_iids),
        "n_train_output": train_df.shape[0],
        "n_test_output": test_df.shape[0],
        "n_phenotypes": train_df.shape[1] - 2,
    })


summary = pd.DataFrame(summary_rows)
summary_file = OUT_DIR / "split_summary.tsv"
summary.to_csv(summary_file, sep="\t", index=False)

print("\nDone.")
print(summary)
print(f"Saved summary: {summary_file}")