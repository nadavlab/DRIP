#!/usr/bin/env python3

import pandas as pd
from pathlib import Path

PHENO_DIR = Path("/path/to/your/project//simulations/simulated_phenotypes_DRIP")
OUT_FILE = PHENO_DIR / "all_simulated_phenotypes_for_GWAS.txt"

pheno_files = sorted(PHENO_DIR.glob("rep*_chr1_N100_h2_*_cov_*_noise_*.csv"))

merged = None
pheno_names = []

for f in pheno_files:
    df = pd.read_csv(f)

    pheno_name = f.stem.replace(".", "p").replace("-", "m")
    pheno_names.append(pheno_name)

    df = df[["FID", "IID", "phenotype"]].copy()
    df = df.rename(columns={"phenotype": pheno_name})

    df["FID"] = df["FID"].astype(str)
    df["IID"] = df["IID"].astype(str)

    if merged is None:
        merged = df
    else:
        merged = merged.merge(df, on=["FID", "IID"], how="outer")

merged.to_csv(OUT_FILE, sep="\t", index=False)

names_file = PHENO_DIR / "all_simulated_phenotype_names.txt"
with open(names_file, "w") as out:
    out.write(",".join(pheno_names))

print(f"Saved phenotype file: {OUT_FILE}")
print(f"Saved phenotype names: {names_file}")
print(f"Number of phenotypes: {len(pheno_names)}")
print(merged.shape)