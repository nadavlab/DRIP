#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
from pathlib import Path

# --------------------
# Parameters
# --------------------
RUNS = [1, 2, 3, 4]
N_SNPS = 100

H2_VALUES = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

VAR_E_BASE = 0.5
H2_COV_BASE = 0.2
PREVALENCE = 0.1


RAW_BASE_DIR = Path("/path/to/your/project//simulations/random_SNPs_samples")
COV_FILE = Path("/path/to/your/project//UKBB/phenotypes/covariates_age_sex_bmi.txt")

OUT_DIR = Path("/path/to/your/project//simulations/simulated_phenotypes_DRIP/heritability")
OUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# --------------------
# Load allowed samples
# --------------------
# --------------------
# Load Europeansf
# --------------------
ethnicity_file = "/path/to/your/project//UKBB/phenotypes/Ethnicity.pheno"

ethnicity = pd.read_csv(ethnicity_file, sep=r"\s+")

ethnicity = ethnicity[
    ethnicity["genetic_ethnic_grouping_f22006_0_0"] == "Caucasian"].copy()

ethnicity["FID"] = ethnicity["FID"].astype(str)
ethnicity["IID"] = ethnicity["IID"].astype(str)

allowed_iids = set(ethnicity["IID"])

# --------------------
# Load covariates
# --------------------
cov = pd.read_csv(COV_FILE, sep=None, engine="python")
cov = pd.read_csv(COV_FILE, sep=r"\s+")

print("columns:", cov.columns.tolist())
print(cov.head(3))

print("IID head:", cov["IID"].head(10).tolist())
print("FID head:", cov["FID"].head(10).tolist())

cov_iids = set(cov["IID"].astype(str))

print("example cov IID:", list(cov_iids)[:10])
print("min/max IID:", cov["IID"].min(), cov["IID"].max())
print("intersection:", len(allowed_iids & cov_iids))

print(cov.head(3))

# ????? ????? ???????
if "Participant ID" in cov.columns:
    id_col = "Participant ID"
elif "IID" in cov.columns:
    id_col = "IID"
else:
    raise ValueError("Could not find ID column. Expected 'Participant ID' or 'IID'.")

cov[id_col] = cov[id_col].astype(str)
cov_iids = set(cov.index.astype(str))

cov = cov[cov[id_col].isin(allowed_iids)].copy()
print(cov)
cov = cov.set_index(id_col)
print(cov.shape)

print("n allowed:", len(allowed_iids))
print("n cov:", len(cov_iids))
print("intersection:", len(allowed_iids & cov_iids))
print("example allowed:", list(allowed_iids)[:5])
print("example cov:", list(cov_iids)[:5])

cov_cols = ["age", "bmi", "sex"]

# ?? sex ????? ???? ????/????
if cov["sex"].dtype == "object":
    cov["sex"] = cov["sex"].map({
        "Female": 0,
        "Male": 1,
        "female": 0,
        "male": 1
    })

cov = cov.replace([np.inf, -np.inf], np.nan)

for c in cov_cols:
    cov[c] = pd.to_numeric(cov[c], errors="coerce")
    cov[c] = cov[c].fillna(cov[c].mean())
    cov[c] = (cov[c] - cov[c].mean()) / cov[c].std()

gamma = np.array([0.4, 0.3, 0.3])

# --------------------
# Main loop
# --------------------
# --------------------
# Main loop
# --------------------
for run in RUNS:
    print(f"\n=== Rep {run} ===")

    raw_file = RAW_BASE_DIR / f"rep_{run}_chro1_N_{N_SNPS}.raw"

    if not raw_file.exists():
        print(f"Skipping missing file: {raw_file}")
        continue

    print(f"Reading: {raw_file}")

    raw = pd.read_csv(raw_file, sep=r"\s+")
    print(raw.head(3))

    raw["IID"] = raw["IID"].astype(str)
    raw["FID"] = raw["FID"].astype(str)

    raw = raw[raw["IID"].isin(cov.index)].copy()

    if raw.empty:
        print(f"No overlapping individuals for rep={run}, N={N_SNPS}")
        continue

    snp_cols = list(raw.columns[6:])

    G = raw[snp_cols].copy()
    G = G.apply(pd.to_numeric, errors="coerce")
    G = G.fillna(G.mean())

    X = cov.loc[raw["IID"], cov_cols].values

    # Fixed genetic effects per rep
    beta = np.random.normal(0, 1, len(snp_cols))
    g = G.values @ beta

    c = X @ gamma

    var_g = np.var(g)
    var_c = np.var(c)

    if var_g == 0 or var_c == 0:
        print(f"Skipping rep={run}: zero variance")
        continue
    for H2_GENETIC in H2_VALUES:

        remaining = 1 - H2_GENETIC

        H2_COV_CURRENT = remaining * (H2_COV_BASE / (H2_COV_BASE + VAR_E_BASE))
        VAR_E_CURRENT = remaining * (VAR_E_BASE / (H2_COV_BASE + VAR_E_BASE))

        scale_g = np.sqrt(H2_GENETIC / var_g) if H2_GENETIC > 0 else 0
        scale_c = np.sqrt(H2_COV_CURRENT / var_c) if H2_COV_CURRENT > 0 else 0

        liability = (
            scale_g * g +
            scale_c * c +
            np.random.normal(0, np.sqrt(VAR_E_CURRENT), len(raw))
        )

        threshold = np.quantile(liability, 1 - PREVALENCE)
        phenotype = np.where(liability > threshold, 2, 1)

        observed_prev = np.mean(phenotype == 2)
        out_file = OUT_DIR / (
              f"rep{run}_chr1_N{N_SNPS}"
              f"_h2_{H2_GENETIC}"
              f"_cov_{H2_COV_CURRENT:.3f}"
              f"_noise_{VAR_E_CURRENT:.3f}.csv"
              )
        
        pd.DataFrame({
            "FID": raw["FID"].values,
            "IID": raw["IID"].values,
            "phenotype": phenotype
        }).to_csv(out_file, index=False)

        print(f"Saved: {out_file}")
        print(
            f"h2={H2_GENETIC}, cov={H2_COV_CURRENT}, "
            f"noise={VAR_E_CURRENT:.2f}, prevalence={observed_prev:.4f}"
        )