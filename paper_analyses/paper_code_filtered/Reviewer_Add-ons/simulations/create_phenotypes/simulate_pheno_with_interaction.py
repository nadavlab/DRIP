
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path

# --------------------
# Parameters
# --------------------
RUNS = [1, 2, 3, 4, 5]
N_SNPS = 100

TOTAL_GENETIC = 0.3
H2_COV_BASE = 0.2
VAR_E_BASE = 0.5

PREVALENCE = 0.1
N_INTERACTION_PAIRS = 25

# additive / interaction split
GI_SCENARIOS = {
    #"GI_00": {"h2_add": 0.30, "h2_gxg": 0.00},
    #"GI_05": {"h2_add": 0.25, "h2_gxg": 0.05},
    #"GI_10": {"h2_add": 0.20, "h2_gxg": 0.10},
    #"GI_15": {"h2_add": 0.15, "h2_gxg": 0.15},
    #"GI_20": {"h2_add": 0.10, "h2_gxg": 0.20},
    "GI_25": {"h2_add": 0.05, "h2_gxg": 0.30},
    "GI_30": {"h2_add": 0.00, "h2_gxg": 0.30},

}

RAW_BASE_DIR = Path(
    "/path/to/your/project//simulations/random_SNPs_samples"
)

COV_FILE = Path(
    "/path/to/your/project//UKBB/phenotypes/covariates_age_sex_bmi.txt"
)

ETHNICITY_FILE = Path(
    "/path/to/your/project//UKBB/phenotypes/Ethnicity.pheno"
)

OUT_DIR = Path(
    "/path/to/your/project//simulations/simulated_phenotypes_DRIP_GxG"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# --------------------
# Load Europeans
# --------------------
ethnicity = pd.read_csv(ETHNICITY_FILE, sep=r"\s+")

ethnicity = ethnicity[
    ethnicity["genetic_ethnic_grouping_f22006_0_0"] == "Caucasian"
].copy()

ethnicity["IID"] = ethnicity["IID"].astype(str)
allowed_iids = set(ethnicity["IID"])

print(f"European individuals: {len(allowed_iids)}")

# --------------------
# Load covariates
# --------------------
cov = pd.read_csv(COV_FILE, sep=r"\s+")

if "IID" in cov.columns:
    id_col = "IID"
elif "Participant ID" in cov.columns:
    id_col = "Participant ID"
else:
    raise ValueError("Could not find IID or Participant ID column in covariates file.")

cov[id_col] = cov[id_col].astype(str)

cov = cov[cov[id_col].isin(allowed_iids)].copy()
cov = cov.set_index(id_col)

cov_cols = ["age", "bmi", "sex"]

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

print(f"Covariates after European filter: {cov.shape}")

# --------------------
# Main loop
# --------------------
for run in RUNS:
    print(f"\n=== Rep {run} ===")

    raw_file = RAW_BASE_DIR / f"rep_{run}_chro1_N_{N_SNPS}.raw"

    if not raw_file.exists():
        print(f"Skipping missing file: {raw_file}")
        continue

    raw = pd.read_csv(raw_file, sep=r"\s+")

    raw["IID"] = raw["IID"].astype(str)
    raw["FID"] = raw["FID"].astype(str)

    raw = raw[raw["IID"].isin(cov.index)].copy()

    if raw.empty:
        print(f"No overlapping individuals for rep={run}")
        continue

    snp_cols = list(raw.columns[6:])
    print(f"Individuals: {len(raw)}, SNPs: {len(snp_cols)}")

    G = raw[snp_cols].copy()
    G = G.apply(pd.to_numeric, errors="coerce")
    G = G.fillna(G.mean())
    G = G.values

    X = cov.loc[raw["IID"], cov_cols].values

    # --------------------
    # Additive genetic component
    # --------------------
    beta_add = np.random.normal(0, 1, G.shape[1])
    g_add = G @ beta_add
    var_add = np.var(g_add)

    if var_add == 0:
        print(f"Skipping rep={run}: zero additive genetic variance")
        continue

    # --------------------
    # GxG interaction component
    # --------------------
    rng = np.random.default_rng(seed=10_000 + run)

    all_snp_indices = np.arange(G.shape[1])

    interaction_pairs = []
    used_pairs = set()

    while len(interaction_pairs) < N_INTERACTION_PAIRS:
        i, j = rng.choice(all_snp_indices, size=2, replace=False)
        pair = tuple(sorted((int(i), int(j))))

        if pair not in used_pairs:
            used_pairs.add(pair)
            interaction_pairs.append(pair)

    GxG = np.column_stack([
        G[:, i] * G[:, j]
        for i, j in interaction_pairs
    ])

    beta_gxg = rng.normal(0, 1, N_INTERACTION_PAIRS)
    g_gxg = GxG @ beta_gxg
    var_gxg = np.var(g_gxg)

    if var_gxg == 0:
        print(f"Skipping rep={run}: zero GxG variance")
        continue

    # --------------------
    # Covariate component
    # --------------------
    c = X @ gamma
    var_c = np.var(c)

    if var_c == 0:
        print(f"Skipping rep={run}: zero covariate variance")
        continue

    # Save interaction pairs for reproducibility
    pairs_file = OUT_DIR / f"rep{run}_chr1_N{N_SNPS}_GxG_pairs.tsv"

    pd.DataFrame({
        "pair_id": np.arange(1, N_INTERACTION_PAIRS + 1),
        "snp1_index": [p[0] for p in interaction_pairs],
        "snp2_index": [p[1] for p in interaction_pairs],
        "snp1": [snp_cols[p[0]] for p in interaction_pairs],
        "snp2": [snp_cols[p[1]] for p in interaction_pairs],
        "beta_gxg": beta_gxg
    }).to_csv(pairs_file, sep="\t", index=False)

    print(f"Saved interaction pairs: {pairs_file}")

    # --------------------
    # Simulate phenotypes
    # --------------------
    for scenario_name, params in GI_SCENARIOS.items():

        h2_add = params["h2_add"]
        h2_gxg = params["h2_gxg"]

        remaining = 1 - h2_add - h2_gxg

        h2_cov_current = remaining * (
            H2_COV_BASE / (H2_COV_BASE + VAR_E_BASE)
        )

        var_e_current = remaining * (
            VAR_E_BASE / (H2_COV_BASE + VAR_E_BASE)
        )

        scale_add = np.sqrt(h2_add / var_add) if h2_add > 0 else 0
        scale_gxg = np.sqrt(h2_gxg / var_gxg) if h2_gxg > 0 else 0
        scale_cov = np.sqrt(h2_cov_current / var_c) if h2_cov_current > 0 else 0

        liability = (
            scale_add * g_add
            + scale_gxg * g_gxg
            + scale_cov * c
            + rng.normal(0, np.sqrt(var_e_current), len(raw))
        )

        threshold = np.quantile(liability, 1 - PREVALENCE)
        phenotype = np.where(liability > threshold, 2, 1)

        observed_prev = np.mean(phenotype == 2)

        out_file = OUT_DIR / (
            f"rep{run}_chr1_N{N_SNPS}"
            f"_{scenario_name}"
            f"_h2add_{h2_add:.2f}"
            f"_h2gxg_{h2_gxg:.2f}"
            f"_cov_{h2_cov_current:.3f}"
            f"_noise_{var_e_current:.3f}.csv"
        )

        pd.DataFrame({
            "FID": raw["FID"].values,
            "IID": raw["IID"].values,
            "phenotype": phenotype
        }).to_csv(out_file, index=False)

        print(
            f"Saved {scenario_name}: "
            f"h2_add={h2_add:.2f}, "
            f"h2_gxg={h2_gxg:.2f}, "
            f"cov={h2_cov_current:.3f}, "
            f"noise={var_e_current:.3f}, "
            f"prev={observed_prev:.4f}"
        )