#!/bin/bash

###############################################################################
# Run PRS-CS for GI_30 simulations, repetitions 1-5
# Chromosome 1 only
#
# Usage:
#   bash run_PRScs_GI30_all_reps.sh
#
# Run selected repetitions:
#   bash run_PRScs_GI30_all_reps.sh 1 3 5
###############################################################################

set -euo pipefail

BASE="/path/to/your/project/"

PRSCS_SCRIPT="${BASE}/PRScs/PRScs/PRScs.py"
REF_DIR="${BASE}/PRScs/ldblk_ukbb_eur"

GWAS_BASE="${BASE}/data/simulations/GWAS_results_DRIP_train"
BIM_BASE="${BASE}/data/training_bed_files"

SST_BASE="${BASE}/data/simulations/PRScs/SS_formatted_data"
PRSCS_OUT_BASE="${BASE}/data/simulations/PRScs/PRScs_results"
LOG_BASE="${BASE}/data/simulations/PRScs/slurm_logs"

MEMORY="32G"
TIME_LIMIT="1-00:00:00"
CPUS="1"

PHENOTYPE="chr1_N100_GI_30_h2add_0p00_h2gxg_0p30_cov_0p200_noise_0p500"

###############################################################################
# Repetitions
###############################################################################

if [[ $# -gt 0 ]]; then
    REPS=("$@")
else
    REPS=(1 2 3 4 5)
fi

###############################################################################
# Check shared inputs
###############################################################################

if [[ ! -f "${PRSCS_SCRIPT}" ]]; then
    echo "ERROR: PRS-CS script not found:"
    echo "${PRSCS_SCRIPT}"
    exit 1
fi

if [[ ! -d "${REF_DIR}" ]]; then
    echo "ERROR: PRS-CS reference directory not found:"
    echo "${REF_DIR}"
    exit 1
fi

mkdir -p "${SST_BASE}" "${PRSCS_OUT_BASE}" "${LOG_BASE}"

###############################################################################
# Process repetitions
###############################################################################

for rep in "${REPS[@]}"
do
    echo
    echo "============================================================"
    echo "Processing rep${rep}"
    echo "Phenotype: ${PHENOTYPE}"
    echo "============================================================"

    PHENO_DIR="${GWAS_BASE}/rep${rep}/rep${rep}_${PHENOTYPE}"

    GWAS_FILE="${PHENO_DIR}/rep${rep}_rep${rep}_${PHENOTYPE}_train_GWAS.rep${rep}_${PHENOTYPE}.glm.logistic.hybrid"

    BIM_PREFIX="${BIM_BASE}/rep${rep}/chr1_X_train_no_cov_no_missing"

    SST_DIR="${SST_BASE}/rep${rep}/${PHENOTYPE}"
    OUT_DIR="${PRSCS_OUT_BASE}/rep${rep}/${PHENOTYPE}"
    LOG_DIR="${LOG_BASE}/rep${rep}/${PHENOTYPE}"

    SST_FILE="${SST_DIR}/chr1.tsv"
    N_GWAS_FILE="${SST_DIR}/n_gwas.txt"
    OUTPUT_PREFIX="${OUT_DIR}/PRScs_${PHENOTYPE}_chr1"

    mkdir -p "${SST_DIR}" "${OUT_DIR}" "${LOG_DIR}"

    ###########################################################################
    # Check repetition-specific files
    ###########################################################################

    if [[ ! -f "${GWAS_FILE}" ]]; then
        echo "WARNING: GWAS file not found for rep${rep}:"
        echo "${GWAS_FILE}"
        echo "Trying to locate the GWAS file automatically."

        GWAS_FILE=$(
            find "${PHENO_DIR}" \
                -maxdepth 1 \
                -type f \
                -name "*.glm.logistic.hybrid" \
                -print \
                -quit 2>/dev/null || true
        )
    fi

    if [[ -z "${GWAS_FILE}" || ! -f "${GWAS_FILE}" ]]; then
        echo "ERROR: no GWAS file found for rep${rep}."
        echo "Skipping rep${rep}."
        continue
    fi

    if [[ ! -f "${BIM_PREFIX}.bim" ]]; then
        echo "ERROR: BIM file not found for rep${rep}:"
        echo "${BIM_PREFIX}.bim"
        echo "Skipping rep${rep}."
        continue
    fi

    echo "GWAS file: ${GWAS_FILE}"
    echo "BIM prefix: ${BIM_PREFIX}"

    ###########################################################################
    # Format PLINK2 GWAS output for PRS-CS
    ###########################################################################

    python - "${GWAS_FILE}" "${SST_FILE}" "${N_GWAS_FILE}" <<'PYTHON'
import sys

import numpy as np
import pandas as pd


gwas_file = sys.argv[1]
output_file = sys.argv[2]
n_gwas_file = sys.argv[3]

df = pd.read_csv(
    gwas_file,
    sep=r"\s+",
    low_memory=False,
    na_values=["NA", "NaN", "nan", "."],
)

print(f"Reading: {gwas_file}")
print(f"Initial rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")


###############################################################################
# Keep additive test only
###############################################################################

if "TEST" in df.columns:
    df = df.loc[
        df["TEST"].astype(str).str.upper() == "ADD"
    ].copy()


###############################################################################
# Keep chromosome 1 only
###############################################################################

if "#CHROM" in df.columns:
    chromosome_column = "#CHROM"
elif "CHROM" in df.columns:
    chromosome_column = "CHROM"
else:
    raise ValueError(
        "No chromosome column was found. "
        f"Available columns: {list(df.columns)}"
    )

chromosome = pd.to_numeric(
    df[chromosome_column],
    errors="coerce",
)

df = df.loc[chromosome == 1].copy()

print(f"Chromosome 1 rows: {len(df):,}")


###############################################################################
# Validate required columns
###############################################################################

required_columns = ["ID", "A1", "P"]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}. "
        f"Available columns: {list(df.columns)}"
    )


###############################################################################
# Construct A2
###############################################################################

if "A2" in df.columns:
    df["PRSCS_A2"] = df["A2"]

elif "REF" in df.columns and "ALT" in df.columns:
    ref = df["REF"].astype(str).str.upper()
    alt = df["ALT"].astype(str).str.upper()
    a1 = df["A1"].astype(str).str.upper()

    df["PRSCS_A2"] = np.where(
        a1 == alt,
        ref,
        np.where(
            a1 == ref,
            alt,
            np.nan,
        ),
    )

else:
    raise ValueError(
        "Cannot construct A2. The GWAS file must contain either "
        "A2 or both REF and ALT."
    )


###############################################################################
# Construct BETA
###############################################################################

if "BETA" in df.columns:
    df["PRSCS_BETA"] = pd.to_numeric(
        df["BETA"],
        errors="coerce",
    )

elif "OR" in df.columns:
    odds_ratio = pd.to_numeric(
        df["OR"],
        errors="coerce",
    )

    df["PRSCS_BETA"] = np.where(
        odds_ratio > 0,
        np.log(odds_ratio),
        np.nan,
    )

else:
    raise ValueError(
        "Neither BETA nor OR was found in the GWAS output."
    )


###############################################################################
# Filter invalid variants
###############################################################################

df["SNP"] = df["ID"].astype(str)
df["A1"] = df["A1"].astype(str).str.upper()
df["A2"] = df["PRSCS_A2"].astype(str).str.upper()
df["P"] = pd.to_numeric(df["P"], errors="coerce")

valid_alleles = {"A", "C", "G", "T"}

valid_rows = (
    (df["SNP"] != ".")
    & df["A1"].isin(valid_alleles)
    & df["A2"].isin(valid_alleles)
    & (df["A1"] != df["A2"])
    & df["PRSCS_BETA"].notna()
    & np.isfinite(df["PRSCS_BETA"])
    & df["P"].notna()
    & np.isfinite(df["P"])
    & (df["P"] > 0)
    & (df["P"] <= 1)
)

df = df.loc[valid_rows].copy()

df = df.drop_duplicates(
    subset=["SNP"],
    keep="first",
)

if df.empty:
    raise ValueError(
        "No valid chromosome 1 SNPs remained after filtering."
    )


###############################################################################
# Derive GWAS sample size
###############################################################################

if "OBS_CT" not in df.columns:
    raise ValueError(
        "OBS_CT is missing from the GWAS output."
    )

obs_ct = pd.to_numeric(
    df["OBS_CT"],
    errors="coerce",
)

obs_ct = obs_ct.loc[
    obs_ct.notna() & (obs_ct > 0)
]

if obs_ct.empty:
    raise ValueError(
        "No valid sample-size values were found in OBS_CT."
    )

n_gwas = int(round(obs_ct.median()))

with open(n_gwas_file, "w", encoding="utf-8") as handle:
    handle.write(f"{n_gwas}\n")


###############################################################################
# Write PRS-CS summary statistics
###############################################################################

output_df = df[
    ["SNP", "A1", "A2", "PRSCS_BETA", "P"]
].rename(
    columns={"PRSCS_BETA": "BETA"}
)

output_df.to_csv(
    output_file,
    sep="\t",
    index=False,
)

print(f"Written SNPs: {len(output_df):,}")
print(f"n_gwas: {n_gwas:,}")
print(f"Output: {output_file}")
PYTHON

    N_GWAS=$(tr -d '[:space:]' < "${N_GWAS_FILE}")

    if [[ -z "${N_GWAS}" ]]; then
        echo "ERROR: n_gwas was not created for rep${rep}."
        echo "Skipping rep${rep}."
        continue
    fi

    ###########################################################################
    # Submit PRS-CS job
    ###########################################################################

    JOB_ID=$(
        sbatch \
            --parsable \
            --mem="${MEMORY}" \
            --time="${TIME_LIMIT}" \
            --cpus-per-task="${CPUS}" \
            --job-name="PRS_GI30_r${rep}" \
            --output="${LOG_DIR}/PRScs_%j.out" \
            --error="${LOG_DIR}/PRScs_%j.err" \
            --wrap="
                set -euo pipefail

                echo 'Repetition: rep${rep}'
                echo 'Phenotype: ${PHENOTYPE}'
                echo 'Chromosome: 1'
                echo 'n_gwas: ${N_GWAS}'
                echo 'BIM prefix: ${BIM_PREFIX}'
                echo 'Summary statistics: ${SST_FILE}'
                echo 'Output prefix: ${OUTPUT_PREFIX}'

                python '${PRSCS_SCRIPT}' \
                    --ref_dir='${REF_DIR}' \
                    --bim_prefix='${BIM_PREFIX}' \
                    --sst_file='${SST_FILE}' \
                    --n_gwas='${N_GWAS}' \
                    --out_dir='${OUTPUT_PREFIX}' \
                    --chrom=1 \
                    --seed=3
            "
    )

    echo "Submitted rep${rep}: job ${JOB_ID}"
    echo "Output prefix: ${OUTPUT_PREFIX}"
done

echo
echo "Finished submitting GI_30 PRS-CS jobs."