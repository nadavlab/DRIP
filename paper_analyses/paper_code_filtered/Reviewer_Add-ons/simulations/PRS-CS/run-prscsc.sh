#!/bin/bash

###############################################################################
# Run PRS-CS for all simulated DRIP phenotypes
# Simulations contain chromosome 1 only
#
# Usage:
#   bash run_PRScs_simulations_chr1.sh
#
# Run one repetition only:
#   bash run_PRScs_simulations_chr1.sh 3
###############################################################################

set -euo pipefail

BASE="/path/to/your/project/"

PRSCS_SCRIPT="${BASE}/PRScs/PRScs/PRScs.py"
REF_DIR="${BASE}/PRScs/ldblk_ukbb_eur"

GWAS_BASE="${BASE}/data/simulations/GWAS_results_DRIP_train"

# Change this if the simulated chromosome-1 genotype prefix is elsewhere
BIM_BASE="${BASE}/data/training_bed_files"

SST_BASE="${BASE}/data/simulations/PRScs/SS_formatted_data"
PRSCS_OUT_BASE="${BASE}/data/simulations/PRScs/PRScs_results"
LOG_BASE="${BASE}/data/simulations/PRScs/slurm_logs"

MEMORY="32G"
TIME_LIMIT="1-00:00:00"
CPUS=1

CHROM=1

if [[ $# -ge 1 ]]; then
    REPS=("$1")
else
    REPS=(4 5)
fi

mkdir -p "${SST_BASE}" "${PRSCS_OUT_BASE}" "${LOG_BASE}"

for rep in "${REPS[@]}"
do
    REP_GWAS_DIR="${GWAS_BASE}/rep${rep}"

    if [[ ! -d "${REP_GWAS_DIR}" ]]; then
        echo "WARNING: missing directory: ${REP_GWAS_DIR}"
        continue
    fi

    echo
    echo "============================================================"
    echo "Processing rep${rep}"
    echo "============================================================"

    while IFS= read -r PHENO_DIR
    do
        FULL_DIR_NAME=$(basename "${PHENO_DIR}")

        # Remove the leading rep number from the phenotype name
        PHENOTYPE="${FULL_DIR_NAME#rep${rep}_}"

        echo
        echo "Phenotype: ${PHENOTYPE}"

        GWAS_FILE=$(find "${PHENO_DIR}" \
            -maxdepth 1 \
            -type f \
            \( \
                -name "*.glm.logistic.hybrid" \
                -o -name "*.glm.logistic" \
                -o -name "*.glm.linear" \
            \) \
            | head -n 1)

        if [[ -z "${GWAS_FILE}" ]]; then
            echo "WARNING: no GWAS result found in ${PHENO_DIR}"
            continue
        fi

        SST_DIR="${SST_BASE}/rep${rep}/${PHENOTYPE}"
        OUT_DIR="${PRSCS_OUT_BASE}/rep${rep}/${PHENOTYPE}"
        PHENO_LOG_DIR="${LOG_BASE}/rep${rep}/${PHENOTYPE}"

        mkdir -p "${SST_DIR}" "${OUT_DIR}" "${PHENO_LOG_DIR}"

        SST_FILE="${SST_DIR}/chr1.tsv"
        N_GWAS_FILE="${SST_DIR}/n_gwas.txt"

        #######################################################################
        # Format chromosome 1 PLINK2 GWAS results for PRS-CS
        #######################################################################

        python - "${GWAS_FILE}" "${SST_FILE}" "${N_GWAS_FILE}" <<'PYTHON'
import os
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


# Keep additive test only
if "TEST" in df.columns:
    df = df.loc[
        df["TEST"].astype(str).str.upper() == "ADD"
    ].copy()


# Identify chromosome column
if "#CHROM" in df.columns:
    chrom_col = "#CHROM"
elif "CHROM" in df.columns:
    chrom_col = "CHROM"
else:
    raise ValueError("No chromosome column was found.")


# Keep chromosome 1 only
chromosome = pd.to_numeric(df[chrom_col], errors="coerce")
df = df.loc[chromosome == 1].copy()


# Validate required columns
required = ["ID", "A1", "P"]

for column in required:
    if column not in df.columns:
        raise ValueError(
            f"Required column {column} is missing. "
            f"Available columns: {list(df.columns)}"
        )


# Construct A2
if "A2" in df.columns:
    df["PRSCS_A2"] = df["A2"]

elif "REF" in df.columns and "ALT" in df.columns:
    ref = df["REF"].astype(str).str.upper()
    alt = df["ALT"].astype(str).str.upper()
    a1 = df["A1"].astype(str).str.upper()

    df["PRSCS_A2"] = np.where(
        a1 == alt,
        ref,
        np.where(a1 == ref, alt, ref),
    )

else:
    raise ValueError(
        "Cannot construct A2: A2 or REF/ALT columns are required."
    )


# Construct BETA
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


df["SNP"] = df["ID"].astype(str)
df["A1"] = df["A1"].astype(str).str.upper()
df["A2"] = df["PRSCS_A2"].astype(str).str.upper()
df["P"] = pd.to_numeric(df["P"], errors="coerce")

valid_alleles = {"A", "C", "G", "T"}

valid = (
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

df = df.loc[valid].copy()

df = df.drop_duplicates(
    subset=["SNP"],
    keep="first",
)

if len(df) == 0:
    raise ValueError(
        "No valid chromosome 1 SNPs remained after filtering."
    )


# Derive GWAS sample size
if "OBS_CT" not in df.columns:
    raise ValueError(
        "OBS_CT is missing. Set n_gwas manually if necessary."
    )

obs_ct = pd.to_numeric(
    df["OBS_CT"],
    errors="coerce",
)

obs_ct = obs_ct.loc[
    obs_ct.notna() & (obs_ct > 0)
]

if len(obs_ct) == 0:
    raise ValueError(
        "No valid sample-size values were found in OBS_CT."
    )

n_gwas = int(round(obs_ct.median()))

with open(n_gwas_file, "w") as handle:
    handle.write(f"{n_gwas}\n")


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

        #######################################################################
        # Chromosome 1 training genotype prefix
        #######################################################################

        BIM_PREFIX="${BIM_BASE}/rep${rep}/chr1_X_train_no_cov_no_missing"

        if [[ ! -f "${BIM_PREFIX}.bim" ]]; then
            echo "ERROR: BIM file not found:"
            echo "${BIM_PREFIX}.bim"
            echo "Skipping ${PHENOTYPE}"
            continue
        fi

        OUTPUT_PREFIX="${OUT_DIR}/PRScs_${PHENOTYPE}_chr1"

        #######################################################################
        # Submit one job for this phenotype
        #######################################################################

        JOB_ID=$(
            sbatch \
                --parsable \
                --mem="${MEMORY}" \
                --time="${TIME_LIMIT}" \
                --cpus-per-task="${CPUS}" \
                --job-name="PCSr${rep}" \
                --output="${PHENO_LOG_DIR}/PRScs_%j.out" \
                --error="${PHENO_LOG_DIR}/PRScs_%j.err" \
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

        echo "Submitted job ${JOB_ID}"
        echo "Output: ${OUTPUT_PREFIX}"

    done < <(
        find "${REP_GWAS_DIR}" \
            -mindepth 1 \
            -maxdepth 1 \
            -type d \
            -name "rep${rep}_*" \
            | sort
    )
done

echo
echo "All chromosome 1 PRS-CS jobs were submitted."