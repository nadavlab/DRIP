#!/bin/bash

###############################################################################
# Calculate PRS scores for simulated phenotypes after PRS-CS
#
# Usage:
#   bash score_PRScs_simulations_chr1.sh
#
# Run one repetition only:
#   bash score_PRScs_simulations_chr1.sh 3
###############################################################################

set -euo pipefail


###############################################################################
# Paths
###############################################################################

BASE="/path/to/your/project/"

PLINK="/path/to/your/project//plink"

# Test genotype files
TEST_BED_BASE="${BASE}/data/test_bed_files"

# PRS-CS results created in the previous step
PRSCS_RESULTS_BASE="${BASE}/data/simulations/PRScs/PRScs_results"

# Slurm logs
LOG_BASE="${BASE}/data/simulations/PRScs/score_logs"


###############################################################################
# Resources
###############################################################################

MEMORY="16G"
TIME_LIMIT="12:00:00"
CPUS=1

CHROM=1


###############################################################################
# Repetitions
###############################################################################

if [[ $# -ge 1 ]]; then
    REPS=("$1")
else
    REPS=(1 2 3 4 5)
fi


mkdir -p "${LOG_BASE}"


###############################################################################
# Process all simulated phenotypes
###############################################################################

for rep in "${REPS[@]}"
do
    REP_RESULTS_DIR="${PRSCS_RESULTS_BASE}/rep${rep}"

    if [[ ! -d "${REP_RESULTS_DIR}" ]]; then
        echo "WARNING: PRS-CS results directory not found:"
        echo "${REP_RESULTS_DIR}"
        continue
    fi

    ###########################################################################
    # Test chromosome 1 genotype prefix
    ###########################################################################

    BED_PREFIX="${TEST_BED_BASE}/rep${rep}/chr1_X_test_no_cov_no_missing"

    if [[ ! -f "${BED_PREFIX}.bed" ]]; then
        echo "ERROR: BED file not found:"
        echo "${BED_PREFIX}.bed"
        echo "Skipping rep${rep}"
        continue
    fi

    if [[ ! -f "${BED_PREFIX}.bim" ]]; then
        echo "ERROR: BIM file not found:"
        echo "${BED_PREFIX}.bim"
        echo "Skipping rep${rep}"
        continue
    fi

    if [[ ! -f "${BED_PREFIX}.fam" ]]; then
        echo "ERROR: FAM file not found:"
        echo "${BED_PREFIX}.fam"
        echo "Skipping rep${rep}"
        continue
    fi

    echo
    echo "============================================================"
    echo "Processing PRS scoring for rep${rep}"
    echo "Test genotype: ${BED_PREFIX}"
    echo "============================================================"

    ###########################################################################
    # Each directory is one simulated phenotype
    ###########################################################################

    while IFS= read -r PHENO_DIR
    do
        PHENOTYPE=$(basename "${PHENO_DIR}")

        echo
        echo "------------------------------------------------------------"
        echo "Repetition: rep${rep}"
        echo "Phenotype:  ${PHENOTYPE}"
        echo "------------------------------------------------------------"

        #######################################################################
        # PRS-CS output file
        #
        # Previous PRS-CS prefix:
        #   PRScs_${PHENOTYPE}_chr1
        #
        # Standard PRS-CS output:
        #   PRScs_${PHENOTYPE}_chr1_pst_eff_a1_b0.5_phiauto_chr1.txt
        #######################################################################

        SCORE_FILE="${PHENO_DIR}/PRScs_${PHENOTYPE}_chr1_pst_eff_a1_b0.5_phiauto_chr1.txt"

        #######################################################################
        # Find the file more flexibly if the exact expected name is absent
        #######################################################################

        if [[ ! -f "${SCORE_FILE}" ]]; then
            SCORE_FILE=$(find "${PHENO_DIR}" \
                -maxdepth 1 \
                -type f \
                -name "*pst_eff_a1*b0.5*phiauto*chr1*.txt" \
                | head -n 1)
        fi

        if [[ -z "${SCORE_FILE}" || ! -f "${SCORE_FILE}" ]]; then
            echo "WARNING: PRS-CS score file not found."
            echo "Expected directory: ${PHENO_DIR}"
            echo "Skipping phenotype."
            continue
        fi

        #######################################################################
        # Output locations
        #######################################################################

        SCORE_OUTPUT_DIR="${PHENO_DIR}/PRS_scores"
        PHENO_LOG_DIR="${LOG_BASE}/rep${rep}/${PHENOTYPE}"

        mkdir -p "${SCORE_OUTPUT_DIR}"
        mkdir -p "${PHENO_LOG_DIR}"

        OUTPUT_PREFIX="${SCORE_OUTPUT_DIR}/PRS_${PHENOTYPE}_rep${rep}_chr1"

        #######################################################################
        # Skip already completed jobs
        #######################################################################

        if [[ -s "${OUTPUT_PREFIX}.profile" ]]; then
            echo "PRS output already exists:"
            echo "${OUTPUT_PREFIX}.profile"
            echo "Skipping."
            continue
        fi

        echo "PRS-CS weights: ${SCORE_FILE}"
        echo "Output prefix:  ${OUTPUT_PREFIX}"

        #######################################################################
        # Submit PLINK scoring job
        #
        # PRS-CS output columns:
        #   1 = chromosome
        #   2 = SNP
        #   3 = position
        #   4 = A1
        #   5 = A2
        #   6 = posterior effect
        #
        # Therefore:
        #   --score FILE 2 4 6 sum
        #######################################################################

        JOB_ID=$(
            sbatch \
                --parsable \
                --mem="${MEMORY}" \
                --time="${TIME_LIMIT}" \
                --cpus-per-task="${CPUS}" \
                --job-name="Score_r${rep}" \
                --output="${PHENO_LOG_DIR}/score_%j.out" \
                --error="${PHENO_LOG_DIR}/score_%j.err" \
                --wrap="
                    set -euo pipefail

                    echo '============================================================'
                    echo 'Repetition: rep${rep}'
                    echo 'Phenotype: ${PHENOTYPE}'
                    echo 'Chromosome: 1'
                    echo 'BED prefix: ${BED_PREFIX}'
                    echo 'Score file: ${SCORE_FILE}'
                    echo 'Output: ${OUTPUT_PREFIX}'
                    echo '============================================================'

                    '${PLINK}' \
                        --bfile '${BED_PREFIX}' \
                        --score '${SCORE_FILE}' 2 4 6 sum \
                        --out '${OUTPUT_PREFIX}'
                "
        )

        echo "Submitted job: ${JOB_ID}"

    done < <(
        find "${REP_RESULTS_DIR}" \
            -mindepth 1 \
            -maxdepth 1 \
            -type d \
            | sort
    )
done

echo
echo "All available chromosome 1 PRS scoring jobs were submitted."