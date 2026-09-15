#!/bin/bash

###############################################################################
# Calculate PRS scores for the GI_30 simulated phenotype after PRS-CS
# Chromosome 1 only
#
# Usage:
#   bash score_PRScs_GI30_chr1.sh
#
# Run selected repetitions:
#   bash score_PRScs_GI30_chr1.sh 1 3 5
###############################################################################

set -euo pipefail


###############################################################################
# Paths
###############################################################################

BASE="/path/to/your/project/"

PLINK="/path/to/your/project//plink"

TEST_BED_BASE="${BASE}/data/test_bed_files"

PRSCS_RESULTS_BASE="${BASE}/data/simulations/PRScs/PRScs_results"

LOG_BASE="${BASE}/data/simulations/PRScs/score_logs"


###############################################################################
# Resources
###############################################################################

MEMORY="16G"
TIME_LIMIT="12:00:00"
CPUS=1


###############################################################################
# Phenotype
###############################################################################

PHENOTYPE="chr1_N100_GI_30_h2add_0p00_h2gxg_0p30_cov_0p200_noise_0p500"


###############################################################################
# Repetitions
###############################################################################

if [[ $# -gt 0 ]]; then
    REPS=("$@")
else
    REPS=(1 2 3 4 5)
fi


mkdir -p "${LOG_BASE}"


###############################################################################
# Check PLINK
###############################################################################

if [[ ! -x "${PLINK}" ]]; then
    echo "ERROR: PLINK executable was not found or is not executable:"
    echo "${PLINK}"
    exit 1
fi


###############################################################################
# Process GI_30 for each repetition
###############################################################################

for rep in "${REPS[@]}"
do
    echo
    echo "============================================================"
    echo "Processing rep${rep}"
    echo "Phenotype: ${PHENOTYPE}"
    echo "============================================================"

    ###########################################################################
    # PRS-CS result directory
    ###########################################################################

    PHENO_DIR="${PRSCS_RESULTS_BASE}/rep${rep}/${PHENOTYPE}"

    if [[ ! -d "${PHENO_DIR}" ]]; then
        echo "WARNING: PRS-CS phenotype directory not found:"
        echo "${PHENO_DIR}"
        echo "Skipping rep${rep}."
        continue
    fi

    ###########################################################################
    # Test chromosome 1 genotype prefix
    ###########################################################################

    BED_PREFIX="${TEST_BED_BASE}/rep${rep}/chr1_X_test_no_cov_no_missing"

    if [[ ! -f "${BED_PREFIX}.bed" ]]; then
        echo "ERROR: BED file not found:"
        echo "${BED_PREFIX}.bed"
        echo "Skipping rep${rep}."
        continue
    fi

    if [[ ! -f "${BED_PREFIX}.bim" ]]; then
        echo "ERROR: BIM file not found:"
        echo "${BED_PREFIX}.bim"
        echo "Skipping rep${rep}."
        continue
    fi

    if [[ ! -f "${BED_PREFIX}.fam" ]]; then
        echo "ERROR: FAM file not found:"
        echo "${BED_PREFIX}.fam"
        echo "Skipping rep${rep}."
        continue
    fi

    ###########################################################################
    # PRS-CS posterior-effect file
    ###########################################################################

    SCORE_FILE="${PHENO_DIR}/PRScs_${PHENOTYPE}_chr1_pst_eff_a1_b0.5_phiauto_chr1.txt"

    if [[ ! -f "${SCORE_FILE}" ]]; then
        echo "Expected score file was not found."
        echo "Searching for a matching PRS-CS output file."

        SCORE_FILE=$(
            find "${PHENO_DIR}" \
                -maxdepth 1 \
                -type f \
                -name "*pst_eff_a1*b0.5*phiauto*chr1*.txt" \
                -print \
                -quit 2>/dev/null || true
        )
    fi

    if [[ -z "${SCORE_FILE}" || ! -f "${SCORE_FILE}" ]]; then
        echo "WARNING: PRS-CS score file not found for rep${rep}."
        echo "Directory: ${PHENO_DIR}"
        echo "Skipping rep${rep}."
        continue
    fi

    ###########################################################################
    # Output locations
    ###########################################################################

    SCORE_OUTPUT_DIR="${PHENO_DIR}/PRS_scores"
    PHENO_LOG_DIR="${LOG_BASE}/rep${rep}/${PHENOTYPE}"

    mkdir -p "${SCORE_OUTPUT_DIR}" "${PHENO_LOG_DIR}"

    OUTPUT_PREFIX="${SCORE_OUTPUT_DIR}/PRS_${PHENOTYPE}_rep${rep}_chr1"

    ###########################################################################
    # Skip completed scoring
    ###########################################################################

    if [[ -s "${OUTPUT_PREFIX}.profile" ]]; then
        echo "PRS output already exists:"
        echo "${OUTPUT_PREFIX}.profile"
        echo "Skipping rep${rep}."
        continue
    fi

    echo "Test genotype:  ${BED_PREFIX}"
    echo "PRS-CS weights: ${SCORE_FILE}"
    echo "Output prefix:  ${OUTPUT_PREFIX}"

    ###########################################################################
    # Submit PLINK scoring job
    #
    # PRS-CS output columns:
    #   1 = chromosome
    #   2 = SNP
    #   3 = position
    #   4 = A1
    #   5 = A2
    #   6 = posterior effect
    ###########################################################################

    JOB_ID=$(
        sbatch \
            --parsable \
            --mem="${MEMORY}" \
            --time="${TIME_LIMIT}" \
            --cpus-per-task="${CPUS}" \
            --job-name="Score_GI30_r${rep}" \
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

    echo "Submitted rep${rep}: job ${JOB_ID}"
done

echo
echo "Finished submitting available GI_30 PRS scoring jobs."