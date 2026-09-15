#!/bin/bash

PLINK2="/path/to/your/project//plink2"

BED_PREFIX="/path/to/your/project//ukbb_filtered_maf_0.001_hwe_1e-6_geno_0.0_ex_mismtach/chr1_imputed_snp_id_filtered"

PHENO_DIR="/path/to/your/project//simulations/combined_DRIP_phenotypes_by_rep"

OUT_BASE="/path/to/your/project//simulations/GWAS_results_DRIP_train"

COV="/path/to/your/project//UKBB/phenotypes/cov_matrix_MinMax_scaled_no_missing.txt"

DRY_RUN=0   # Change to 0 after checking the printed list

for rep in {1..5}
do
    echo "Checking rep${rep}"

    PHENO_FILE="${PHENO_DIR}/rep${rep}_train_phenotypes.txt"
    OUT_DIR="${OUT_BASE}/rep${rep}"

    if [[ ! -s "${PHENO_FILE}" ]]
    then
        echo "ERROR: phenotype file is missing or empty: ${PHENO_FILE}"
        continue
    fi

    read -ra PHENO_ARRAY <<< "$(head -n 1 "${PHENO_FILE}" | cut -f3-)"

    for pheno in "${PHENO_ARRAY[@]}"
    do
        PHENO_OUT_DIR="${OUT_DIR}/${pheno}"
        PREFIX="${PHENO_OUT_DIR}/rep${rep}_${pheno}_train_GWAS"

        # Skip the phenotype when its output directory contains any file.
        if [[ -d "${PHENO_OUT_DIR}" ]] &&
           find "${PHENO_OUT_DIR}" -maxdepth 1 -type f -print -quit | grep -q .
        then
            echo "SKIP — output files already exist: rep${rep} ${pheno}"
            continue
        fi

        echo "SUBMIT — no output files found: rep${rep} ${pheno}"

        if [[ "${DRY_RUN}" -eq 0 ]]
        then
            mkdir -p "${PHENO_OUT_DIR}"

            sbatch \
                --mem=48g \
                --time=6-0 \
                -c 16 \
                --job-name="GWAS_r${rep}_${pheno}" \
                --partition=cpu \
                -o "${PHENO_OUT_DIR}/GWAS_rep${rep}_${pheno}.out" \
                -e "${PHENO_OUT_DIR}/GWAS_rep${rep}_${pheno}.err" \
                --wrap="${PLINK2} \
                    --bed ${BED_PREFIX}.bed \
                    --bim ${BED_PREFIX}.bim \
                    --fam ${BED_PREFIX}.fam \
                    --pheno ${PHENO_FILE} \
                    --pheno-name ${pheno} \
                    --keep ${PHENO_FILE} \
                    --glm hide-covar \
                    --covar ${COV} \
                    --vif 100 \
                    --threads 16 \
                    --out ${PREFIX}"
        fi
    done
done