#!/bin/bash

PLINK2="/path/to/your/project//plink2"

BED_PREFIX="/path/to/your/project//ukbb_filtered_maf_0.001_hwe_1e-6_geno_0.0_ex_mismtach/chr1_imputed_snp_id_filtered"

PHENO_DIR="/path/to/your/project//simulations/combined_DRIP_phenotypes_by_rep"

OUT_BASE="/path/to/your/project//simulations/GWAS_results_DRIP_train"

COV="/path/to/your/project//UKBB/phenotypes/cov_matrix_MinMax_scaled_no_missing.txt"

DRY_RUN=1   # change to 0 when the printed list looks correct

for rep in {1..5}
do
    echo "Checking rep${rep}"

    PHENO_FILE="${PHENO_DIR}/rep${rep}_train_phenotypes.txt"
    OUT_DIR="${OUT_BASE}/rep${rep}"

    read -a PHENO_ARRAY <<< "$(head -n 1 "${PHENO_FILE}" | cut -f3-)"

    for pheno in "${PHENO_ARRAY[@]}"
    do
        PHENO_OUT_DIR="${OUT_DIR}/${pheno}"
        PREFIX="${PHENO_OUT_DIR}/rep${rep}_${pheno}_train_GWAS"
        LOG_FILE="${PREFIX}.log"

        HAS_GLM=0
        compgen -G "${PREFIX}"'*.glm.*' > /dev/null && HAS_GLM=1

        FINISHED=0
        if [[ -s "${LOG_FILE}" ]] && grep -q "End time:" "${LOG_FILE}" && [[ "${HAS_GLM}" -eq 1 ]]
        then
            FINISHED=1
        fi

        if [[ "${FINISHED}" -eq 1 ]]
        then
            echo "DONE: rep${rep} ${pheno}"
        else
            echo "RESUBMIT: rep${rep} ${pheno}"

            mkdir -p "${PHENO_OUT_DIR}"

            if [[ "${DRY_RUN}" -eq 0 ]]
            then
                sbatch \
                    --mem=48g \
                    --time=6-0 \
                    -c 16 \
                    --job-name="GWAS_r${rep}" \
                    --partition=cpu \
                    -o "${PHENO_OUT_DIR}/GWAS_rep${rep}_${pheno}.resub.out" \
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
                        --out ${PREFIX}"
            fi
        fi
    done
done