#!/bin/bash

PLINK2="/path/to/your/project//plink2"

BED_PREFIX="/path/to/your/project//ukbb_filtered_maf_0.001_hwe_1e-6_geno_0.0_ex_mismtach/chr1_imputed_snp_id_filtered"

PHENO_DIR="/path/to/your/project//simulations/combined_DRIP_phenotypes_by_rep"

OUT_BASE="/path/to/your/project//simulations/GWAS_results_DRIP_train"

COV="/path/to/your/project//UKBB/phenotypes/cov_matrix_MinMax_scaled_no_missing.txt"

mkdir -p "${OUT_BASE}"

for rep in {1..5}
do
    echo "Submitting GWAS jobs for rep${rep}"

    PHENO_FILE="${PHENO_DIR}/rep${rep}_train_phenotypes.txt"
    OUT_DIR="${OUT_BASE}/rep${rep}"
    mkdir -p "${OUT_DIR}"

    # Extract phenotype names from columns 3 onward
    read -a PHENO_ARRAY <<< "$(head -n 1 "${PHENO_FILE}" | cut -f3-)"

    for pheno in "${PHENO_ARRAY[@]:1}"
    do
        echo "Submitting GWAS for rep${rep}, phenotype ${pheno}"

        PHENO_OUT_DIR="${OUT_DIR}/${pheno}"
        mkdir -p "${PHENO_OUT_DIR}"

        sbatch \
            --mem=200g \
            --time=5-0 \
            -c 48 \
            --job-name="GWAS_r${rep}_${pheno}" \
            --partition=cpu \
            -o "${PHENO_OUT_DIR}/GWAS_rep${rep}_${pheno}.out" \
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
                --out ${PHENO_OUT_DIR}/rep${rep}_${pheno}_train_GWAS"
    done
done