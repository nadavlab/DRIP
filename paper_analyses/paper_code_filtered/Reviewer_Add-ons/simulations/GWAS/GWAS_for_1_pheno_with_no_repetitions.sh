#!/bin/bash

# User provides phenotype file path as the first argument
PHENO_FILE=$1

if [ -z "${PHENO_FILE}" ]; then
    echo "Usage: $0 /path/to/phenotype_file.csv"
    exit 1
fi

GWAS_DIR=/path/to/your/project//data/simulations/GWAS_results
BED_DIR=/path/to/your/project/s
COVAR_FILE=/path/to/your/project//data/phenotype_files/covariates_for_gwas_Neals_like.txt
PLINK=/path/to/your/project//plink2

mkdir -p ${GWAS_DIR}

echo "Processing ${PHENO_FILE}"

# extract phenotype name (3rd column header)
PHENO_NAME=$(head -n 1 ${PHENO_FILE} | cut -d',' -f3)

# use file name (without .csv) as directory
PHENO_BASENAME=$(basename ${PHENO_FILE} .csv)
OUTDIR=${GWAS_DIR}/${PHENO_BASENAME}_chro1_only
mkdir -p ${OUTDIR}

# loop over chromosome(s) - here only chromosome 1
for chr in {1..1}
do
    sbatch -A nati --mem=48g --time=5-0 -c 16 --exclude joey-01 \
      --job-name=GWAS_${PHENO_BASENAME}_chr${chr} \
      -o ${OUTDIR}/chr${chr}.out \
      --wrap="${PLINK} \
        --bed ${BED_DIR}/chr${chr}_filtered_mafe-6_hwee-5_geno_02.bed \
        --bim ${BED_DIR}/chr${chr}_filtered_mafe-6_hwee-5_geno_02.bim \
        --fam ${BED_DIR}/chr${chr}_filtered_mafe-6_hwee-5_geno_02.fam \
        --pheno ${PHENO_FILE} \
        --pheno-name ${PHENO_NAME} \
        --glm hide-covar --vif 100 \
        --covar ${COVAR_FILE} \
        --covar-name sex,age,PC1-PC20 \
        --covar-variance-standardize \
        --threads 16 --hwe 1e-6 --geno 0.1 --maf 0.001 \
        --out ${OUTDIR}/chr${chr}_${PHENO_BASENAME}chro1_only_maf_0.001_prev_0.1_round3"
done
