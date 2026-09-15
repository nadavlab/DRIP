#!/bin/bash

PHENO="Hypertension"
DRM="PCA"          # or Autoencoder

for rep in {1..5}
do
    OUTDIR="/path/to/your/project//reviewer_feature_set_comparisondata/logistic_regression/${PHENO}/rep${rep}/"

    mkdir -p "${OUTDIR}"

    echo "Submitting ${PHENO}, rep ${rep}"

    sbatch \
        --mem=64g \
        --time=4-0 \
        -c 8 \
        --job-name=LRReview_${PHENO}_rep${rep} \
        --mail-user=/yourmailadress \
        --mail-type=BEGIN,END,FAIL \
        -o "${OUTDIR}/reviewer_%J.out" \
        --wrap="python3 /path/to/your/project//code/from_the_begining_in_bgu/for_reviewers/run_LR_with_different_features_levels_hypertension.py ${PHENO} ${rep} ${DRM}"

done