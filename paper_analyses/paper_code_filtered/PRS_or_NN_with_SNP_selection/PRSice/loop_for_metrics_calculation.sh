#!/bin/bash

phenotypes=("Parkinson" "Schizophrenia" "Alzheimer" "Multiple_Sclerosis" "Type_2_Diabetes" "Platelet_Count" "Height" "Hypertension")

for pheno in "${phenotypes[@]}"; do
    for rep in {1..5}; do
                sbatch --mem=50g --time=1-0 -c 10 --job-name=$1r${rep} -o /path/to/your/project/PRSice_results/${pheno}/rep${rep}/calaulate_metrics.out --wrap="python /path/to/your/project/impoving_PRS/code/from_the_begining_in_bgu/PRS_or_NN_with_SNP_selection/PRSice/get_ROC_for_binary_targets.py $pheno $rep"
    done
done

