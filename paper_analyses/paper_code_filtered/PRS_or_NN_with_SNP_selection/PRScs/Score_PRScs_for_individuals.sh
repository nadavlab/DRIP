#!/bin/bash

plink_path="/path/to/your/project/plink"
data_path="/path/to/your/project/impoving_PRS/data"

#phenotypes=("Hypertension" "Platelet_Count" "Parkinson" "Schizophrenia" "Alzheimer")
#phenotypes=("Multiple_Sclerosis" "Height" "Type_2_Diabetes")
phenotypes=("BMI")

for pheno in "${phenotypes[@]}"; do
    for rep in {1..5}; do
        for chr in {1..22}; do
            bed_file="${data_path}/test_bed_files/rep${rep}/chr${chr}_X_test_no_cov_no_missing"
            score_file="${data_path}/PRScs/PRScs_results/${pheno}/rep${rep}/PRScs_chr${chr}_${pheno}_results_pst_eff_a1_b0.5_phiauto_chr${chr}.txt"
            output_file="${data_path}/PRScs/PRScs_results/${pheno}/rep${rep}/PRS_results_chr${chr}_rep${rep}"

            if [[ -f "$bed_file.bed" && -f "$score_file" ]]; then
                sbatch --mem=100g --time=1-0 -c 48 --job-name=Scr$1r${rep}c${chr} -o /path/to/your/project/PRScs/PRScs_results/$1/rep${rep}/score_chr${chr}.out --wrap="$plink_path --bfile $bed_file --score $score_file 2 4 6 sum --out $output_file"

                echo "Processed ${pheno}, rep${rep}, chromosome ${chr}"
            else
                echo "Skipping ${pheno}, rep${rep}, chromosome ${chr}: Missing file(s)"
            fi
        done
    done
done

