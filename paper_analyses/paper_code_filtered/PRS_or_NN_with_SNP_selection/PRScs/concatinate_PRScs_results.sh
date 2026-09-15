#phenotypes=("Hypertension" "Type_2_Diabetes" "Multiple_Sclerosis" "Height" "Platelet_Count")
phenotypes=("BMI")

for pheno in "${phenotypes[@]}"; do
    for rep in {1..5}; do
        dir="my_storage/my_storage/impoving_PRS/data/PRScs/PRScs_results/${pheno}/rep${rep}"
        output_file="${dir}/merged_results.txt"

        # Check if the directory exists before running the command
        if [ -d "$dir" ]; then
            cat $(ls ${dir}/PRScs_chr*_${pheno}_results_pst_eff_a1_b0.5_phiauto_chr*.txt | sort -V) > "$output_file"
            echo "Merged results for ${pheno}, rep${rep} saved in: $output_file"
        else
            echo "Directory $dir not found, skipping..."
        fi
    done
done
