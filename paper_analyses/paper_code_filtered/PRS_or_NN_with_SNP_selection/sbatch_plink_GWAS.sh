#!/bin/bash
#$1 is the phenotype [the user insert Height/Hypertension/BMI/Platelet_Count...]
#$2 is the number of repS

pheno_file=$(echo $1|tr '[:upper:]' '[:lower:]')
echo $pheno_file

echo $(head -n1 "/path/to/your/project/phenotypes/${pheno_file}"| cut -f3 -d$' ')
pheno_name=$(head -n1 "/path/to/your/project/phenotypes/${pheno_file}"| cut -f3 -d$' ')
echo $pheno_name

mkdir /path/to/your/project/GWAS_results/$1/

for rep in {1..5}
do
echo $rep
mkdir /path/to/your/project/GWAS_results/$1/rep${rep}/

for x in {1..22}
do
        sbatch --mem=100g --time=2-0 -c 48 --job-name=GWc${x}r${rep}  --partition=cpu -o /path/to/your/project/GWAS_results/$1/rep${rep}/chro${x}.out --wrap="/path/to/your/project/plink2 --bed /path/to/your/project/training_bed_files/rep${rep}/chr${x}_X_train_no_cov_no_missing.bed --fam /path/to/your/project/training_bed_files/rep${rep}/chr${x}_X_train_no_cov_no_missing.fam --bim /path/to/your/project/training_bed_files/rep${rep}/chr${x}_X_train_no_cov_no_missing.bim --pheno /path/to/your/project/phenotypes/$pheno_file --pheno-name $pheno_name --glm hide-covar --covar /path/to/your/project/cov_matrix/rep${rep}/cov_matrix_MinMax_scaled_no_missing_1_11_23_rep${rep}.txt --out /path/to/your/project/GWAS_results/$1/rep${rep}/chr${x}_$1_cov_MinMax_scaled_with_40_PCs"

done
done
