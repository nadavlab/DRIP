#!/bin/bash
#$1 is the phenotype [the user insert Height/Hypertension/BMI/Platelet_Count...]
#$2 is the l for linear phenotypes and b for binnary phenotypes [the user insert Height/Hypertension/BMI/Platelet_Count...]

##create the directories
mkdir /path/to/your/project/PRSice_results/
mkdir /path/to/your/project/PRSice_results/$1/

#define phenotype file name 
pheno_file=$(echo $1|tr '[:upper:]' '[:lower:]')
echo $pheno_file

#define the name of the phenotype file name 

echo $(head -n1 "/path/to/your/project/phenotypes/${pheno_file}"| cut -f3 -d$' ')
pheno_name=$(head -n1 "/path/to/your/project/phenotypes/${pheno_file}"| cut -f3 -d$' ')
echo $pheno_name


if ["$2"=="b"];
then
	file_name_end=".glm.logistic.hybrid"
else
if ["$2"=="l"];
then
	file_name_end=".glm.linear"
fi

for rep in {1..5}
do
echo $rep

mkdir /path/to/your/project/PRSice_results/$1/rep${rep}/

Rscript /path/to/your/project//apps/PRSice.R --dir . \
--prsice /path/to/your/project//apps/PRSice_linux \
--base /path/to/your/project/merge_GWAS_results/$1/rep${rep}/all_chr_train_cov_MinMax_scaled_with_values_A2.$1$file_name_end --snp ID --chr CHROM --bp POS --a1 A1 \
--a2 A2 --stat OR --pvalue P \
--target /path/to/your/project/test_bed_files/rep${rep}/chr#_X_test_no_cov_no_missing \
--binary-target F \
--bar-levels 5e-08,5e-07,5e-06,5e-05,5e-04,0.001,0.05,0.1,0.2,0.3,0.4,0.5,1 \
--fastscore \
--no-clump \
--pheno /path/to/your/project/phenotypes/$pheno_file --pheno-col $pheno_name \
--cov /path/to/your/project/cov_matrix/rep${rep}/cov_matrix_MinMax_scaled_no_missing.txt \
--out /path/to/your/project/PRSice_results/$1/rep${rep}/without_clump/PRS_hypertension_no-clump_cov_MinMax_scaled \
--print-snp \
--score sum \
--quantile 20