#!/bin/bash
#$1 is the phenotype [the user insert Height/Hypertension/BMI/Platelet_Count...]
#$2 is the l for linear phenotypes and b for binnary phenotypes [the user insert Height/Hypertension/BMI/Platelet_Count...]
#$3 is the rep

##create the directories

#define phenotype file name 
pheno_file=$(echo $1|tr '[:upper:]' '[:lower:]')
echo $pheno_file

#define the name of the phenotype file name 

echo $(head -n1 "/path/to/your/project/phenotypes/${pheno_file}"| cut -f3 -d$' ')
pheno_name=$(head -n1 "/path/to/your/project/phenotypes/${pheno_file}"| cut -f3 -d$' ')
echo $pheno_name


if [ "$2" = "b" ]; then
    file_name_end=".glm.logistic.hybrid"
    bin_target="T"
    stat="OR"
elif [ "$2" = "l" ]; then
    file_name_end=".glm.linear"
    bin_target="F"
    stat="BETA"
fi

echo $bin_target
echo $file_name_end

Rscript /path/to/your/project/impoving_PRS/code/from_the_begining_in_bgu/PRS_or_NN_with_SNP_selection/PRSice/X.R

Rscript /home/hkaufman/PRSice.R --dir . \
--prsice /home/hkaufman/PRSice_linux \
--base /path/to/your/project/merge_GWAS_results/$1/rep$3/all_chr_train_cov_MinMax_scaled_with_values_A2.$1$file_name_end --snp ID --chr CHROM --bp POS --a1 A1 \
--a2 A2 --stat $stat --pvalue P \
--target /path/to/your/project/test_bed_files/rep$3/chr#_X_test_no_cov_no_missing \
--binary-target $bin_target \
--bar-levels 5e-08,5e-07,5e-06,5e-05,5e-04,0.001,0.05,0.1,0.2,0.3,0.4,0.5,1 \
--fastscore \
--no-clump \
--pheno /path/to/your/project/phenotypes/$pheno_file --pheno-col $pheno_name \
--cov /path/to/your/project/cov_matrix/rep$3/cov_matrix_MinMax_scaled_no_missing.txt \
--out /path/to/your/project/PRSice_results/$1/rep$3/PRS_$1_no-clump_cov_MinMax_scaled \
--print-snp \
--score sum \
--quantile 20
