#!/bin/bash
#$1 is the phenotype [the user insert Height/Hypertension/BMI/Platelet_Count...]
#$2 is the l for linear phenotypes and b for binnary phenotypes [the user insert Height/Hypertension/BMI/Platelet_Count...]
#$3 is the rep

##create the directories

#define phenotype file name 
pheno_file=$(echo $1|tr '[:upper:]' '[:lower:]')
echo $pheno_file

#define the name of the phenotype file name 

echo $(head -n1 "/path/to/your/project/phenotypes/${pheno_file}"| cut -f3 -d$'\t')
pheno_name=$(head -n1 "/path/to/your/project/phenotypes/${pheno_file}"| cut -f3 -d$' ')
echo $pheno_name


if [ "$2" = "b" ]; then
    file_name_end=".glm.logistic.hybrid"
    bin_target="T"
elif [ "$2" = "l" ]; then
    file_name_end=".glm.linear"
    bin_target="F"
fi

echo $bin_target

echo $"bin_terget"


Rscript /path/to/your/project/impoving_PRS/code/from_the_begining_in_bgu/PRS_or_NN_with_SNP_selection/PRSice/X.R