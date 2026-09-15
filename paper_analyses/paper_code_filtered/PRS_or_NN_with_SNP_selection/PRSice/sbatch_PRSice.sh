#!/bin/bash
#$1 is the phenotype [the user insert Height/Hypertension/BMI/Platelet_Count...]
#$2 is the l for linear phenotypes and b for binnary phenotypes [the user insert Height/Hypertension/BMI/Platelet_Count...]

##create the directories
mkdir /path/to/your/project/PRSice_results/
mkdir /path/to/your/project/PRSice_results/$1/



for rep in {1..5}
do
echo $rep

mkdir /path/to/your/project/PRSice_results/$1/rep${rep}/

sbatch --mem=100g --time=1-0 -c 50 --job-name=PRSc$1r${rep} -o /path/to/your/project/PRSice_results/$1/rep${rep}/PRSice.out --wrap="bash /path/to/your/project/impoving_PRS/code/from_the_begining_in_bgu/PRS_or_NN_with_SNP_selection/PRSice/run_PRSice_for_test.sh $1 $2 ${rep}"

done
