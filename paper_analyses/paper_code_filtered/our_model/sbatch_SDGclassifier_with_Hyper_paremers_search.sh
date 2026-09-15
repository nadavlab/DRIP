#!/bin/bash
##$1 is the phenotype
##$2 is l for linear phenotype and b for binary phenotype
##$3 is the dimension reduction model 


if [ "$2" == "l" ]; then
  for rep in {1..5}
  do
	echo "pheno $1"
	echo "type $2"
	echo "rep ${rep}"

    mkdir "/path/to/your/project/linear_regression_$3"
    mkdir "/path/to/your/project/linear_regression_$3/$1"

    mkdir "/path/to/your/project/linear_regression_$3/$1/rep${rep}/"

    sbatch --mem=200g --time=1-0 -c 50 --job-name=lr$1rep${rep} --mail-user=/yourmailadress --mail-type=BEGIN,END,FAIL -o /path/to/your/project/linear_regression_$3/$1/rep${rep}/lr_1e-5_%J.out --wrap="/home/hkaufman/.conda/envs/xgb-cpu-env/bin/python3 /path/to/your/project/impoving_PRS/code/from_the_begining_in_bgu/our_model/linear_regression_with_pca_data.py $1 ${rep} $3"
  done
elif [ "$2" == "b" ]; then
    mkdir "/path/to/your/project/SDGclassifier_with_best_hp/"
    mkdir "/path/to/your/project/SDGclassifier_with_best_hp/$1/"
  
for rep in {1..2}
  do
    mkdir "/path/to/your/project/SDGclassifier_with_best_hp/$1/rep${rep}/"

    mkdir "/path/to/your/project/logistic_regression_$3/$1/rep${rep}/"

    sbatch --mem=400g --time=4-0 -c 100 --job-name=lr$1rep${rep} --mail-user=/yourmailadress --mail-type=BEGIN,END,FAIL -o /path/to/your/project/SDGclassifier_with_best_hp/$1/rep${rep}/best_hyper_paremeters_SGDClassifier_rep_${rep}%J.out --wrap="python3 /path/to/your/project/impoving_PRS/code/from_the_begining_in_bgu/our_model/hyper_parameters_serch.py $1 ${rep} $3"
  done
else
  echo "you must insert l/b for linear/binary phenotype"
fi
