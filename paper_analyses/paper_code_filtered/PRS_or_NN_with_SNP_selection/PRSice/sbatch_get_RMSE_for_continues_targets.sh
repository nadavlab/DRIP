#$1 is the pheno
#$2 is the rep

sbatch --out /path/to/your/project/PRSice_results/$1/rep$2/RMSE.out --wrap="python3 /path/to/your/project/impoving_PRS/code/from_the_begining_in_bgu/PRS_or_NN_with_SNP_selection/PRSice/get_RMSE_for_continues_targets.py $1 $2"
