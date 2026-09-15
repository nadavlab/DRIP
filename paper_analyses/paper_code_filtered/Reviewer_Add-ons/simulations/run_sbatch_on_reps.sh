#!/bin/bash

SCRIPT="/path/to/your/project//code/from_the_begining_in_bgu/simulations/match_data_to_pheno_IDs.py"

for rep in {1..5}
do

    TRAIN_X="/path/to/your/project//our_model/PCA/X_train_1k_chunks_PCA_dim_remove_no_missing/rep${rep}/X_train_MinMax_cov_MinMax.pkl"

    TEST_X="/path/to/your/project//our_model/PCA/X_test_1k_chunks_PCA_dim_remove_no_missing/rep${rep}/X_test_MinMax_cov_MinMax.pkl"

    OUTPUT="/path/to/your/project//simulations/matched_data/rep${rep}"

    mkdir -p "${OUTPUT}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=match_rep${rep}
#SBATCH --output=logs/match_rep${rep}.out
#SBATCH --error=logs/match_rep${rep}.err
#SBATCH --time=2-02:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G

source ~/.bashrc
conda activate pandas-env-3

python ${SCRIPT} \
    ${rep} \
    ${TRAIN_X} \
    ${TEST_X} \
    ${OUTPUT}

EOF

done