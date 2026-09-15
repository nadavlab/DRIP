#!/bin/bash

SCRIPT="/path/to/your/project//code/from_the_begining_in_bgu/simulations/run_LR_for_all_simulated_pheno.py"

MATCHED_DIR="/path/to/your/project//simulations/matched_data_ids"
OUTPUT_ROOT="/path/to/your/project//simulations/LR_results"

PHENO_FILE="${OUTPUT_ROOT}/phenotypes.txt"

mkdir -p logs
mkdir -p "${OUTPUT_ROOT}/metrics_by_pheno"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=LR_parallel
#SBATCH --output=logs/LR_%A_%a.out
#SBATCH --error=logs/LR_%A_%a.err
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=4-0:00:00
#SBATCH --array=0-94%10

source ~/.bashrc
conda activate pandas-env-3

SCRIPT="${SCRIPT}"
MATCHED_DIR="${MATCHED_DIR}"
OUTPUT_ROOT="${OUTPUT_ROOT}"
PHENO_FILE="${PHENO_FILE}"

# We skip the first phenotype, so this file should have 19 phenotypes.
# If phenotypes.txt still has all 20, this command starts from line 2.
N_PHENOS=19

TASK_ID=\${SLURM_ARRAY_TASK_ID}

REP=\$(( TASK_ID / N_PHENOS + 1 ))
PHENO_INDEX=\$(( TASK_ID % N_PHENOS + 2 ))

PHENO_TEMPLATE=\$(sed -n "\${PHENO_INDEX}p" "\${PHENO_FILE}")
PHENO=\${PHENO_TEMPLATE/rep1_/rep\${REP}_}

echo "SLURM task: \${TASK_ID}"
echo "Running rep=\${REP}"
echo "Phenotype index in file=\${PHENO_INDEX}"
echo "Phenotype=\${PHENO}"

python "\${SCRIPT}" \
    --rep "\${REP}" \
    --x-train "\${MATCHED_DIR}/X_train_match_to_pheno_rep_\${REP}.pkl" \
    --y-train "\${MATCHED_DIR}/Y_train_all_rep_\${REP}.pkl" \
    --x-test "\${MATCHED_DIR}/X_test_match_to_pheno_rep_\${REP}.pkl" \
    --y-test "\${MATCHED_DIR}/Y_test_all_rep_\${REP}.pkl" \
    --output-root "\${OUTPUT_ROOT}" \
    --metrics-file "\${OUTPUT_ROOT}/metrics_by_pheno/rep_\${REP}_\${PHENO}.csv" \
    --pheno-cols "\${PHENO}" \
    --skip-existing

EOF