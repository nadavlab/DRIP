#!/bin/bash

SCRIPT="/path/to/your/project//code/from_the_begining_in_bgu/simulations/run_LR_all_simulated_pheno_with_chr1_PCs_only.py"

MATCHED_DIR="/path/to/your/project//simulations/matched_data_ids"

# New output directory
OUTPUT_ROOT="/path/to/your/project//simulations/LR_results_chr1_only"

PHENO_FILE="${OUTPUT_ROOT}/phenotypes.txt"

mkdir -p logs
mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}/metrics_by_pheno"

# Copy the phenotype list if it doesn't already exist
cp /path/to/your/project//simulations/LR_results/phenotypes.txt \
   "${PHENO_FILE}"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=LR_chr1
#SBATCH --output=logs/LR_chr1_%A_%a.out
#SBATCH --error=logs/LR_chr1_%A_%a.err
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=4-0:00:00
#SBATCH --array=0-109%37

source ~/.bashrc
conda activate pandas-env-3

SCRIPT="${SCRIPT}"
MATCHED_DIR="${MATCHED_DIR}"
OUTPUT_ROOT="${OUTPUT_ROOT}"
PHENO_FILE="${PHENO_FILE}"

N_PHENOS=20

TASK_ID=\${SLURM_ARRAY_TASK_ID}

REP=\$(( TASK_ID / N_PHENOS + 1 ))
PHENO_INDEX=\$(( TASK_ID % N_PHENOS + 1 ))

PHENO_TEMPLATE=\$(sed -n "\${PHENO_INDEX}p" "\${PHENO_FILE}")
PHENO=\${PHENO_TEMPLATE/rep1_/rep\${REP}_}

echo "SLURM task: \${TASK_ID}"
echo "Running repetition: \${REP}"
echo "Phenotype index: \${PHENO_INDEX}"
echo "Phenotype: \${PHENO}"

python "\${SCRIPT}" \
    --rep "\${REP}" \
    --x-train "\${MATCHED_DIR}/X_train_match_to_pheno_rep_\${REP}.pkl" \
    --y-train "\${MATCHED_DIR}/Y_train_all_rep_\${REP}.pkl" \
    --x-test "\${MATCHED_DIR}/X_test_match_to_pheno_rep_\${REP}.pkl" \
    --y-test "\${MATCHED_DIR}/Y_test_all_rep_\${REP}.pkl" \
    --output-root "\${OUTPUT_ROOT}" \
    --metrics-file "\${OUTPUT_ROOT}/metrics_by_pheno/rep_\${REP}_\${PHENO}.csv" \
    --pheno-cols "\${PHENO}" \

EOF

#    --skip-existing
