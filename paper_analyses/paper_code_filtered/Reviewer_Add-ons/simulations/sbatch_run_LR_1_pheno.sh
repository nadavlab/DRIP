#!/bin/bash

SCRIPT="/path/to/your/project//code/from_the_begining_in_bgu/simulations/run_LR_for_all_simulated_pheno.py"

MATCHED_DIR="/path/to/your/project//simulations/matched_data_ids"

OUTPUT_ROOT="/path/to/your/project//simulations/LR_results_chr1_only"

mkdir -p logs
mkdir -p "${OUTPUT_ROOT}/metrics_by_pheno"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=LR_one_pheno
#SBATCH --output=logs/LR_one_pheno_%A_%a.out
#SBATCH --error=logs/LR_one_pheno_%A_%a.err
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=4-0:00:00
#SBATCH --array=1-5%5

source ~/.bashrc
conda activate pandas-env-3

SCRIPT="${SCRIPT}"
MATCHED_DIR="${MATCHED_DIR}"
OUTPUT_ROOT="${OUTPUT_ROOT}"

REP=\${SLURM_ARRAY_TASK_ID}

# Write the phenotype name using rep1 as the template.
PHENO_TEMPLATE="rep1_chr1_N100_GI_30_h2add_0p00_h2gxg_0p30_cov_0p200_noise_0p500"

# Replace rep1_ with the current repetition.
PHENO=\${PHENO_TEMPLATE/rep1_/rep\${REP}_}

echo "SLURM task: \${SLURM_ARRAY_TASK_ID}"
echo "Running repetition: \${REP}"
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
    --skip-existing

EOF