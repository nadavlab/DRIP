#!/bin/bash

#SBATCH --partition=rtx6000
#SBATCH --qos=nadavrap
#SBATCH --time=6-10:30:00
#SBATCH --job-name=NN_GxG
#SBATCH --output=/path/to/your/project//simulations/GxG_NN_results/NN_GxG_%A_%a.out
#SBATCH --mail-user=/yourmailadress
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus=1
#SBATCH --mem=58G
#SBATCH --array=1-14

echo "Date: $(date)"
echo "JobID: $SLURM_JOBID"
echo "TaskID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_JOB_NODELIST"

module load anaconda
source activate py-tf-gpu

#############################################
# Determine repetition and phenotype
#############################################

N_PHENOS=1
TASK=${SLURM_ARRAY_TASK_ID}

REP=$(( TASK / N_PHENOS + 1 ))
PHENO_INDEX=$(( TASK % N_PHENOS ))

PHENOS=(
#"rep${REP}_chr1_N100_GI_00_h2add_0p30_h2gxg_0p00_cov_0p200_noise_0p500"
#"rep${REP}_chr1_N100_GI_05_h2add_0p25_h2gxg_0p05_cov_0p200_noise_0p500"
#"rep${REP}_chr1_N100_GI_10_h2add_0p20_h2gxg_0p10_cov_0p200_noise_0p500"
#"rep${REP}_chr1_N100_GI_15_h2add_0p15_h2gxg_0p15_cov_0p200_noise_0p500"
"rep${REP}_chr1_N100_GI_30_h2add_0p00_h2gxg_0p20_cov_0p200_noise_0p500"
)

PHENO=${PHENOS[$PHENO_INDEX]}

echo "======================================"
echo "Running repetition : ${REP}"
echo "Running phenotype  : ${PHENO}"
echo "======================================"

#############################################
# Paths
#############################################

NN_SCRIPT="/path/to/your/project//code/from_the_begining_in_bgu/simulations/run_LR_all_simulated_pheno_with_chr1_PCs_only.py"

MATCHED_ROOT="/path/to/your/project//simulations/matched_data_ids"

OUT_ROOT="/path/to/your/project//simulations/GxG_NN_results"

METRICS_FILE="${OUT_ROOT}/NN_results.csv"

mkdir -p "${OUT_ROOT}/rep${REP}"

#############################################
# GPU information
#############################################

nvidia-smi

#############################################
# Run NN
#############################################

/home/hkaufman/.conda/envs/py-tf-gpu/bin/python3 "${NN_SCRIPT}" \
    --rep "${REP}" \
    --matched-root "${MATCHED_ROOT}" \
    --output-root "${OUT_ROOT}" \
    --metrics-file "${METRICS_FILE}" \
    --pheno-cols "${PHENO}" \
    --max-epochs 1000 \
    --batch-size 50 \
    --learning-rate 1e-7 \
    --dropout 0.1 \
    --patience 5

echo "Finished at $(date)"