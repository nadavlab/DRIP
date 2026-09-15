#!/bin/bash

#SBATCH --partition=rtx6000
#SBATCH --qos=nadavrap
#SBATCH --time=6-10:30:00
#SBATCH --job-name=NN_GI
#SBATCH --output=/path/to/your/project//simulations/GxG_NN_results_UKB_settings/logs/NN_GI_%A_%a.out
#SBATCH --error=/path/to/your/project//simulations/GxG_NN_results_UKB_settings/logs/NN_GI_%A_%a.err
#SBATCH --mail-user=/yourmailadress
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=58G

# 3 repetitions × 4 phenotypes = 12 independent jobs.
# At most 12 jobs will run simultaneously.
##SBATCH --array=0-11%12
# 3 repetitions × 1 phenotype = 3 independent jobs.
#SBATCH --array=0-2%3

set -eo pipefail

#############################################
# Activate Conda environment
#############################################

module load anaconda

# Avoid errors from Conda activation scripts when nounset is enabled.
set +u
source activate py-tf-gpu
set -u

PYTHON="/home/hkaufman/.conda/envs/py-tf-gpu/bin/python"

echo "Activated Conda environment: ${CONDA_DEFAULT_ENV:-unknown}"
echo "Python executable: ${PYTHON}"
"${PYTHON}" --version


#############################################
# Map array task to repetition and phenotype
#############################################

TASK_ID="${SLURM_ARRAY_TASK_ID}"

# One phenotype per repetition:
# task 0 -> rep1
# task 1 -> rep2
# task 2 -> rep3
REP=$(( TASK_ID + 1 ))

GI_SETTING="GI_30_h2add_0p00_h2gxg_0p30"
PHENO_INDEX=0

PHENO="rep${REP}_chr1_N100_${GI_SETTING}_cov_0p200_noise_0p500"

echo "======================================"
echo "Start date          : $(date)"
echo "SLURM job ID        : ${SLURM_JOB_ID}"
echo "SLURM array task    : ${SLURM_ARRAY_TASK_ID}"
echo "Node                : ${SLURM_JOB_NODELIST}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Repetition          : ${REP}"
echo "Phenotype index     : ${PHENO_INDEX}"
echo "Phenotype           : ${PHENO}"
echo "======================================"

#############################################
# Paths
#############################################

NN_SCRIPT="/path/to/your/project//code/from_the_begining_in_bgu/simulations/NN_for_all_simulated_phenotypes_chro1_only.py"

MATCHED_ROOT="/path/to/your/project//simulations/matched_data_ids"

# New output directory, so previous results are not overwritten.
OUT_ROOT="/path/to/your/project//simulations/GxG_NN_results_UKB_settings"

# Use one metrics file per job.
# This prevents 25 parallel jobs from writing to the same CSV simultaneously.
METRICS_DIR="${OUT_ROOT}/metrics_by_pheno"
METRICS_FILE="${METRICS_DIR}/rep${REP}_${GI_SETTING}.csv"

mkdir -p "${OUT_ROOT}/rep${REP}"
mkdir -p "${METRICS_DIR}"

#############################################
# Validate inputs
#############################################

if [[ ! -f "${NN_SCRIPT}" ]]; then
    echo "ERROR: Python script not found:"
    echo "${NN_SCRIPT}"
    exit 1
fi

if [[ ! -d "${MATCHED_ROOT}" ]]; then
    echo "ERROR: Matched-data directory not found:"
    echo "${MATCHED_ROOT}"
    exit 1
fi

X_TRAIN="${MATCHED_ROOT}/X_train_match_to_pheno_rep_${REP}.pkl"
Y_TRAIN="${MATCHED_ROOT}/Y_train_all_rep_${REP}.pkl"
X_TEST="${MATCHED_ROOT}/X_test_match_to_pheno_rep_${REP}.pkl"
Y_TEST="${MATCHED_ROOT}/Y_test_all_rep_${REP}.pkl"

for INPUT_FILE in \
    "${X_TRAIN}" \
    "${Y_TRAIN}" \
    "${X_TEST}" \
    "${Y_TEST}"
do
    if [[ ! -f "${INPUT_FILE}" ]]; then
        echo "ERROR: Required input file not found:"
        echo "${INPUT_FILE}"
        exit 1
    fi
done

#############################################
# Check TensorFlow and GPU
#############################################

echo "======================================"
echo "NVIDIA GPU information"
echo "======================================"

nvidia-smi

echo "======================================"
echo "TensorFlow information"
echo "======================================"

"${PYTHON}" - <<'PY'
import sys
import tensorflow as tf

print("Python executable:", sys.executable)
print("TensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")
print("Visible GPUs:", gpus)

if not gpus:
    raise RuntimeError(
        "TensorFlow cannot detect a GPU. "
        "Check the CUDA installation and TensorFlow environment."
    )
PY

#############################################
# Run one phenotype in this array task
#############################################

echo "======================================"
echo "Starting neural-network model"
echo "======================================"

"${PYTHON}" "${NN_SCRIPT}" \
    --rep "${REP}" \
    --matched-root "${MATCHED_ROOT}" \
    --output-root "${OUT_ROOT}" \
    --metrics-file "${METRICS_FILE}" \
    --pheno-cols "${PHENO}" \
    --max-epochs 80 \
    --batch-size 50 \
    --learning-rate 1e-7 \
    --dropout 0.1 \
    --patience 5 \
    --min-delta 1e-4

echo "======================================"
echo "Finished successfully at $(date)"
echo "Metrics file: ${METRICS_FILE}"
echo "======================================"