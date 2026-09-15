#!/bin/bash

#SBATCH --partition=rtx6000
#SBATCH --qos=nadavrap
#SBATCH --time=6-10:30:00
#SBATCH --job-name=NN_GI30
#SBATCH --output=/path/to/your/project//simulations/GxG_NN_results/NN_GI30_%A_%a.out
#SBATCH --error=/path/to/your/project//simulations/GxG_NN_results/NN_GI30_%A_%a.err
#SBATCH --mail-user=/yourmailadress
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=58G
#SBATCH --array=1-5

# Exit on errors and failed pipelines.
# Do not use set -u before activating Conda.
set -eo pipefail

#############################################
# Activate Conda environment
#############################################

module load anaconda

set +u
source activate py-tf-gpu
set -u

# Use Python from the activated environment.
PYTHON="/home/hkaufman/.conda/envs/py-tf-gpu/bin/python"

echo "Activated Conda environment: ${CONDA_DEFAULT_ENV:-unknown}"
echo "Python executable: ${PYTHON}"
"${PYTHON}" --version

#############################################
# Job information
#############################################

echo "======================================"
echo "Start date : $(date)"
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Array ID   : ${SLURM_ARRAY_TASK_ID}"
echo "Node       : ${SLURM_JOB_NODELIST}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "======================================"

#############################################
# Repetition and phenotype
#############################################

#############################################
# Repetition and phenotypes
#############################################

REP="${SLURM_ARRAY_TASK_ID}"

PHENO_ARRAY=(
    "rep${REP}_chr1_N100_GI_00_h2add_0p30_h2gxg_0p00_cov_0p200_noise_0p500"
    "rep${REP}_chr1_N100_GI_05_h2add_0p25_h2gxg_0p05_cov_0p200_noise_0p500"
    "rep${REP}_chr1_N100_GI_10_h2add_0p20_h2gxg_0p10_cov_0p200_noise_0p500"
    "rep${REP}_chr1_N100_GI_15_h2add_0p15_h2gxg_0p15_cov_0p200_noise_0p500"
    "rep${REP}_chr1_N100_GI_20_h2add_0p10_h2gxg_0p20_cov_0p200_noise_0p500"
)

# Join the phenotype names with commas because --pheno-cols
# expects a comma-separated string.
PHENOS=$(IFS=,; echo "${PHENO_ARRAY[*]}")

echo "Running repetition: ${REP}"
echo "Number of phenotypes: ${#PHENO_ARRAY[@]}"
echo "Phenotypes:"
printf '  %s\n' "${PHENO_ARRAY[@]}"

echo "Comma-separated argument:"
echo "${PHENOS}"

#############################################
# Paths
#############################################

NN_SCRIPT="/path/to/your/project//code/from_the_begining_in_bgu/simulations/NN_for_all_simulated_phenotypes_chro1_only.py"

MATCHED_ROOT="/path/to/your/project//simulations/matched_data_ids"

OUT_ROOT="/path/to/your/project//simulations/GxG_NN_results"

METRICS_FILE="${OUT_ROOT}/NN_results.csv"

mkdir -p "${OUT_ROOT}/rep${REP}"

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
print("Visible GPUs:", tf.config.list_physical_devices("GPU"))

if not tf.config.list_physical_devices("GPU"):
    raise RuntimeError(
        "TensorFlow cannot detect a GPU. "
        "Check the CUDA installation and the TensorFlow environment."
    )
PY

#############################################
# Run model
#############################################

echo "======================================"
echo "Starting neural-network model"
echo "======================================"

"${PYTHON}" "${NN_SCRIPT}" \
    --rep "${REP}" \
    --matched-root "${MATCHED_ROOT}" \
    --output-root "${OUT_ROOT}" \
    --metrics-file "${METRICS_FILE}" \
    --pheno-cols "${PHENOS}" \
    --max-epochs 1000 \
    --batch-size 50 \
    --learning-rate 1e-7 \
    --dropout 0.1 \
    --patience 5

echo "======================================"
echo "Finished successfully at $(date)"
echo "======================================"