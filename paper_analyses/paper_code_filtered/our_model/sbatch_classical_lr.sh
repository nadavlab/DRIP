#!/bin/bash

# Arguments:
# $1 - phenotype name
# $2 - model type: "l" for linear, "b" for binary/logistic
# $3 - DRM method (e.g., "raw", "pca", etc.)

if [ "$2" != "l" ] && [ "$2" != "b" ]; then
  echo "You must insert 'l' or 'b' for linear or binary phenotype"
  exit 1
fi

BASE_PATH="/path/to/your/project/impoving_PRS"
PHENO=$1
MODEL_TYPE=$2
DRM=$3

OUT_DIR="${BASE_PATH}/data/classical_lr_${DRM}/${PHENO}"

mkdir -p "$OUT_DIR"

for rep in {1..5}; do
  echo "pheno $PHENO"
  echo "type $MODEL_TYPE"
  echo "rep ${rep}"

  REP_DIR="${OUT_DIR}/rep${rep}"
  mkdir -p "$REP_DIR"

  sbatch --mem=400g --time=2-0 -c 100 \
      --job-name=lr${PHENO}rep${rep} \
      --mail-user=/yourmailadress --mail-type=BEGIN,END,FAIL \
      -o "${REP_DIR}/${PHENO}_${rep}_Ridge.out" \
      --wrap="/home/hkaufman/.conda/envs/xgb-cpu-env/bin/python3 ${BASE_PATH}/code/from_the_begining_in_bgu/our_model/Classical_lr_with_pca_data.py ${PHENO} ${rep} ${DRM} ${MODEL_TYPE} "

done

