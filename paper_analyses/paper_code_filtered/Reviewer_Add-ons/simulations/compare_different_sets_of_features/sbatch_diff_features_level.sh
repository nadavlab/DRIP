#!/bin/bash

SCRIPT="/path/to/your/project//code/from_the_begining_in_bgu/simulations/run_LR_with_different_features_levels_simulation.py"

LOG_DIR="/path/to/your/project//simulations/LR_feature_sets/logs"
mkdir -p "${LOG_DIR}"

# <<< replace with the name of the baseline phenotype column >>>

for rep in {1..5}
do

PHENO="rep${rep}_chr1_N100_GI_00_h2add_0p30_h2gxg_0p00_cov_0p200_noise_0p500"

sbatch <<EOF
#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=5-24:00:00
#SBATCH --job-name=LRFS_r${rep}
#SBATCH --output=${LOG_DIR}/LRFS_rep${rep}_%J.out
#SBATCH --error=${LOG_DIR}/LRFS_rep${rep}_%J.err
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --mail-user=/yourmailadress
#SBATCH --mail-type=END,FAIL

echo \`date\`
echo "SLURM_JOBID: \$SLURM_JOBID"
echo "NODE: \$SLURM_JOB_NODELIST"

module load anaconda
source activate pandas-env-3

python ${SCRIPT} \
    --rep ${rep} \
    --pheno-col ${PHENO}
    --ukb-pc-cols-file 

EOF

done