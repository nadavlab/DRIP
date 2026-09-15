F#!/bin/bash
#$1 is the phenotype [the user insert Height/Hypertension/BMI/Platelet_Count...]

##create the directories
mkdir /path/to/your/project/PRSice_results/
mkdir /path/to/your/project/PRSice_results/$1/



for rep in {1..5}
do
echo $rep

mkdir /path/to/your/project/PRSice_results/$1/rep${rep}/

for chro in {1..22}
do

sbatch --mem=100g --time=1-0 -c 48 --job-name=PRScs$1r${rep} -o /path/to/your/project/PRScs/PRScs_results/$1/rep${rep}/chr_${chro}_PRScs.out --wrap="python /path/to/your/project/impoving_PRS/PRScs/PRScs/PRScs.py --ref_dir=/path/to/your/project/impoving_PRS/PRScs/ldblk_ukbb_eur --bim_prefix=/path/to/your/project/training_bed_files/rep${rep}/chr${chro}_X_train_no_cov_no_missing --sst_file=/path/to/your/project/PRScs/SS_formated_data/$1/rep${rep}/GWAS_results_PRScs_formatted/chr${chro}.tsv --n_gwas=290688 --out_dir=/path/to/your/project/PRScs/PRScs_results/$1/rep${rep}/PRScs_chr${chro}_$1_results --chrom=${chro} --seed=3"

done
done
