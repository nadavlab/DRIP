#first trying with rep1 chro1 hypertension 
#$1 is the phenotype

python /path/to/your/project/impoving_PRS/PRScs/PRScs/PRScs.py --ref_dir=/path/to/your/project/impoving_PRS/PRScs/ldblk_ukbb_eur --bim_prefix=/path/to/your/project/test_bed_files/rep1/chr1_X_test_no_cov_no_missing --sst_file=/path/to/your/project/PRScs/SS_formated_data/Hypertension/rep1/GWAS_results_PRScs_formatted/chr1.tsv --n_gwas=290688 --out_dir=/path/to/your/project/PRScs/PRScs_results/Hypertension/rep1/PRScs_chr1_Hypertension_results --chrom=1 --seed=3