#!/bin/bash


for rep in {5..5}
do


        sbatch --mem=28g --time=1-00:00:00 -c 10 --job-name=scl${rep} -o //path/to/your/project/our_model/Autoencoder/X_test_1k_chunks_dim_remove_no_missing_500_epochs/rep${rep}/scale_genes.out --wrap="python3 /path/to/your/project/impoving_PRS/code/from_the_begining_in_bgu/our_model/scale_genes.py ${rep} Autoencoder
 ${rep}"

done
