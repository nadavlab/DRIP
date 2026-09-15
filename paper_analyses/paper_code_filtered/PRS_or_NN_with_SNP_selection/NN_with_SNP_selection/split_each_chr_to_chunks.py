import pandas as pd
import pickle5 as pickle
from pandas_plink import read_plink, read_plink1_bin
import numpy as np
import pickle
import os
import sys


def split_X_to_chunks(gene_end_file_name,input_path, gene_output_end_file_name, chunk_size,chr,pheno_name,rep):
    from_ = 0
    to_ = chunk_size
    print("in")
    print(chr)
    gene_output_file = '/path/to/your/project/NN_with_SNP_selection/splited_bed_files_for_all_chro/'+pheno_name+'/rep'+str(rep)+"/chr_" + str(chr) + gene_output_end_file_name
    if not os.path.exists(
            "/path/to/your/project/NN_with_SNP_selection/splited_bed_files_for_all_chro/"+pheno_name+'/rep'+str(rep)+"/"):
        os.makedirs(
            "/path/to/your/project/NN_with_SNP_selection/splited_bed_files_for_all_chro/"+pheno_name+'/rep'+str(rep)+"/")
        print("1created")
    else:
        print(
            "/path/to/your/project/NN_with_SNP_selection/splited_bed_files_for_all_chro/"+pheno_name+'/rep'+str(rep)+'/ exist')


    with open(gene_output_file, 'wb') as gene_file_handle:
        file_name = input_path +'chr' + str(chr) + gene_end_file_name
        print(file_name)
        G = read_plink1_bin(file_name, verbose=False)  # xarray.core.dataarray.DataArray
        while True:  # while there are still samples
            G_to_pandas = G[from_:to_, :].to_pandas()  # pandas.core.frame.DataFrame
            if G_to_pandas.empty == False:  # there's more samples
                (bim, fam, bed) = read_plink(file_name, verbose=False)  # bed is dask.array.core.Array
                snp_columns = bim['snp'].to_numpy()
                G_to_pandas.set_axis(snp_columns, axis=1, inplace=True)
                G_to_pandas = G_to_pandas.astype(np.int8)
                G_to_pandas.reset_index(level=0, inplace=True)
                G_to_pandas.rename(columns={'sample': 'FID'}, inplace=True)
                pickle.dump(G_to_pandas, gene_file_handle, protocol=4)
                from_ = to_
                to_ = to_ + chunk_size
            else:
                print("finished")
                return


def main():
    chr = sys.argv[1]
    pheno_name=sys.argv[2]
    rep=sys.argv[3]
    print (chr)
    chunk_size=1000

    #train
    #my_snapless_storage / improve_PRS / NN_with_snp_selection / training_bed_files / Height / chr10_training_filtered_snp_10_precent.bed
    split_X_to_chunks(gene_end_file_name="_training_filtered_snp_10_precent.bed",input_path="/path/to/your/project/NN_with_SNP_selection/training_bed_files/" + pheno_name+'/rep'+str(rep)+"/",
                     gene_output_end_file_name="_X_train_1k_chunks_no_missing.pkl",
                     chunk_size=chunk_size,chr=chr,pheno_name=pheno_name,rep=str(rep))
    #test
    split_X_to_chunks(gene_end_file_name="_test_filtered_snp_10_precent.bed",input_path='/path/to/your/project/NN_with_SNP_selection/test_bed_files/'+pheno_name+'/rep'+str(rep)+"/",
                     gene_output_end_file_name="_X_test_1k_chunks_no_missing.pkl",
                     chunk_size=chunk_size,chr=chr,pheno_name=pheno_name,rep=str(rep))


if __name__ == "__main__":
    main()

