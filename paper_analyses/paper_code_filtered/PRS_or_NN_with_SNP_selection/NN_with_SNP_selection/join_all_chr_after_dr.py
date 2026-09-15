import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys
import pickle5 as pickle


def join(X_file_name_beginning, chunk_size, gene_output_file, cov_df,X_file_name_end):
    number_samples = 0 
    with open(gene_output_file, 'wb') as gene_output_file_handle:
        this_chunk = 1
        while True:
            try:
                for chr in range(1, 23):
                    dim_remove_file = X_file_name_beginning + str(chr)
                    dim_remove_file = dim_remove_file +  X_file_name_end
                    with open(dim_remove_file, 'rb') as dim_remove_file_handle:
                        for c in range(1,this_chunk+1):
                            batch = pickle.load(dim_remove_file_handle)
                        ID = batch['FID']
                        batch = batch.drop(['FID'], axis=1)
                        suffix = '_chr'+str(chr)
                        batch = batch.add_suffix(suffix)
                        batch['FID'] = ID
                        if chr == 1:
                            df = pd.merge(cov_df, batch, on='FID')
                        else:
                            df = pd.merge(df, batch, on='FID')
                        
                pickle.dump(df, gene_output_file_handle, protocol=4)
                this_chunk = this_chunk + 1
                number_samples = number_samples + len(df)
            except EOFError:
                print('number_samples after match cov =',number_samples)
                break


                
 
# handle covariate
pheno_name=sys.argv[1]
rep=sys.argv[2]


with open('/path/to/your/project/cov_matrix/rep'+rep
          +'/cov_matrix_MinMax_scaled_no_missing.pkl', "rb") as fh:
  cov_df = pickle.load(fh)
#cov_df = pd.read_pickle('/path/to/your/project//cov_matrix_MinMax_scaled_no_missing.pkl')
cov_df.rename(columns={"#FID":"FID"},inplace=True)
print(cov_df.head(1))
chunk_size=1000
X_file_name_beginning = '/path/to/your/project/NN_with_SNP_selection/splited_bed_files_for_all_chro/'+pheno_name+'/rep'+rep+"/chr_"
train_X_file_name_end="_X_train_1k_chunks_no_missing.pkl"
test_X_file_name_end="_X_test_1k_chunks_no_missing.pkl"

#train
union_train_gene_output_file = "/path/to/your/project/NN_with_SNP_selection/X_files/rep"+rep+"/"+pheno_name+"_X_train_all_chr_MinMax_cov_MinMax.pkl"
print('train')
join(X_file_name_beginning, chunk_size, union_train_gene_output_file, cov_df,train_X_file_name_end)

#test
union_test_gene_output_file = "/path/to/your/project/NN_with_SNP_selection/X_files/rep"+rep+"/"+pheno_name+"_X_test_all_chr_MinMax_cov_MinMax.pkl"
print('test')
join(X_file_name_beginning, chunk_size, union_test_gene_output_file, cov_df,test_X_file_name_end)
