import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys
import pickle5 as pickle
from os import walk



def join(input_files_dir, chunk_size, output_file_name, cov_df):
    number_samples = 0 
    first_chro=True
    filenames = next(walk(input_files_dir), (None, None, []))[2]  # [] if no file
    with open(output_file_name, 'wb') as gene_output_file_handle:
        this_chunk = 1
        while True:
            try:
                for file in filenames:
                    print(file)
                    with open(file, 'rb') as chr_file_handle:
                        for c in range(1,this_chunk+1):
                            batch = pickle.load(chr_file_handle)
                        ID = batch['FID']
                        batch = batch.drop(['FID'], axis=1)
                        suffix = '_chr'+str(chr)
                        batch = batch.add_suffix(suffix)
                        batch['FID'] = ID
                        if first_chro == True:
                            df = pd.merge(cov_df, batch, on='FID')
                        else:
                            df = pd.merge(df, batch, on='FID')
                            first_chro = False
                        
                pickle.dump(df, gene_output_file_handle, protocol=4)
                this_chunk = this_chunk + 1
                number_samples = number_samples + len(df)
            except EOFError:
                print('error: '+e)
                print('number_samples after match cov =',number_samples)
                break


                
 
# handle covariate
cov_file=sys.argv[1]
input_files_dir=sys.argv[2]
output_file_name=sys.argv[3]


with open(cov_file, "rb") as fh:
    cov_df = pickle.load(fh)
    cov_df.rename(columns={"#FID":"FID"},inplace=True)
    print(cov_df.head(1))
    chunk_size=1000

    join(input_files_dir, chunk_size, output_file_name, cov_df)
    
