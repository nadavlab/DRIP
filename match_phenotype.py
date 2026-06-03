import pandas as pd
import numpy as np
import pickle
import sys
import os

def match(x_file_name, pheno_df, x_output_file):   
    number_samples = 0
    num_batch = 1 
    with open(x_output_file, 'wb') as x_output_file_handle:
        with open(x_file_name, 'rb') as x_file_handle:
            while True:
                try:
                    X_batch = pickle.load(x_file_handle)
                    X_batch = X_batch[X_batch['FID'].isin(pheno_df['FID'])].reset_index(drop=True)
                    print('number samples in X_batch =', len(X_batch))
                    if num_batch==1:
                        print('number samples in validation =',len(X_batch))
                    num_batch = num_batch + 1
                    pickle.dump(X_batch, x_output_file_handle, protocol=4)
                    number_samples = number_samples + len(X_batch)
                except EOFError:
                    print('number_samples =',number_samples)
                    break
                
def split_y_train_to_chunks(y_output_file, pheno_df, x_chunk_file):
    with open(y_output_file, 'wb') as y_file_handle:
       with open(x_chunk_file, 'rb') as x_file_handle:
           while True:
               try:
                   X_batch = pickle.load(x_file_handle)
                   df = pd.merge(X_batch, pheno_df, on='FID') #in order to order samples in the same order as in genes
                   #print(df.head(1))
                   #print(df.iloc[:,[0,-1]])
                   y_batch = df.iloc[:,[0,-1]].reset_index(drop=True)
                   print('number samples in y_batch =', len(y_batch))
                   pickle.dump(y_batch, y_file_handle, protocol=4) 
               except EOFError:
                   break
                               
def split_test_y_to_chunks(input_df, x_chunk_file):
    pheno_df = pd.DataFrame()
    with open(x_chunk_file, 'rb') as x_file_handle:
        while True:
            try:
                X_batch = pickle.load(x_file_handle)
                df = pd.merge(X_batch, input_df, on='FID') #in order to order samples in the same order as in genes
                print(df.head(1))
                print(df.iloc[:, [0, -1]])
                y_batch =df.iloc[:,[0,-1]].reset_index(drop=True)
                pheno_df = pd.concat([pheno_df, y_batch], ignore_index=True)
            except EOFError:
                return pheno_df

pheno_file=sys.argv[1]
train_X_files=sys.argv[2]
test_X_files=sys.argv[3]
output_path=sys.argv[4]

# handle phenotypes dataframe
pheno = pd.read_csv(pheno_file, sep='\t')
if (pheno.shape[1]==1):
        pheno = pd.read_csv(pheno_file, sep=' ')
    
pheno.rename(columns={"#FID":"FID"},inplace=True)

pheno['FID']=pheno['FID'].astype(str)
pheno = pheno.iloc[:,[0,2]]
pheno = pheno.dropna(axis=0).reset_index(drop=True)

x_train_output_file = output_path+"/X_train_match_to_pheno_MinMax_cov_MinMax.pkl"
x_test_output_file = output_path+"/X_test_match_to_pheno_MinMax_cov_MinMax.pkl"

y_train_output_file = output_path+"/Y_train_match_to_X_file.pkl"
y_test_output_file = output_path+"/Y_test_match_to_X_file.pkl"

match(train_X_files, pheno_file, x_train_output_file)

#--------------------y train--------------------
split_y_train_to_chunks(y_train_output_file, pheno, x_train_output_file)
#----------------------------------------

print('test')
match(test_X_files, pheno, x_test_output_file)

#--------------------y test--------------------
test_pheno_df = split_test_y_to_chunks(pheno, x_test_output_file)
test_pheno_df = test_pheno_df.reset_index(drop=True)
test_pheno_df.to_pickle(y_test_output_file)
#----------------------------------------
