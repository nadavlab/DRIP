import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
import os
import pickle
from pandas_plink import read_plink
import pandas as pd
import numpy as np
import time
from sklearn.metrics import r2_score
import sys

def PCA_transform_incremental(PCA_transformer, X_chunks_file, output_file):
    full_data = pd.DataFrame()
    full_data_after_dr = pd.DataFrame()
    with open(output_file, 'wb') as save_file_handle:
        with open(X_chunks_file, 'rb') as file_handle:
            while True:
                try:
                    batch = pickle.load(file_handle)
                    ID = batch['FID']
                    batch = batch.drop(['FID'], axis=1)
                    full_data = full_data.append(batch)
                    batch_PCA = PCA_transformer.transform(batch)
                    batch_PCA_df = pd.DataFrame(batch_PCA)
                    full_data_after_dr = full_data_after_dr.append(batch_PCA_df)
                    batch_PCA_df['FID'] = ID
                    pickle.dump(batch_PCA_df, save_file_handle, protocol=4)
                except EOFError:
                    return full_data, full_data_after_dr


def main():
    chunks_file = sys.argv[1]
    PCA_file = sys.argv[2]
    PCA_transformer= sys.argv[3]
    chunk_size = 1000

   
    #predict
    PCA_transformer = pickle.load(open(PCA_transformer,'rb'))
    train_full_data, train_full_data_after_dr = PCA_transform_incremental(PCA_transformer, chunks_file, PCA_file)
    temp = PCA_transformer.inverse_transform(train_full_data_after_dr)
    df_temp = pd.DataFrame(temp)
    print('Variance explained:',sum(PCA_transformer.explained_variance_ratio_))
    print('R2_score:',r2_score(train_full_data, PCA_transformer.inverse_transform(train_full_data_after_dr), multioutput='variance_weighted'))
    del train_full_data
    del train_full_data_after_dr


if __name__ == "__main__":
    main()

