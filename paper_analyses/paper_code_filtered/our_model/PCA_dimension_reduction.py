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

                    full_data = pd.concat([full_data, batch], ignore_index=True)

                    batch_PCA = PCA_transformer.transform(batch)
                    batch_PCA_df = pd.DataFrame(batch_PCA)

                    full_data_after_dr = pd.concat([full_data_after_dr, batch_PCA_df], ignore_index=True)

                    batch_PCA_df['FID'] = ID
                    pickle.dump(batch_PCA_df, save_file_handle, protocol=4)
                except EOFError:
                    return full_data, full_data_after_dr


def learn_PCA_matrix(X_chunks_file):
    with open(X_chunks_file, 'rb') as file_handle:
        batch = pickle.load(file_handle)

    full_data = pd.DataFrame(columns=batch.keys())
    batch_num = 1

    with open(X_chunks_file, 'rb') as file_handle:
        while True:
            try:
                if batch_num == 1:
                    batch_num += 1
                    continue

                batch = pickle.load(file_handle)
                full_data = pd.concat([full_data, batch], ignore_index=True)
                batch_num += 1

            except EOFError:
                break

    #pca = PCA(n_components=int((len(full_data.keys()) - 1) * 0.1))
    pca = PCA(n_components=int((len(full_data.keys()) - 1) ))
    full_data = full_data.drop(['FID'], axis=1)
    pca.fit(full_data)

    return pca


def main():
    chr=sys.argv[1]
    rep=sys.argv[2]

'''
    sys.stdout.flush()
    start_time = time.time()
    print(chr)
    X_train_chunks_file = '/path/to/your/project/our_model/splited_bed_files_for_all_chro/rep'+rep+'/' + str(chr) +"_X_train_1k_chunks_no_missing.pkl"
    train_output_file = "/path/to/your/project/our_model/PCA/X_train_1k_chunks_PCA_dim_remove_no_missing/rep"+rep+'/chr_' + str(chr) + "_full_PCs.pkl"
    PCA_transformer = learn_PCA_matrix(X_train_chunks_file)
    print('Time to learn PCA for chromosome ', str(chr),':')
    time_in_minutes = float(time.time() - start_time)/float(60)
    print("--- %s minutes for PCA fiting---" % time_in_minutes)
    pca_name = "/path/to/your/project/our_model/PCA/PCA_transformer/rep"+rep+"/PCA_transformer_" + str(chr) + "_full_PCs.pkl"
    pickle.dump(PCA_transformer, open(pca_name, 'wb'), protocol=4)
'''

    #predict
    #-----train------
pca_name = "/path/to/your/project/our_model/PCA/PCA_transformer/rep" + rep + "/PCA_transformer_" + str(
    chr) + "_full_PCs.pkl"

PCA_transformer = pickle.load(open(pca_name,'rb'))
    train_full_data, train_full_data_after_dr = PCA_transform_incremental(PCA_transformer, X_train_chunks_file, train_output_file)
    temp = PCA_transformer.inverse_transform(train_full_data_after_dr)
    df_temp = pd.DataFrame(temp)
    print('Train variance explained:',sum(PCA_transformer.explained_variance_ratio_))
    print('Train r2_score:',r2_score(train_full_data, PCA_transformer.inverse_transform(train_full_data_after_dr), multioutput='variance_weighted'))
    del train_full_data
    del train_full_data_after_dr

    #-----test------
    X_test_chunks_file = '/path/to/your/project/our_model/splited_bed_files_for_all_chro/rep'+rep+'/' + str(chr) +"_X_test_1k_chunks_no_missing.pkl"
    test_output_file = "/path/to/your/project/our_model/PCA/X_test_1k_chunks_PCA_dim_remove_no_missing/rep"+rep+'/chr_' + str(chr) + "_full_PCs.pkl"
    test_full_data, test_full_data_after_dr = PCA_transform_incremental(PCA_transformer, X_test_chunks_file, test_output_file)
    print('Test r2_score:',r2_score(test_full_data, PCA_transformer.inverse_transform(test_full_data_after_dr), multioutput='variance_weighted'))

    with open(X_train_chunks_file, 'rb') as file_handle: 
        batch = pickle.load(file_handle)
        batch = batch.drop(['FID'], axis=1) #validation
    print('Validation r2_score:',r2_score(batch, PCA_transformer.inverse_transform(PCA_transformer.transform(batch)), multioutput='variance_weighted'))


if __name__ == "__main__":
    main()

