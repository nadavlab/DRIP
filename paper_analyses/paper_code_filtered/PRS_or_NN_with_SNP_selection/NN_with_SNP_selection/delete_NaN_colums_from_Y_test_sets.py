import pickle
import pandas as pd

pheno_name="Multiple_Sclerosis"
for i in range(1,6):
    rep=str(i)
    df=pd.read_pickle("/path/to/your/project/NN_with_SNP_selection/Y_files/rep"+rep+"/"+pheno_name+"_Y_test_1k_chunks_no_missing.pkl")
    print(len(df.index))
    print(df.head(20))
    new_df=df.iloc[:,[0,-1]]

    new_df.to_pickle("/path/to/your/project/NN_with_SNP_selection/Y_files/rep"+rep+"/"+pheno_name+"_Y_test_1k_chunks_no_na_no_missing.pkl")
    new_df=pd.read_pickle("/path/to/your/project/NN_with_SNP_selection/Y_files/rep"+rep+"/"+pheno_name+"_Y_test_1k_chunks_no_na_no_missing.pkl")
    print(new_df.head(20))


has_nan = new_df.isna().any().any()
if has_nan:
    print("There are NaN values in the DataFrame.")
else:
    print("There are no NaN values in the DataFrame.")

 #   file2.close()


