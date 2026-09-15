import pandas as pd
import pickle
import os
import sys


def match_x_to_ids(x_file_name, phenotype_df, output_file):
    """
    Keep only samples that appear in phenotype table.
    Preserves the chunk structure.
    """

    total = 0

    with open(output_file, "wb") as fout:
        with open(x_file_name, "rb") as fin:

            while True:
                try:
                    X = pickle.load(fin)

                    X["FID"] = X["FID"].astype(str)

                    X = X[X["FID"].isin(phenotype_df["FID"])].reset_index(drop=True)

                    pickle.dump(X, fout, protocol=4)

                    total += len(X)

                except EOFError:
                    break

    print(f"{output_file}: {total} samples")


def create_y_chunks(pheno_df, matched_x_file, output_file):
    """
    Creates Y chunks (ALL phenotypes) in exactly the same order as X chunks.
    """

    with open(output_file, "wb") as fout:
        with open(matched_x_file, "rb") as fin:

            while True:
                try:

                    X = pickle.load(fin)

                    merged = pd.merge(
                        X[["FID"]],
                        pheno_df,
                        on="FID",
                        how="left"
                    )

                    pickle.dump(merged, fout, protocol=4)

                except EOFError:
                    break


def create_test_pickle(pheno_df, matched_x_file, output_file):
    """
    Test labels are stored as one dataframe (same as original code),
    but now containing ALL simulated phenotypes.
    """

    dfs = []

    with open(matched_x_file, "rb") as fin:

        while True:

            try:

                X = pickle.load(fin)

                merged = pd.merge(
                    X[["FID"]],
                    pheno_df,
                    on="FID",
                    how="left"
                )

                dfs.append(merged)

            except EOFError:
                break

    pd.concat(dfs, ignore_index=True).to_pickle(output_file)


#############################################################

rep = sys.argv[1]

train_pheno = f"/path/to/your/project//simulations/combined_DRIP_phenotypes_by_rep/rep{rep}_train_phenotypes.txt"

test_pheno = f"/path/to/your/project//simulations/combined_DRIP_phenotypes_by_rep/rep{rep}_test_phenotypes.txt"

#train_x = sys.argv[2]
#test_x = sys.argv[3]
path = '/path/to/your/project//our_model/PCA/'
#train
train_x =path+'X_train_1k_chunks_PCA_dim_remove_no_missing/rep'+rep+"/X_train_all_chr_MinMax_cov_MinMax.pkl"
# test
test_x = path+'X_test_1k_chunks_PCA_dim_remove_no_missing/rep' + rep + "/X_test_all_chr_MinMax_cov_MinMax.pkl"

#output = sys.argv[4]
output = f"/path/to/your/project//simulations/matched_data_ids"
os.makedirs(output, exist_ok=True)

#############################################################

train_df = pd.read_csv(train_pheno, sep="\t")
test_df = pd.read_csv(test_pheno, sep="\t")

train_df.rename(columns={"#FID": "FID"}, inplace=True)
test_df.rename(columns={"#FID": "FID"}, inplace=True)

train_df["FID"] = train_df["FID"].astype(str)
test_df["FID"] = test_df["FID"].astype(str)

#############################################################

matched_train_x = os.path.join(output, "X_train_match_to_pheno_rep_"+str(rep)+".pkl")
matched_test_x = os.path.join(output, "X_test_match_to_pheno_rep_"+str(rep)+".pkl")

match_x_to_ids(train_x, train_df, matched_train_x)
match_x_to_ids(test_x, test_df, matched_test_x)

#############################################################

train_y = os.path.join(output, "Y_train_all_rep_"+str(rep)+".pkl")
test_y = os.path.join(output, "Y_test_all_rep_"+str(rep)+".pkl")

create_y_chunks(train_df, matched_train_x, train_y)
create_test_pickle(test_df, matched_test_x, test_y)

print("Done.")