import pickle
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys

def load_all_chunks(x_file, y_file):
    X_all = []
    y_all = []
    with open(x_file, 'rb') as fx, open(y_file, 'rb') as fy:
        while True:
            try:
                X_batch = pickle.load(fx).drop(['FID'], axis=1)
                X_batch.columns = [str(col) if not isinstance(col, tuple) else "_".join(map(str, col)) for col in X_batch.columns]

                y_batch = pickle.load(fy).drop(['FID'], axis=1)
                y_batch_arr = y_batch.iloc[:, 0].to_numpy().ravel()

                X_all.append(X_batch)
                y_all.append(y_batch_arr)
            except EOFError:
                break
    return pd.concat(X_all, axis=0), np.concatenate(y_all)

def plot_loss(train_loss, val_loss, pheno_name, output_path):
    fig = plt.figure()
    plt.plot(train_loss, label='Train Loss (MSE)')
    plt.plot(val_loss, label='Validation Loss (MSE)')
    plt.title('Linear Regression Loss on ' + pheno_name)
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epochs')
    plt.legend(loc='upper left')
    name = os.path.join(output_path, "mse_loss.png")
    fig.savefig(name)
    plt.close(fig)

def create_validation_set(X_train_chunks_file, y_train_chunks_file):
    with open(X_train_chunks_file, 'rb') as file_handle:
        X_val = pd.read_pickle(file_handle).drop(['FID'], axis=1)
        X_val.columns = [str(col) if not isinstance(col, tuple) else "_".join(map(str, col)) for col in X_val.columns]

    with open(y_train_chunks_file, 'rb') as y_file_handle:
        y_val = pd.read_pickle(y_file_handle).drop(['FID'], axis=1)
    return X_val, y_val

# Timing
start_time = time.time()
pheno_name = sys.argv[1]
rep = str(sys.argv[2])

# File paths
import pickle
import pandas as pd

pheno_name="Hypertension"
for i in range(1,6):
    rep=str(i)
    df=pd.read_pickle(f"/path/to/your/project/NN_with_SNP_selection/X_files/rep{rep}/{pheno_name}_X_train_all_chr_MinMax_cov_MinMax.pkl")
    print(len(df.index))
    #print(df.head(20))
    print(df.columns[198:])
    #df.iloc[:, 1] = df.iloc[:, 1].replace([1], "0")
    #df.iloc[:, 1] = df.iloc[:, 1].replace([2], "1")
    SNPs_list = df.columns[198:]
    #print(df.head(20))
    #df.to_pickle("/path/to/your/project/NN_with_SNP_selection/Y_files/rep"+rep+"/"+pheno_name+"_Y_test_1k_chunks_no_missing_1.pkl")
    new_df = (pd.DataFrame({"SNP": SNPs_list})
    new_df[['snp', 'chr']] = df['SNP'].str.split('_chr', expand=True)
    df.drop(columns=['SNP'], inplace=True)
    new_df.to_csv(
        f"/path/to/your/project/NN_with_SNP_selection/SNPs_list/rep{rep}_{pheno_name}_all_chro_SNP_list.csv",
        index=False))

#kjX_test_chunks_file = "/path/to/your/project/NN_with_SNP_selection/X_files/rep"+rep+"/"+pheno_name+"_X_test_match_to_pheno_cov_MinMax.pkl"
y_train_chunks_file = "/path/to/your/project/NN_with_SNP_selection/Y_files/rep"+rep+"/"+pheno_name+"_Y_train_1k_chunks_no_missing.pkl"
y_test_file = "/path/to/your/project/NN_with_SNP_selection/Y_files/rep"+rep+"/"+pheno_name+"_Y_test_1k_chunks_no_missing.pkl"
output_path = "/path/to/your/project/classical_lr_model_with_snp_selection/"+ pheno_name+"/rep"+rep+"/"
#output_path = "/path/to/your/project/classical_lr_model_with_snp_selection/"+pheno_name+"/rep"+rep+"/genetic_only/"


if not os.path.exists(output_path):
    os.makedirs(output_path)
else:
    print("predictions directory existed for rep" + rep)

# Load data
X_train_full, y_train_full = load_all_chunks(X_train_chunks_file, y_train_chunks_file)
X_train_full.columns = [str(col) if not isinstance(col, str) else col for col in X_train_full.columns]

X_val, y_val = create_validation_set(X_train_chunks_file, y_train_chunks_file)

# Fit classical Linear Regression
model = LinearRegression()
model.fit(X_train_full, y_train_full)

# Validation performance
pred_val = model.predict(X_val)
val_loss = mean_squared_error(y_val.iloc[:, 0].to_numpy().ravel(), pred_val)
print(f"Validation MSE: {val_loss:.4f}")

# Save model
with open(os.path.join(output_path, "classical_lr_model_with_snp_selection.pkl"), 'wb') as f:
    pickle.dump(model, f)

# Predict test set
def predict_pheno(model, x_test_chunks_file):
    predictions = []
    with open(x_test_chunks_file, 'rb') as file_handle:
        while True:
            try:
                X_batch = pd.read_pickle(file_handle).drop(['FID'], axis=1)
                X_batch.columns = [str(col) if not isinstance(col, tuple) else "_".join(map(str, col)) for col in X_batch.columns]
                prediction = model.predict(X_batch)
                predictions.extend(prediction.flatten())
            except EOFError:
                break
    return predictions

predictions = predict_pheno(model, X_test_chunks_file)
predictions_df = pd.DataFrame(predictions, columns=[pheno_name])
predictions_df.to_csv(os.path.join(output_path, "predictions_df_classical_lr_model_with_snp_selection.csv"), index=False)

# Evaluate test performance
y_test = pd.read_pickle(y_test_file).drop(['FID'], axis=1).astype(int)
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))
print("Test R² score:", r2_score(y_test, predictions))

# Total time
print('Time fitting:')
print('--- %.2f minutes ---' % ((time.time() - start_time) / 60.0))