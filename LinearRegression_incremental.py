import pickle
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import SGDRegressor
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys

def LinearRegression_incremental(x_train_chunks_file, y_train_chunks_file, X_val, y_val, pheno_name, output_path):
    train_loss = []   # to store average training loss for each epoch
    val_loss = []     # to store validation loss for each epoch

    # Initialize the SGDRegressor
    model=SGDRegressor(loss='squared_error', learning_rate='optimal', eta0=1e-5, penalty='l2', random_state=42)
    is_initialized = False  # Flag to check if model has been initialized with partial_fit

    batch_train_loss = []
    with open(x_train_chunks_file, 'rb') as file_handle:
        with open(y_train_chunks_file, 'rb') as file_handle2:
            batch_num = 1
            while True:
                try:
                    print('-------------------------------', str(batch_num), '-----------------------------------')
                    X_batch = pickle.load(file_handle)
                    X_batch = X_batch.drop(['FID'], axis=1)
                    X_batch.columns = X_batch.columns.map(str)
                    X_batch.columns = X_batch.columns.map(lambda x: str(x) if isinstance(x, tuple) else x)

                    #print('X_batch.head()')
                    #print(X_batch.head())
                    #print(X_batch.describe())

                    y_batch = pickle.load(file_handle2)
                    y_batch = y_batch.drop(['FID'], axis=1)
                    y_batch_arr = y_batch.iloc[:, 0].to_numpy().ravel()
                    #print('y_batch_arr.head()')
                    #print(y_batch_arr[:10])

                    if batch_num == 1:
                        batch_num += 1
                        continue

                    # Incremental training using partial_fit
                    if not is_initialized:
                        model.partial_fit(X_batch, y_batch_arr)
                        is_initialized = True
                    else:
                        model.partial_fit(X_batch, y_batch_arr)

                    # Compute training loss (MSE)
                    predictions = model.predict(X_batch)
                    loss_batch = mean_squared_error(y_batch_arr, predictions)
                    batch_train_loss.append(loss_batch)
                    batch_num += 1
                except EOFError:
                    break

    # Average training loss for this epoch
    avg_train_loss = np.mean(batch_train_loss)
    train_loss.append(avg_train_loss)

    # Compute validation loss (MSE)
    X_val.columns = X_val.columns.map(str)
    predictions_val = model.predict(X_val)
    current_val_loss = mean_squared_error(y_val.iloc[:, 0].to_numpy().ravel(), predictions_val)
    val_loss.append(current_val_loss)
    print(f"Avg training loss = {avg_train_loss:.4f}, Validation loss = {current_val_loss:.4f}")

    print('Time fitting:')
    time_in_minutes = float(time.time() - start_time)/60.0
    print('--- %s minutes---' % time_in_minutes)
    #save_to = os.path.join(output_path, str(epoch))
    save_to = os.path.join(output_path, "learning_rate_constant_eta0=1e-4_penalty=l2")

    with open(save_to, 'wb') as f:
        pickle.dump(model, f)

    #plot_loss(train_loss, val_loss, pheno_name, output_path)
    return model

def plot_loss(train_loss, val_loss, pheno_name, output_path):
    fig = plt.figure()
    plt.plot(train_loss, label='Train Loss (MSE)')
    plt.plot(val_loss, label='Validation Loss (MSE)')
    plt.title('Linear Regression Loss on ' + pheno_name)
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epochs')
    plt.legend(loc='upper left')
    plt.xlim(0, len(train_loss))
    plt.ylim(0, max(max(train_loss), max(val_loss)))
    name = os.path.join(output_path, "mse_loss.png")
    fig.savefig(name)
    plt.close(fig)


def predict_pheno(model, x_test_chunks_file):
    predictions = []
    with open(x_test_chunks_file, 'rb') as file_handle:
        while True:
            try:
                X_batch = pd.read_pickle(file_handle)
                X_batch = X_batch.drop(['FID'], axis=1)
                X_batch.columns = X_batch.columns.map(str)
                prediction = model.predict(X_batch)
                predictions.extend(prediction.flatten())
            except EOFError:
                break
    return predictions


def create_validation_set(X_train_chunks_file, y_train_chunks_file):
    with open(X_train_chunks_file, 'rb') as file_handle:
        X_val = pd.read_pickle(file_handle)
        X_val = X_val.drop(['FID'], axis=1)
    with open(y_train_chunks_file, 'rb') as y_file_handle:
        y_val = pd.read_pickle(y_file_handle)
        y_val = y_val.drop(['FID'], axis=1)
    return X_val, y_val


# start_time defined globally for timing logs
start_time = time.time()
pheno_name = sys.argv[1]
rep = str(sys.argv[2])
DRM = sys.argv[3]

# files
if DRM == "PCA":
    # train
    X_train_chunks_file = '/path/to/project/PCA/X_train_1k_chunks_PCA_dim_remove_no_missing/rep'+rep+"/"+pheno_name+"_X_train_match_to_pheno_MinMax_cov_MinMax.pkl"
    X_test_chunks_file = "/path/to/project/PCA/X_test_1k_chunks_PCA_dim_remove_no_missing/rep"+rep+"/"+pheno_name+"_X_test_match_to_pheno_MinMax_cov_MinMax.pkl"
    y_train_chunks_file = "/path/to/project/PCA/Y_files/rep"+rep+"/"+pheno_name+"_Y_train_1k_chunks_no_missing.pkl"
    y_test_file = "/path/to/project/PCA/Y_files/rep"+rep+"/"+pheno_name+"_Y_test_1k_chunks_no_missing.pkl"

    output_path = "/path/to/project/PCA/linear_regression/"+pheno_name+"/rep"+rep+"/incremental_linear_regression_MinMax_scaled_cov_MinMax_scaled_lr1e-5_20_epochs/"

if DRM == "Autoencoder":
    print("Autoencoder")
    # train
    X_train_chunks_file = "/path/to/project/Autoencoder/X_train_1k_chunks_dim_remove_no_missing_500_epochs/rep" + rep + "/" + pheno_name + "_X_train_match_to_pheno_MinMax_cov_MinMax.pkl"
    X_test_chunks_file = "/path/to/project/Autoencoder/X_test_1k_chunks_dim_remove_no_missing_500_epochs/rep" + rep + "/" + pheno_name + "_X_test_match_to_pheno_MinMax_cov_MinMax.pkl"
    y_train_chunks_file = "/path/to/project/Autoencoder/Y_files/rep" + rep + "/" + pheno_name + "_Y_train_1k_chunks_no_missing.pkl"
    y_test_file = "/path/to/project/Y_files/rep" + rep + "/" + pheno_name + "_Y_test_1k_chunks_no_missing.pkl"
    output_path = "/path/to/project/Autoencoder/NN/" + pheno_name + "/rep" + rep + "/incremental_linear_regression_MinMax_scaled_cov_MinMax_scaled_lr0.001_40_epochs/"

if not os.path.exists(output_path):
    os.makedirs(output_path)
else:
    print("predictions directory existed for rep" + rep)

X_val, y_val = create_validation_set(X_train_chunks_file=X_train_chunks_file,
                                     y_train_chunks_file=y_train_chunks_file)

#################################### fit model ###########################################
LogReg = LinearRegression_incremental(x_train_chunks_file=X_train_chunks_file,
                                          y_train_chunks_file=y_train_chunks_file,
                                          X_val=X_val,
                                          y_val=y_val,
                                          pheno_name=pheno_name,
                                          output_path=output_path)

# Save the final model
with open(os.path.join(output_path, "LinearRegression_model.pkl"), 'wb') as f:
    pickle.dump(LogReg, f)

print('Time fitting:')
time_in_minutes = float(time.time() - start_time)/60.0
print('--- %s minutes---' % time_in_minutes)

##########################################predict only######################################

#with open(os.path.join(output_path, "LinearRegression_model.pkl"), 'rb') as f:
#    LogReg = pickle.load(f)
########################################## predict ###########################################
predictions = predict_pheno(model=LogReg,
                            x_test_chunks_file=X_test_chunks_file)
predictions_df = pd.DataFrame(predictions, columns=[pheno_name])
predictions_df.to_csv(os.path.join(output_path, "predictions_df.csv"), index=False)
y_test = pd.read_pickle(y_test_file)
y_test = y_test.drop(['FID'], axis=1)
y_test = y_test.astype(int)

print("mean_squared_error")
print(np.sqrt(mean_squared_error(y_test, predictions)))
print("r2 score")
print(r2_score(y_test, predictions))
