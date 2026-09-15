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
                X_batch.columns = X_batch.columns.map(str)
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
    with open(y_train_chunks_file, 'rb') as y_file_handle:
        y_val = pd.read_pickle(y_file_handle).drop(['FID'], axis=1)
    return X_val, y_val

# Timing
start_time = time.time()
pheno_name = sys.argv[1]
rep = str(sys.argv[2])
DRM = sys.argv[3]

# File paths
if DRM == "PCA":
    X_train_chunks_file = f'/.../X_train_1k_chunks_PCA.../rep{rep}/{pheno_name}_X_train_match_to_pheno_MinMax_cov_MinMax.pkl'
    X_test_chunks_file = f"/.../X_test_1k_chunks_PCA.../rep{rep}/{pheno_name}_X_test_match_to_pheno_MinMax_cov_MinMax.pkl"
    y_train_chunks_file = f"/.../Y_files/rep{rep}/{pheno_name}_Y_train_1k_chunks_no_missing.pkl"
    y_test_file = f"/.../Y_files/rep{rep}/{pheno_name}_Y_test_1k_chunks_no_missing.pkl"
    output_path = f"/.../linear_regression/{pheno_name}/rep{rep}/classical_linear_regression/"

elif DRM == "Autoencoder":
    X_train_chunks_file = f"/.../X_train_1k_chunks.../rep{rep}/{pheno_name}_X_train_match_to_pheno_MinMax_cov_MinMax.pkl"
    X_test_chunks_file = f"/.../X_test_1k_chunks.../rep{rep}/{pheno_name}_X_test_match_to_pheno_MinMax_cov_MinMax.pkl"
    y_train_chunks_file = f"/.../Y_files/rep{rep}/{pheno_name}_Y_train_1k_chunks_no_missing.pkl"
    y_test_file = f"/.../Y_files/rep{rep}/{pheno_name}_Y_test_1k_chunks_no_missing.pkl"
    output_path = f"/.../NN_with_Autoencoder/{pheno_name}/rep{rep}/classical_linear_regression/"

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Load data
X_train_full, y_train_full = load_all_chunks(X_train_chunks_file, y_train_chunks_file)
X_val, y_val = create_validation_set(X_train_chunks_file, y_train_chunks_file)

# Fit classical Linear Regression
model = LinearRegression()
model.fit(X_train_full, y_train_full)

# Validation performance
pred_val = model.predict(X_val)
val_loss = mean_squared_error(y_val.iloc[:, 0].to_numpy().ravel(), pred_val)
print(f"Validation MSE: {val_loss:.4f}")

# Save model
with open(os.path.join(output_path, "LinearRegression_model.pkl"), 'wb') as f:
    pickle.dump(model, f)

# Predict test set
def predict_pheno(model, x_test_chunks_file):
    predictions = []
    with open(x_test_chunks_file, 'rb') as file_handle:
        while True:
            try:
                X_batch = pd.read_pickle(file_handle).drop(['FID'], axis=1)
                X_batch.columns = X_batch.columns.map(str)
                prediction = model.predict(X_batch)
                predictions.extend(prediction.flatten())
            except EOFError:
                break
    return predictions

predictions = predict_pheno(model, X_test_chunks_file)
predictions_df = pd.DataFrame(predictions, columns=[pheno_name])
predictions_df.to_csv(os.path.join(output_path, "predictions_df.csv"), index=False)

# Evaluate test performance
y_test = pd.read_pickle(y_test_file).drop(['FID'], axis=1).astype(int)
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))
print("Test R² score:", r2_score(y_test, predictions))

# Total time
print('Time fitting:')
print('--- %.2f minutes ---' % ((time.time() - start_time) / 60.0))
