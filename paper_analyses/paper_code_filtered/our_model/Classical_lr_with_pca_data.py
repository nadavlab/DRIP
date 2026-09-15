import pickle
import pandas as pd
import numpy as np
import os
import sys
import time
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score, log_loss, average_precision_score
import warnings
from sklearn.exceptions import ConvergenceWarning



def load_all_chunks(x_file, y_file):
    X_all = []
    y_all = []
    i=1
    with open(x_file, 'rb') as fx, open(y_file, 'rb') as fy:
        while True:
            try:
                X_batch = pickle.load(fx).drop(['FID'], axis=1)
                X_batch.columns = X_batch.columns.map(str)
                y_batch = pickle.load(fy).drop(['FID'], axis=1)
                X_batch = X_batch.astype(np.float32)  # Reduces memory usage
                y_batch_arr = y_batch.iloc[:, 0].to_numpy().ravel()
                X_all.append(X_batch)
                y_all.append(y_batch_arr)
                print(i)
                i+=1
                #if i>3:
                #    break
            except EOFError:
                break
    return pd.concat(X_all, axis=0), np.concatenate(y_all)

def run_model(x_train_file, y_train_file, x_test_file, y_test_file, pheno_name, output_path, model_type, rep):
    # Load all training data
    print("Loading training data...")
    X_train, y_train = load_all_chunks(x_train_file, y_train_file)

    # Choose model
    if model_type == "l":
        model = LinearRegression()
        #model = ElasticNet(alpha=0.1, l1_ratio=0.5)
        #model = Ridge(alpha=1.0)
        #model = Lasso(alpha=0.1)
        print("Training Linear Regression model...")
        model.fit(X_train, y_train)

    elif model_type == "b":
        with warnings.catch_warnings():
            warnings.filterwarnings("always", category=ConvergenceWarning)
            model = LogisticRegression(max_iter=5000, solver='saga', class_weight='balanced',verbose=1)
            model.fit(X_train, y_train)
            print("Training Logistic Regression model...")
            print(f"Number of iterations to convergence: {model.n_iter_}")

    else:
        raise ValueError("model_type must be 'l' for linear or 'b' for logistic regression")

    # Train
    # model.fit(X_train, y_train)

    # Save model
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, f"{pheno_name}_model_Lasso.pkl"), 'wb') as f:
        pickle.dump(model, f)


    #with open(os.path.join(output_path, f"{pheno_name}_model_ElasticNet.pkl"), 'rb') as f:
    #    model = pickle.load(f)

    # Predict
    print("Predicting on test data...")
    predictions = []
    with open(x_test_file, 'rb') as file_handle:
        while True:
            try:
                X_batch = pd.read_pickle(file_handle).drop(['FID'], axis=1)
                X_batch.columns = X_batch.columns.map(str)
                if model_type == "b":
                    preds = model.predict_proba(X_batch)[:, 1]
                else:
                    preds = model.predict(X_batch)
                predictions.extend(preds.flatten())
            except EOFError:
                break

    '''
    # Load the test data from all chunks
    X_test, y_test = load_all_chunks(x_test_file, y_test_file)

    # Predict probabilities for class 1
    predictions = model.predict_proba(X_test)[:, 1]

    # Convert to binary predictions using threshold (e.g., 0.5)
    preds_binary = (predictions >= 0.5).astype(int)

'''
    predictions_df = pd.DataFrame(predictions, columns=[pheno_name])
    predictions_df.to_csv(os.path.join(output_path, "predictions_df_Lasso.csv"), index=False)

    #predictions_df = pd.read_csv(os.path.join(output_path, "predictions_df_5000_iter.csv"))
    predictions = predictions_df[pheno_name]
    #predictions = np.array(predictions, dtype=int)
    # Load test labels
    y_test = pd.read_pickle(y_test_file).drop(['FID'], axis=1)
    y_test = y_test.iloc[:, 0].to_numpy().ravel()

    # Evaluate
    print("Evaluation:")
    metrics = {}
    if model_type == "l":
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        print("RMSE:", rmse)
        print("R² score:", r2)
        metrics.update({'pheno': pheno_name , 'rep' : rep , 'model' : 'classical_lr', 'RMSE': rmse, 'R2': r2, 'model_hyp':'LinearRegression'})
    else:
        preds_binary = (predictions >= 0.5).astype(int)
        y_test = np.array(y_test, dtype=int)
        acc = accuracy_score(y_test, preds_binary)
        log_los = log_loss(y_test, predictions)
        prec =  average_precision_score(y_test, predictions)
        try:
            auc = roc_auc_score(y_test, predictions)
        except:
            auc = np.nan
        print("Accuracy:", acc)
        print("ROC AUC:", auc)
        print('log_loss:', log_los)
        print('average_precision_score =',prec )

        metrics.update({'pheno': pheno_name , 'rep' : rep ,'model' : 'classical_lr', 'Accuracy': acc, 'ROC_AUC': auc , 'log_loss': log_los , 'average_precision_score': prec , 'iterations': model.n_iter_[0],'model':'balanced_l1'})
    return metrics

# ------------------ main -----------------------
start_time = time.time()

pheno_name = sys.argv[1]
rep = sys.argv[2]
DRM = sys.argv[3]
model_type = sys.argv[4]  # "l" or "b"

if DRM == "PCA":
    X_train_chunks_file = f"/path/to/your/project/our_model/PCA/X_train_1k_chunks_PCA_dim_remove_no_missing/rep{rep}/{pheno_name}_X_train_match_to_pheno_MinMax_cov_MinMax.pkl"
    X_test_chunks_file = f"/path/to/your/project/our_model/PCA/X_test_1k_chunks_PCA_dim_remove_no_missing/rep{rep}/{pheno_name}_X_test_match_to_pheno_MinMax_cov_MinMax.pkl"
    y_train_chunks_file = f"/path/to/your/project/our_model/PCA/Y_files/rep{rep}/{pheno_name}_Y_train_1k_chunks_no_missing.pkl"
    y_test_file = f"/path/to/your/project/our_model/PCA/Y_files/rep{rep}/{pheno_name}_Y_test_1k_chunks_no_missing.pkl"
    output_path = f"/path/to/your/project/classical_lr_PCA/{pheno_name}/rep{rep}/model_{model_type}/"

elif DRM == "Autoencoder":
    X_train_chunks_file = f"/path/to/your/project/our_model/Autoencoder/X_train_1k_chunks_dim_remove_no_missing_500_epochs/rep{rep}/{pheno_name}_X_train_match_to_pheno_MinMax_cov_MinMax.pkl"
    X_test_chunks_file = f"/path/to/your/project/our_model/Autoencoder/X_test_1k_chunks_dim_remove_no_missing_500_epochs/rep{rep}/{pheno_name}_X_test_match_to_pheno_MinMax_cov_MinMax.pkl"
    y_train_chunks_file = f"/path/to/your/project/our_model/Autoencoder/Y_files/rep{rep}/{pheno_name}_Y_train_1k_chunks_no_missing.pkl"
    y_test_file = f"/path/to/your/project/our_model/Autoencoder/Y_files/rep{rep}/{pheno_name}_Y_test_1k_chunks_no_missing.pkl"
    #output_path = f"/path/to/your/project/impoving_PRS/results/Autoencoder/{pheno_name}/rep{rep}/model_{model_type}/"

else:
    raise ValueError("DRM must be 'PCA' or 'Autoencoder'")

metrics = run_model(X_train_chunks_file, y_train_chunks_file, X_test_chunks_file, y_test_file,
          pheno_name, output_path, model_type,rep)

# Append metrics to CSV
metrics_file = os.path.join("/path/to/your/project/classical_lr_PCA/metrics_log.csv")
metrics_df = pd.DataFrame([metrics])
if os.path.exists(metrics_file):
    metrics_df.to_csv(metrics_file, mode='a', header=False, index=False)
else:
    metrics_df.to_csv(metrics_file, index=False)

print(f"Done in {(time.time() - start_time)/60:.2f} minutes.")
