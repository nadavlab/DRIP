import pickle
from sklearn.metrics import log_loss, roc_auc_score, roc_curve, average_precision_score
from sklearn.linear_model import SGDClassifier
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import pickle




def load_all_chunks(x_file, y_file):
    X_all = []
    y_all = []
    i=1
    with open(x_file, 'rb') as fx, open(y_file, 'rb') as fy:
    #with open(y_file, 'rb') as fy:
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
    #return pd.concat(X_all, axis=0), np.concatenate(y_all)
    return np.concatenate(y_all)


def LogisticRegression_incremental(x_train_chunks_file, y_train_chunks_file, X_val, y_val, pheno_name, output_path,epoch_from,epoch_to,model):
    train_loss = []   # to store average training loss for each epoch
    val_loss = []     # to store validation loss for each epoch

    # initialize the SGDClassifier with logistic loss (i.e. logistic regression)
    if model==None:
        model = SGDClassifier(loss='log_loss', learning_rate='constant', eta0=1e-5, penalty='l2', random_state=42)


    # flag to check if model has been initialized with classes via partial_fit
    is_initialized = False

    for epoch in range(epoch_from, epoch_to+1):
        print(f"Epoch {epoch}")
        batch_train_loss = []
        with open(x_train_chunks_file, 'rb') as file_handle:
            with open(y_train_chunks_file, 'rb') as file_handle2:
                batch_num = 1
                while True:
                    try:

                        print('-------------------------------', str(batch_num), '-----------------------------------')
                        X_batch = pickle.load(file_handle)
                        # drop unwanted column
                        X_batch = X_batch.drop(['FID'], axis=1)
                        X_batch.columns = X_batch.columns.astype(str)

                        y_batch = pickle.load(file_handle2)
                        y_batch = y_batch.drop(['FID'], axis=1)
                        print("y_batch_with_drop")
                        print(y_batch)
                        print("Columns in y_batch:", y_batch.columns)
                        print("Number of columns in y_batch:", len(y_batch.columns))
                        print("type:", y_batch.iloc[:,0].dtype)
                        y_batch.iloc[:,0] = y_batch.iloc[:,0].astype(int)
                        print("type:", y_batch.iloc[:,0].dtype)
                        print("y_batch")
                        print(y_batch)

                        if batch_num == 1:  # this chunk is used as the validation set (already loaded before)
                            batch_num += 1
                            continue

                        # Convert target to 1d array
                        y_batch_arr = y_batch.iloc[:, 0].astype(int).to_numpy()


                        # incremental training using partial_fit
                        if not is_initialized:
                            print(X_batch.head(6))
                            model.partial_fit(X_batch, y_batch_arr, classes=np.array([0, 1]))
                            is_initialized = True
                        else:
                            model.partial_fit(X_batch, y_batch_arr)

                        # compute training loss on this batch
                        probs = model.predict_proba(X_batch)
                        loss_batch = log_loss(y_batch_arr, probs, labels=[0, 1])
                        batch_train_loss.append(loss_batch)
                        batch_num += 1
                    except EOFError:
                        break

        # average training loss for this epoch
        avg_train_loss = np.mean(batch_train_loss)
        train_loss.append(avg_train_loss)

        # compute validation loss
        probs_val = model.predict_proba(X_val)
#        current_val_loss = log_loss(y_val.iloc[:,0].values, probs_val)
        current_val_loss = log_loss(y_val.iloc[:, 0].astype(int).to_numpy().ravel(), probs_val)

        val_loss.append(current_val_loss)
        print(f"Epoch {epoch}: Avg training loss = {avg_train_loss:.4f}, Validation loss = {current_val_loss:.4f}")

        if epoch % 10 == 0:
            print('Time fitting '+str(epoch)+' epochs:')
            time_in_minutes = float(time.time() - start_time)/60.0
            print('--- %s minutes---' % time_in_minutes)
            save_to = os.path.join(output_path, str(epoch))
            with open(save_to, 'wb') as f:
                pickle.dump(model, f)

    plot_loss(train_loss, val_loss, pheno_name, output_path,epoch_from,epoch_to)
    return model

def create_validation_set(X_train_chunks_file, y_train_chunks_file):
    with open(X_train_chunks_file, 'rb') as file_handle:
        X_val = pickle.load(file_handle)
        X_val = X_val.drop(['FID'], axis=1)
        X_val.columns = X_val.columns.astype(str)
    with open(y_train_chunks_file, 'rb') as y_file_handle:
        y_val = pd.read_pickle(y_file_handle)
        y_val = y_val.drop(['FID'], axis=1)
        print("type:", y_val.iloc[:,0].dtype)
        y_val.iloc[:,0] = y_val.iloc[:,0].astype(int)
        print("type:", y_val.iloc[:,0].dtype)
    return X_val, y_val

def predict_pheno(model, x_chunks_file):
    predictions = []
    with open(x_chunks_file, 'rb') as file_handle:
        while True:
            try:
                X_batch = pickle.load(file_handle)
                X_batch = X_batch.drop(['FID'], axis=1)
                X_batch.columns = X_batch.columns.astype(str)
                prediction = model.predict_proba(X_batch)
                # take probability for the positive class
                prediction_prob_1 = [row[1] for row in prediction]
                predictions.extend(prediction_prob_1)
            except EOFError:
                break
    return predictions

def plot_loss(train_loss, val_loss, pheno_name, output_path,epoch_from,epoch_to):
    fig = plt.figure()
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Logistic Regression Loss on ' + pheno_name)
    plt.ylabel('Log Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='upper left')
    plt.xlim(0,21)
    plt.ylim(0, max(max(train_loss), max(val_loss)))
    name = os.path.join(output_path, str(epoch_from)+"_"+str(epoch_to)+"_logloss.png")
    fig.savefig(name)
    plt.close(fig)



#Define param grid
param_grid_sgd_conservative = [
{
'sgd__loss': ['log_loss'],
'sgd__penalty': ['l2'],
'sgd__alpha': [1e-4, 1e-3, 1e-2],
'sgd__learning_rate': ['optimal', 'adaptive'],
'sgd__eta0': [0.01],
'sgd__max_iter': [1500],
'sgd__early_stopping': [True],
'sgd__n_iter_no_change': [10]
}
]

#Load all training data at once
start_time = time.time()

# start_time defined globally for timing logs
start_time = time.time()
pheno_name = sys.argv[1]
rep = str(sys.argv[2])
DRM = sys.argv[3]


# files
if DRM == "PCA":
    # train
    X_train_chunks_file = '/path/to/your/project/our_model/PCA/X_train_1k_chunks_PCA_dim_remove_no_missing/rep'+rep+"/"+pheno_name+"_X_train_match_to_pheno_MinMax_cov_MinMax.pkl"
    X_test_chunks_file = "/path/to/your/project/our_model/PCA/X_test_1k_chunks_PCA_dim_remove_no_missing/rep"+rep+"/"+pheno_name+"_X_test_match_to_pheno_MinMax_cov_MinMax.pkl"
    y_train_chunks_file = "/path/to/your/project/our_model/PCA/Y_files/rep"+rep+"/"+pheno_name+"_Y_train_1k_chunks_no_missing.pkl"
    y_test_file = "/path/to/your/project/our_model/PCA/Y_files/rep"+rep+"/"+pheno_name+"_Y_test_1k_chunks_no_missing.pkl"

    #output_path_pre = "/path/to/your/project/our_model/PCA/logistic_regression/"+pheno_name+"/rep"+rep+"/incremental_logistic_regression_MinMax_scaled_cov_MinMax_scaled_lr_1e-5_40_epochs"
    output_path = "/path/to/your/project/SDGclassifier_with_best_hp/"+pheno_name+"/rep"+rep+"/best_hyperparameters/"

if DRM == "Autoencoder":
    print("Autoencoder")
    # train
    X_train_chunks_file = "/path/to/your/project/our_model/Autoencoder/X_train_1k_chunks_dim_remove_no_missing_500_epochs/rep" + rep + "/" + pheno_name + "_X_train_match_to_pheno_MinMax_cov_MinMax.pkl"
    X_test_chunks_file = "/path/to/your/project/our_model/Autoencoder/X_test_1k_chunks_dim_remove_no_missing_500_epochs/rep" + rep + "/" + pheno_name + "_X_test_match_to_pheno_MinMax_cov_MinMax.pkl"
    y_train_chunks_file = "/path/to/your/project/our_model/Autoencoder/Y_files/rep" + rep + "/" + pheno_name + "_Y_train_1k_chunks_no_missing.pkl"
    y_test_file = "/path/to/your/project/our_model/Autoencoder/Y_files/rep" + rep + "/" + pheno_name + "_Y_test_1k_chunks_no_missing.pkl"
    output_path = "/path/to/your/project/NN_with_Autoencoder/" + pheno_name + "/rep" + rep + "/incremental_logistic_regression_MinMax_scaled_cov_MinMax_scaled_lr0.001_40_epochs/"

if not os.path.exists(output_path):
    os.makedirs(output_path)
else:
    print("predictions directory existed for rep" + rep)


X_train, y_train = load_all_chunks(X_train_chunks_file, y_train_chunks_file)
#y_train = load_all_chunks(X_train_chunks_file, y_train_chunks_file)

X_train.to_csv(os.path.join(output_path,'not_splited_data_X_train.csv'),index=False)
pd.Series(y_train).to_csv(os.path.join(output_path, 'not_splited_data_y_train.csv'),index=False)

#X_train=pd.read_csv(os.path.join(output_path,'not_splited_data_X_train.csv'))

#Define pipeline
pipeline = Pipeline([
('sgd', SGDClassifier(random_state=42))
])

#Run grid search
grid = GridSearchCV(pipeline, param_grid_sgd_conservative, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best score:", grid.best_score_)

#Now extract the best parameters to use for incremental training
best_params = grid.best_params_
sgd_best = SGDClassifier(
loss=best_params['sgd__loss'],
penalty=best_params['sgd__penalty'],
alpha=best_params['sgd__alpha'],
learning_rate=best_params['sgd__learning_rate'],
eta0=best_params.get('sgd__eta0', 0.01),
max_iter=1, # We override max_iter because we will use partial_fit
warm_start=True,
early_stopping=False, # Not used in incremental
l1_ratio=best_params.get('sgd__l1_ratio', 0.15),
random_state=42
)

#Now pass this model to your incremental training function
LogReg = LogisticRegression_incremental(x_train_chunks_file=X_train_chunks_file,
                                          y_train_chunks_file=y_train_chunks_file,
                                          X_val=X_val,
                                          y_val=y_val,
                                          pheno_name=pheno_name,
                                          output_path=output_path,
                                        epoch_from = 1,
                                        epoch_to = 20,
                                        model = sgd_best
                                        )

# Save the final model
with open(os.path.join(output_path, "LogisticRegression_model_with_best_hyper_parameters.pkl"), 'wb') as f:
    pickle.dump(LogReg, f)

print('Time fitting:')
time_in_minutes = float(time.time() - start_time)/60.0
print('--- %s minutes---' % time_in_minutes)


########################################## predict ###########################################
predictions = predict_pheno(model=LogReg,
                            x_chunks_file=X_test_chunks_file)
predictions_df = pd.DataFrame(predictions, columns=[pheno_name])
predictions_df.to_csv(os.path.join(output_path, "predictions_df_with_best_hyper_parameters.csv"), index=False)
y_test = pd.read_pickle(y_test_file)
y_test = y_test.drop(['FID'], axis=1)
y_test = y_test.astype(int)

print('log_loss =', log_loss(y_test, predictions))
fpr, tpr, thresholds = roc_curve(y_test, predictions)
print('roc_auc_score =', roc_auc_score(y_test, predictions))
print('average_precision_score =', average_precision_score(y_test, predictions))

metrics={}
metrics.update(
    {'pheno': pheno_name, 'rep': rep, 'model': 'SGDregressor_best_parameters', 'ROC_AUC': roc_auc_score(y_test, predictions), 'log_loss': log_loss(y_test, predictions),
     'average_precision_score': average_precision_score(y_test, predictions)})

# Append metrics to CSV
metrics_file = os.path.join("/path/to/your/project/SDGclassifier_with_best_hp/metrics_log.csv")
metrics_df = pd.DataFrame([metrics])
if os.path.exists(metrics_file):
    metrics_df.to_csv(metrics_file, mode='a', header=False, index=False)
else:
    metrics_df.to_csv(metrics_file, index=False)

print(f"Done in {(time.time() - start_time)/60:.2f} minutes.")

'''
,
{
'sgd__loss': ['log_loss'],
'sgd__penalty': ['elasticnet'],
'sgd__l1_ratio': [0.15, 0.5],
'sgd__alpha': [1e-4, 1e-3],
'sgd__learning_rate': ['optimal', 'adaptive'],
'sgd__eta0': [0.01],
'sgd__max_iter': [1500],
'sgd__early_stopping': [True],
'sgd__n_iter_no_change': [10]
},
{
'sgd__loss': ['hinge'],
'sgd__penalty': ['l2'],
'sgd__alpha': [1e-4, 1e-3],
'sgd__learning_rate': ['optimal'],
'sgd__max_iter': [1500],
'sgd__early_stopping': [True],
'sgd__n_iter_no_change': [10]
}'''