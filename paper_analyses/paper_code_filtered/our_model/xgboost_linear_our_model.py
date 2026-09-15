
import pickle
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import time
import matplotlib.pyplot as plt
import sys
import xgboost
import os


def XGBRegressor_incremental(x_train_chunks_file, y_train_chunks_file, X_val, y_val, output_path,pheno_name, pretrain=False, model=None):
    loss = []
    val_loss = []
    if model == None:
        model = xgboost.XGBRegressor(n_estimators=1, max_depth=5, learning_rate=0.001, subsample=1.0, colsample_bytree=0.2, gamma=0.01)
    #default objective=reg:squarederror
    #default eval_metric=rmse
    #gamma (min_split_loss) - minimum loss reduction required to make a further partition on a leaf node of the tree.
    #colsample_bytree - subsample ratio of columns when constructing each tree.
    for epoch in range(1,41):
        print(epoch)
        batch_train_loss = []
        with open(x_train_chunks_file, 'rb') as file_handle:
            with open(y_train_chunks_file, 'rb') as file_handle2:
                batch_num = 1
                while True:
                    try:
                        print('-------------------------------',str(batch_num),'-----------------------------------')
                        X_batch = pickle.load(file_handle)
                        X_batch = X_batch.drop(['FID'], axis=1)
                        y_batch = pickle.load(file_handle2)
                        y_batch = y_batch.drop(['FID'], axis=1)
                        if batch_num == 1: #its the validation set
                            batch_num = batch_num + 1
                            continue
                        # fit
                        eval_set = [(X_batch, y_batch), (X_val, y_val)]
                        if batch_num == 2 and epoch==1 and pretrain==False:
                            model.fit(X_batch, y_batch, eval_set=eval_set, verbose=True)
                        else:
                            model.fit(X_batch, y_batch, xgb_model=model.get_booster(), eval_set=eval_set, verbose=True)
                        batch_num = batch_num + 1
                        results = model.evals_result()
                        batch_train_loss.extend(results['validation_0']['rmse'])
                    except EOFError:
                        break
        
        loss.append(np.average(batch_train_loss))
        val_loss.extend(results['validation_1']['rmse'])
        if epoch%10 == 0:
            print('Time fiting '+str(epoch)+' epochs:')
            time_in_minutes = float(time.time() - start_time)/float(60)
            print('--- %s minutes---' % time_in_minutes)
            save_to = output_path+str(epoch)
            model.save_model(save_to)
            
    plot_loss(loss, val_loss, pheno_name,output_path)
    return model
    
def create_validation_set(X_train_chunks_file, y_train_chunks_file):
    with open(X_train_chunks_file, 'rb') as file_handle:
        X_val = pickle.load(file_handle)
        X_val = X_val.drop(['FID'], axis=1)
    with open(y_train_chunks_file, 'rb') as y_file_handle:
        y_val = pd.read_pickle(y_file_handle)
        y_val = y_val.drop(['FID'], axis=1)
    return X_val, y_val
    
def predict_pheno(model, x_chunks_file):
    predictions = []
    with open(x_chunks_file, 'rb') as file_handle:
        while True:
            try:
                X_batch = pickle.load(file_handle)
                X_batch = X_batch.drop(['FID'], axis=1)
                prediction = model.predict(X_batch)
                predictions.extend(prediction.flatten())
            except EOFError:
                break
    return predictions
    
def plot_loss(loss, val_loss, pheno_name,output_path):
    fig = plt.figure()
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('snp selection XGBoost Loss on '+pheno_name)
    plt.ylabel('Loss')
    plt.xlabel('Boosting rounds')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.xlim(0, len(loss))
    plt.ylim(0, max(np.max(loss),np.max(val_loss)))
    name = output_path+"logloss.png"
    fig.savefig(name)


###################################begining################
start_time = time.time()
pheno_name = sys.argv[1]
rep = str(sys.argv[2])
DRM = sys.argv[3]

# files
if DRM == "PCA":

    #train
    X_train_chunks_file = '/path/to/your/project/our_model/PCA/X_train_1k_chunks_PCA_dim_remove_no_missing/rep'+rep+"/"+pheno_name+"_X_train_match_to_pheno_MinMax_cov_MinMax.pkl"
    X_test_chunks_file = "/path/to/your/project/our_model/PCA/X_test_1k_chunks_PCA_dim_remove_no_missing/rep"+rep+"/"+pheno_name+"_X_test_match_to_pheno_MinMax_cov_MinMax.pkl"
    y_train_chunks_file = "/path/to/your/project/our_model/PCA/Y_files/rep"+rep+"/"+pheno_name+"_Y_train_1k_chunks_no_missing.pkl"
    y_test_file = "/path/to/your/project/our_model/PCA/Y_files/rep"+rep+"/"+pheno_name+"_Y_test_1k_chunks_no_missing.pkl"

    output_path = "/path/to/your/project/our_model/PCA/xgboost/"+pheno_name+"/rep"+rep+"/incremental_xgboost_regressor_MinMax_scaled_cov_MinMax_scaled_max_depth5_learning_rate0.001_subsample1.0_olsample_bytree0.2_gamma0.01_40_epochs/"

if DRM == "Autoencoder":
    print ("Autoencoder")
    # train
    X_train_chunks_file = "/path/to/your/project/our_model/Autoencoder/X_train_1k_chunks_dim_remove_no_missing_500_epochs/rep" + rep + "/" + pheno_name + "_X_train_match_to_pheno_MinMax_cov_MinMax.pkl"
    X_test_chunks_file = "/path/to/your/project/our_model/Autoencoder/X_test_1k_chunks_dim_remove_no_missing_500_epochs/rep" + rep + "/" + pheno_name + "_X_test_match_to_pheno_MinMax_cov_MinMax.pkl"
    y_train_chunks_file = "/path/to/your/project/our_model/Autoencoder/Y_files/rep" + rep + "/" + pheno_name + "_Y_train_1k_chunks_no_missing.pkl"
    y_test_file = "/path/to/your/project/our_model/Autoencoder/Y_files/rep" + rep + "/" + pheno_name + "_Y_test_1k_chunks_no_missing.pkl"
    output_path = "/path/to/your/project/NN_with_Autoencoder/" + pheno_name + "/rep" + rep + "/incremental_xgboost_regressor_MinMax_scaled_cov_MinMax_scaled_max_depth5_learning_rate0.001_subsample1.0_olsample_bytree0.2_gamma0.01_40_epochs/"


if not os.path.exists(output_path):
    os.makedirs(output_path)
else:
    print( "predictions directory existed for rep"+rep)

X_val, y_val = create_validation_set(X_train_chunks_file = X_train_chunks_file,
                                     y_train_chunks_file = y_train_chunks_file)

###############################################prediction only#############################################

#XGB = xgboost.XGBRegressor()
#XGB.load_model(output_path+"XGBRegressor_model")
###############################################prediction only#############################################

###############################################new model#############################################

XGB = XGBRegressor_incremental(x_train_chunks_file = X_train_chunks_file,
                               y_train_chunks_file = y_train_chunks_file,
                               X_val = X_val,
                               y_val = y_val,
                               pheno_name=pheno_name,
                               pretrain = False,
                               model = None,
                               output_path=output_path)
                               
XGB.save_model(output_path+"XGBRegressor_model")

print('Time fiting:')
time_in_minutes = float(time.time() - start_time)/float(60)
print('--- %s minutes---' % time_in_minutes)
###############################################new model#############################################


                  
#test predict
predictions = predict_pheno(model=XGB,
                            x_chunks_file = X_test_chunks_file)
predictions_df = pd.DataFrame(predictions)         
predictions_df.to_csv(output_path+"predictions_df")
y_test = pd.read_pickle(y_test_file)
y_test = y_test.drop(['FID'], axis=1)
print("np.sqrt(mean_squared_error(y_test,predictions))")
print(np.sqrt(mean_squared_error(y_test,predictions)))
print("Rsquared")

print(r2_score(y_test, predictions))





