import xgboost
import pickle
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score,log_loss, recall_score, roc_auc_score, roc_curve,average_precision_score
import time
import matplotlib.pyplot as plt
import os
import sys

def XGBRegressor_incremental(x_train_chunks_file, y_train_chunks_file, X_val, y_val,pheno_name,output_path):
    loss = []
    val_loss = []
    model = xgboost.XGBClassifier(n_estimators=1, max_depth=2, learning_rate=0.001, subsample=1.0, colsample_bytree=0.2, gamma=0.01)
    #default objective=binary:logistic
    #default eval_metric=error
    #gamma (min_split_loss) - inimum loss reduction required to make a further partition on a leaf node of the tree.
    #colsample_bytree - subsample ratio of columns when constructing each tree.
    for epoch in range(1,21):
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
                        print("y_batch_with_drop")
                        print(y_batch)
                        print("Columns in y_batch:", y_batch.columns)
                        print("Number of columns in y_batch:", len(y_batch.columns))
                        print("type:", y_batch.iloc[:,0].dtype)
                        y_batch.iloc[:,0] = y_batch.iloc[:,0].astype(int)
                        print("type:", y_batch.iloc[:,0].dtype)

                        print("y_batch")
                        print(y_batch)

                        if batch_num == 1: #its the validation set
                            batch_num = batch_num + 1
                            continue
                        # fit
                        eval_set = [(X_batch, y_batch), (X_val, y_val)]
                        if batch_num == 2 and epoch==1:
                            model.fit(X_batch, y_batch, eval_metric='logloss', eval_set=eval_set, verbose=True)
                            #change eval_metric=logloss
                        else:
                            model.fit(X_batch, y_batch, xgb_model=model, eval_metric='logloss', eval_set=eval_set, verbose=True)
                            #change eval_metric=logloss
                        batch_num = batch_num + 1
                        results = model.evals_result()
                        batch_train_loss.extend(results['validation_0']['logloss'])
                    except EOFError:
                        break
        loss.append(np.average(batch_train_loss))
        val_loss.extend(results['validation_1']['logloss'])
        if epoch%10 == 0:
            print('Time fiting '+str(epoch)+' epochs:')
            time_in_minutes = float(time.time() - start_time)/float(60)
            print('--- %s minutes---' % time_in_minutes)
            save_to = output_path+str(epoch) #for Autoencoder
            model.save_model(save_to)            
    plot_loss(loss, val_loss,pheno_name,output_path)
    return model
    
def create_validation_set(X_train_chunks_file, y_train_chunks_file):
    with open(X_train_chunks_file, 'rb') as file_handle:
        X_val = pickle.load(file_handle)
        X_val = X_val.drop(['FID'], axis=1)
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
                prediction = model.predict_proba(X_batch)
                prediction_prob_1 = [row[1] for row in prediction]
                predictions.extend(prediction_prob_1)
            except EOFError:
                break
    return predictions
    
def plot_loss(loss, val_loss,pheno_name, output_path):
    fig = plt.figure()
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('SNP selection XGBoost Loss on '+pheno_name)
    plt.ylabel('Loss') 
    plt.xlabel('Boosting rounds')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.xlim(0, len(loss))
    plt.ylim(0, max(np.max(loss),np.max(val_loss)))
    name = output_path+"logloss.png"
    fig.savefig(name)


    
start_time = time.time()
pheno_name = sys.argv[1]
rep = str(sys.argv[2])

# files
X_train_chunks_file = "/path/to/your/project/NN_with_SNP_selection/X_files/rep"+rep+"/"+pheno_name+"_X_train_match_to_pheno_cov_MinMax.pkl"
X_test_chunks_file = "/path/to/your/project/NN_with_SNP_selection/X_files/rep"+rep+"/"+pheno_name+"_X_test_match_to_pheno_cov_MinMax.pkl"

y_train_chunks_file = "/path/to/your/project/NN_with_SNP_selection/Y_files/rep"+rep+"/"+pheno_name+"_Y_train_1k_chunks_no_missing.pkl"
y_test_file = "/path/to/your/project/NN_with_SNP_selection/Y_files/rep"+rep+"/"+pheno_name+"_Y_test_1k_chunks_no_missing.pkl"

output_path = "/path/to/your/project/XGB_with_SNP_selection/"+pheno_name+"/rep"+rep+"/incremental_xgboost_classifier_MinMax_scaled_cov_MinMax_scaled_max_depth2_learning_rate0.001_subsample1.0_olsample_bytree0.2_gamma0.01_20_epochs/"
if not os.path.exists(output_path):
        os.makedirs(output_path)
else:
    print( "predictions directory existed for rep"+rep)

        
X_val, y_val = create_validation_set(X_train_chunks_file = X_train_chunks_file,
                                     y_train_chunks_file = y_train_chunks_file)
                                     
#x_df, y_df = concat_genes(X_train_chunks_dim_remove_file, y_train_chunks_file)

#################################### fit model ###########################################

XGB = XGBRegressor_incremental(x_train_chunks_file = X_train_chunks_file,
                               y_train_chunks_file = y_train_chunks_file,
                               X_val = X_val,
                               y_val = y_val,
                               pheno_name=pheno_name,
                               output_path = output_path)
                              
XGB.save_model(output_path+"XGBRegressor_model")

print('Time fiting:')
time_in_minutes = float(time.time() - start_time)/float(60)
print('--- %s minutes---' % time_in_minutes)

                
########################################## predict only ###########################################
#load_model
#XGB = xgboost.XGBClassifier()
#XGB.load_model(output_path+"XGBRegressor_model")

######################################################prediction###############################
predictions = predict_pheno(model=XGB,
                            x_chunks_file = X_test_chunks_file)
predictions_df = pd.DataFrame(predictions,columns=['hypertension'])         
predictions_df.to_csv(output_path+"/predictions_df")
y_test = pd.read_pickle(y_test_file)
y_test = y_test.drop(['FID'], axis=1)
y_test=y_test.astype(int)

print('log_loss=', log_loss(y_test,predictions))
#print('recall_score=',recall_score(y_test,predictions))
fpr, tpr, thresholds = roc_curve(y_test, predictions)
print('roc_auc_score=',roc_auc_score(y_test,predictions))
print('average_precision_score =',average_precision_score(y_test, predictions))

#optimal_idx = np.argmax(tpr - fpr)
#optimal_threshold = thresholds[optimal_idx]
#print("Threshold value is:", optimal_threshold)
#predictions_df['pred_label'] = -1
#predictions_df.loc[predictions_df['hypertension']>optimal_threshold,'pred_label'] = 1
#predictions_df.loc[predictions_df['hypertension']<=optimal_threshold,'pred_label'] = 0
#predictions_df['pred_label'].to_csv("/home/hochyard/my_model/autoencoder/autoencoder_models_5_layers_prelu_act_no_cov_adam0.00001/hypertension_pheno/XGB/incremental_xgboost_regressor_MinMax_scaled_cov_MinMax_scaled_max_depth7_learning_rate0.001_subsample1.0_olsample_bytree0.2_gamma0.01/pred_labels")
#print(r2_score(y_test, predictions_df['pred_label'].to_numpy()))
#plot_roc_curve(fpr, tpr)

