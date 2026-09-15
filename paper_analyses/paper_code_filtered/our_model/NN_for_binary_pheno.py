import numpy as np
from pandas_plink import read_plink
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)


#import torch
from math import sqrt
from sklearn.metrics import r2_score,log_loss, roc_auc_score, roc_curve, average_precision_score
import pickle
import matplotlib.pyplot as plt
import os


def build_model(batch):
    # Define model
    number_snps = batch.shape[1]
    model = models.Sequential(name='NN')
    model.add(layer=layers.Dense(units=number_snps*0.2, activation=layers.PReLU(), input_shape=[number_snps]))
    model.add(layers.Dropout(0.1))
    model.add(layer=layers.Dense(units=number_snps*0.1, activation=layers.PReLU()))
    model.add(layers.Dropout(0.1))
    model.add(layer=layers.Dense(units=1, activation='sigmoid'))


    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer=Adam(learning_rate=1e-4))
    model.summary()
    return model


def fit_NN(x_train_chunks_file, y_train_chunks_file, num_epochs_from, num_epochs_to,
           X_val, y_val, start_time, model=None, patience=5):
    loss = []
    val_loss = []
    best_val_loss = np.inf  # Initialize best validation loss
    epochs_since_improvement = 0

    # Define callbacks manually (will call them manually per epoch)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    for epoch in range(num_epochs_from, num_epochs_to + 1):
        print(f"Epoch {epoch}/{num_epochs_to}")

        with open(x_train_chunks_file, 'rb') as file_handle, open(y_train_chunks_file, 'rb') as file_handle2:
            batch_num = 1
            while True:
                try:
                    X_batch = pickle.load(file_handle).drop(['FID'], axis=1)
                    y_batch = pickle.load(file_handle2).drop(['FID'], axis=1)

                    # Ensure correct dtype
                    y_batch.iloc[:, 0] = y_batch.iloc[:, 0].astype(int)
                    X_batch.iloc[:, 0] = X_batch.iloc[:, 0].astype(int)

                    # Initialize model if needed
                    if epoch == 1 and batch_num == 1 and model is None:
                        model = build_model(X_batch)

                    # Train on current batch
                    history = model.fit(X_batch, y_batch, epochs=1, batch_size=50, verbose=1)

                    # Track batch loss
                    loss.extend(history.history['loss'])

                    batch_num += 1

                except EOFError:
                    break  # End of file

        # Evaluate on validation set at the end of the epoch
        epoch_val_loss = model.evaluate(X_val, y_val, verbose=0)
        val_loss.append(epoch_val_loss)

        print(f"Validation Loss: {epoch_val_loss}")

        # Manually call ReduceLROnPlateau & EarlyStopping
        reduce_lr.on_epoch_end(epoch, logs={'val_loss': epoch_val_loss})
        early_stopping.on_epoch_end(epoch, logs={'val_loss': epoch_val_loss})

        # Check if EarlyStopping was triggered
        if early_stopping.stopped_epoch > 0:
            print("Early stopping triggered. Stopping training.")
            break

    plot_loss(history.history['loss'], history.history['val_loss'], num_epochs_to, num_epochs_from, epoch)
    return model


def fit_NN_1(x_train_chunks_file, y_train_chunks_file, num_epochs_from, num_epochs_to, X_val, y_val, start_time, model=None,patience=5):
    loss = []
    val_loss = []
    best_val_loss = np.inf  # Set to a very high value initially
    epochs_since_improvement = 0

    reduce_lr = ReduceLROnPlateau( monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    for epoch in range(num_epochs_from, num_epochs_to + 1):
        print('Epoch = ', epoch)
        with open(x_train_chunks_file, 'rb') as file_handle:
            with open(y_train_chunks_file, 'rb') as file_handle2:
                batch_num = 1
                while True:
                    try:
                        X_batch = pickle.load(file_handle)
                        X_batch = X_batch.drop(['FID'], axis=1)
                        y_batch = pickle.load(file_handle2)
                        y_batch = y_batch.drop(['FID'], axis=1)
                        #print("y_batch_with_drop")
                        #print(y_batch)
                        print("Columns in y_batch:", y_batch.columns)
                        print("Number of columns in y_batch:", len(y_batch.columns))
                        print("type:", y_batch.iloc[:,0].dtype)
                        y_batch.iloc[:,0] = y_batch.iloc[:,0].astype(int)
                        print("type:", y_batch.iloc[:,0].dtype)

                        #print("y_batch")
                        #print(y_batch)

                        #print("X_batch_with_drop")
                        #print(X_batch)
                        print("Columns in X_batch:", X_batch.columns)
                        print("Number of columns in X_batch:", len(X_batch.columns))
                        print("type:", X_batch.iloc[:,0].dtype)
                        X_batch.iloc[:,0] = X_batch.iloc[:,0].astype(int)
                        print("type:", X_batch.iloc[:,0].dtype)

                        #print("X_batch")
                        #print(X_batch)

                        if epoch == 1 and batch_num == 1 and model==None:
                            model = build_model(X_batch)
                        if batch_num == 1: #its the validation set
                            batch_num = batch_num + 1
                            continue
                        # fit

                        history = model.fit(X_batch, y_batch, epochs=1, batch_size=50,validation_data=(X_val, y_val), callbacks=[reduce_lr,early_stopping])
                        #history = model.fit(X_batch, y_batch, epochs=1, batch_size=50, validation_data=(X_val, y_val))
                        batch_num = batch_num + 1
                    except EOFError:
                        break

                        # Track loss
        epoch_val_loss = history.history['val_loss'][-1]
        loss.extend(history.history['loss'][-1])
        val_loss.append(epoch_val_loss)
                    
                    
        # Early stopping check
        if epoch_val_loss < best_val_loss:
            print('epoch_val_loss')
            print(epoch_val_loss)
            print('best_val_loss')
            print(best_val_loss)

            best_val_loss = epoch_val_loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            print(f'Time training {epoch} epochs:')
            time_in_minutes = (time.time() - start_time) / 60
            print(f'--- {time_in_minutes} minutes ---')
            save_to = output_path + str(epoch)
            model.save(save_to)
            break  # Stop training
                        
        if epoch%10 == 0:
            print('Time fiting'+str(epoch)+'epochs:')
            time_in_minutes = float(time.time() - start_time)/float(60)
            print('--- %s minutes---' % time_in_minutes)
            save_to = output_path + str(epoch)
            model.save(save_to)
        print("loss")
        print(history.history['loss'])
        loss.extend(history.history['loss'])
        print(loss)
        val_loss.extend(history.history['val_loss'])
    plot_loss(loss, val_loss, num_epochs_to, num_epochs_from,epoch)
    return model
    
def plot_loss(loss, val_loss,num_epochs_to,num_epochs_from,epoch):
    fig = plt.figure()
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('SNP selection with DNN for '+pheno_name) #change!!!!!!!!!!!!!!!!!!!!!!
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    temp = int(epoch + 5)
    plt.xlim(0, temp)
    plt.ylim(0, 1)
    name = output_path+str(num_epochs_from)+"-"+str(epoch)+"_epochs.png"
    fig.savefig(name)

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
        
  
import time
import sys
import os

###################################begining################
start_time = time.time()
pheno_name = sys.argv[1]
rep = str(sys.argv[2])
DRM = sys.argv[3]



# files
if DRM == "PCA":
    # train
    X_train_chunks_file = '/path/to/your/project/our_model/PCA/X_train_1k_chunks_PCA_dim_remove_no_missing/rep' + rep + "/" + pheno_name + "_X_train_match_to_pheno_MinMax_cov_MinMax.pkl"
    X_test_chunks_file = "/path/to/your/project/our_model/PCA/X_test_1k_chunks_PCA_dim_remove_no_missing/rep" + rep + "/" + pheno_name + "_X_test_match_to_pheno_MinMax_cov_MinMax.pkl"
    y_train_chunks_file = "/path/to/your/project/our_model/PCA/Y_files/rep" + rep + "/" + pheno_name + "_Y_train_1k_chunks_no_missing.pkl"
    y_test_file = "/path/to/your/project/our_model/PCA/Y_files/rep" + rep + "/" + pheno_name + "_Y_test_1k_chunks_no_missing.pkl"
    output_path = "/path/to/your/project/NN_with_PCA/" + pheno_name + "/rep" + rep + "/NN_0.2_0.1_dropout_0.1_relu_Adam_updated_lr_early_stoping_batch_size_50_genes_MinMax_cov_MinMax/"


if DRM == "Autoencoder":
    print("Autoencoder")
    X_train_chunks_file = "/path/to/your/project/our_model/Autoencoder/X_train_1k_chunks_dim_remove_no_missing_500_epochs/rep" + rep + "/" + pheno_name + "_X_train_match_to_pheno_MinMax_cov_MinMax.pkl"
    X_test_chunks_file = "/path/to/your/project/our_model/Autoencoder/X_test_1k_chunks_dim_remove_no_missing_500_epochs/rep" + rep + "/" + pheno_name + "_X_test_match_to_pheno_MinMax_cov_MinMax.pkl"
    y_train_chunks_file = "/path/to/your/project/our_model/Autoencoder/Y_files/rep" + rep + "/" + pheno_name + "_Y_train_1k_chunks_no_missing.pkl"
    y_test_file = "/path/to/your/project/our_model/Autoencoder/Y_files/rep" + rep + "/" + pheno_name + "_Y_test_1k_chunks_no_missing.pkl"
    output_path = "/path/to/your/project/NN_with_Autoencoder/" + pheno_name + "/rep" + rep + "/NN_0.2_0.1_dropout_0.1_relu_Adam0.0000001_batch_size_50_genes_MinMax_cov_MinMax/"

if not os.path.exists(output_path):
    os.makedirs(output_path)
else:
    print("predictions directory existed for rep" + rep)

# train NN on train set
val_size = 1000
num_of_epochs = 80
trained_epoches = 0
trained_model = None


X_val, y_val = create_validation_set(X_train_chunks_file = X_train_chunks_file,
                                     y_train_chunks_file = y_train_chunks_file)

#---------for continue training model that stoped due to time/storage problems-----
#trained_epoches = 30
#model_name = output_path+"/"+str(trained_epoches)
#trained_model = keras.models.load_model(model_name)
#-----------------------------

model = fit_NN(x_train_chunks_file = X_train_chunks_file,
               y_train_chunks_file = y_train_chunks_file,
               num_epochs_from = trained_epoches+1,
               num_epochs_to = num_of_epochs,
               X_val = X_val,
               y_val = y_val,
               start_time = start_time,
               model = trained_model)

#-----------for predict only----------
#load NN model
#model_name = "/path/to/your/project/NN_with_snp_selection/predictions/"+pheno_name+"/NN_0.2_dropout_0.1_dropout_relu_Adam0.0000001_batch_size_50_genes_MinMax_cov_MinMax/80"
#model = keras.models.load_model(model_name)

#-----------predict---------------

predictions = predict_pheno(model=model,
                            x_chunks_file = X_test_chunks_file)
predictions_df = pd.DataFrame(predictions,columns=[pheno_name])                            
predictions_df.to_csv(output_path+"predictions_df_"+str(num_of_epochs)+"_epochs")
predictions_df = predictions_df.rename(columns={"0":pheno_name})
predictions = predictions_df.to_numpy()
predictions_df=predictions_df.astype(float)
y_test = pd.read_pickle(y_test_file)
y_test = y_test.drop(['FID'], axis=1)
y_test=y_test.astype(int)
print(y_test.head(20))

print('log_loss=', log_loss(y_test,predictions))
fpr, tpr, thresholds = roc_curve(y_test, predictions)
print('roc_auc_score=',roc_auc_score(y_test,predictions))
print('average_precision_score =',average_precision_score(y_test, predictions))
#plot_roc_curve(fpr, tpr)

