import numpy as np
from pandas_plink import read_plink
import pandas as pd
pd.__version__
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
from tensorflow.keras.callbacks import EarlyStopping

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)



#import torch
from math import sqrt
from sklearn.metrics import r2_score
import pickle
import matplotlib.pyplot as plt

from pynvml import *
import os

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used // 1024 ** 2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def build_model(batch,first_layer_units=0.12,second_layer_units=0.1,dropout=0.1,learning_rate=0.000004):
    # Define model
    number_snps = batch.shape[1]
    model = models.Sequential(name='NN')
    model.add(layer=layers.Dense(units=number_snps * first_layer_units, activation='elu', # was relu
                                 input_shape=[number_snps]))  # relu for autoencoder, layers.PReLU() for PCA
    ## OPT: Add layers.BatchNormalization()
    model.add(layers.Dropout(dropout))
   # model.add(layer=layers.Dense(units=number_snps *second_layer_units, activation='relu'))  # relu for autoencoder, PReLU() for PCA
    model.add(layer=layers.Dense(units=1024,
                                 activation='elu'))  # relu for autoencoder, PReLU() for PCA
    ## OPT: Add layers.BatchNormalization()
    model.add(layers.Dropout(dropout))
    model.add(layer=layers.Dense(units=1,activation="linear"))
    model.compile(loss="mse", optimizer=Adam(learning_rate=learning_rate))
    model.summary()
    return model


def fit_NN(x_train_chunks_file, y_train_chunks_file, num_epochs_from, num_epochs_to, X_val, y_val, pheno_name,
           model=None, patience=5):
    loss = []
    val_loss = []
    best_val_loss = np.inf  # Set to a very high value initially
    epochs_since_improvement = 0

    for epoch in range(num_epochs_from, num_epochs_to + 1):
        print('Epoch = ', epoch)
        with open(x_train_chunks_file, 'rb') as file_handle:
            with open(y_train_chunks_file, 'rb') as file_handle2:
                batch_num = 1
                while True:
                    try:
                        X_batch = pd.read_pickle(file_handle)
                        X_batch = X_batch.drop(['FID'], axis=1)
                        #print('X_batch')
                        #print(X_batch)
                        #print(X_batch.shape[0])

                        y_batch = pd.read_pickle(file_handle2)
                        y_batch = y_batch.drop(['FID'], axis=1)
                        #print('y_batch')
                        #print(y_batch)
                        #print(y_batch.shape[0])

                        if epoch == 1 and batch_num == 1 and model == None:
                            model = build_model(X_batch)
                            print_gpu_utilization()
                        if batch_num == 1:  # its the validation set
                            batch_num = batch_num + 1
                            continue

                        # fit
                        #if (X_batch)
                        #print('batch_num')
                        #print(batch_num)
                        history = model.fit(X_batch, y_batch, epochs=1, batch_size=64, validation_data=(X_val, y_val))
                        print_gpu_utilization()
                        batch_num = batch_num + 1
                    except EOFError:
                        break

                # Track loss
                epoch_val_loss = history.history['val_loss'][-1]
                loss.extend(history.history['loss'])
                val_loss.append(epoch_val_loss)

                # Early stopping check
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1

                if epochs_since_improvement >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    print(f'Time training {epoch} epochs:')
                    time_in_minutes = (time.time() - start_time) / 60
                    print(f'--- {time_in_minutes} minutes ---')
                    save_to = f"/path/to/your/project/NN_with_SNP_selection/predictions/rep{rep}/{pheno_name}/NN_model_{epoch}"
                    model.save(save_to)
                    break  # Stop training

        if epoch % 10 == 0:
            print('Time fiting' + str(epoch) + 'epochs:')
            time_in_minutes = float(time.time() - start_time) / float(60)
            print('--- %s minutes---' % time_in_minutes)
            
            save_to = "/path/to/your/project/NN_with_SNP_selection/predictions/rep"+rep+"/" + pheno_name + "/NN_0.2_0.1_dropout_0.1_relu_Adam0.0000001_batch_size_50_genes_MinMax_cov_MinMax/" + str(epoch)
            model.save(save_to, num_epochs_to)
        loss.extend(history.history['loss'])
        val_loss.extend(history.history['val_loss'])
    plot_loss(loss, val_loss, num_epochs_to, num_epochs_from, pheno_name)
    return model


def plot_loss(loss, val_loss, num_epochs_to, num_epochs_from, pheno_name):
    fig = plt.figure()
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('SNP selection with DNN for ' + pheno_name)  # change!!!!!!!!!!!!!!!!!!!!!!
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    temp = int(num_epochs_to - num_epochs_from + 5)
    plt.xlim(0, temp)
    plt.ylim(0, 100)
    name = "/path/to/your/project/NN_with_SNP_selection/predictions/rep"+rep+"/" + pheno_name + "/NN_0.2_0.1_dropout_0.1_relu_Adam0.0000001_batch_size_50_genes_MinMax_cov_MinMax/"+str(num_epochs_from)+"-"+str(num_epochs_to)+"_epochs.png"
    fig.savefig(name)


def predict_pheno(model, x_test_chunks_file):
    predictions = []
    with open(x_test_chunks_file, 'rb') as file_handle:
        while True:
            try:
                X_batch = pd.read_pickle(file_handle)
                X_batch = X_batch.drop(['FID'], axis=1)
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


print('available GPUs: ')
#print(torch.cuda.device_count())

import time
import sys

start_time = time.time()
pheno_name = sys.argv[1]
rep=str(sys.argv[2])

#------------predict-------------------
if not os.path.exists(
        "/path/to/your/project/NN_with_SNP_selection/predictions/rep"+rep+"/" + pheno_name+"/NN_0.2_0.1_dropout_0.1_relu_Adam0.0000001_batch_size_50_genes_MinMax_cov_MinMax/"):
    os.makedirs(
        "/path/to/your/project/NN_with_SNP_selection/predictions/rep"+rep+"/" + pheno_name+"/NN_0.2_0.1_dropout_0.1_relu_Adam0.0000001_batch_size_50_genes_MinMax_cov_MinMax/")
else:
    print( "predictions directory existed for rep"+rep)


# files
X_train_chunks_file = "/path/to/your/project/NN_with_SNP_selection/X_files/rep"+rep+"/"+pheno_name+"_X_train_match_to_pheno_cov_MinMax.pkl"

X_test_chunks_file = "/path/to/your/project/NN_with_SNP_selection/X_files/rep"+rep+"/"+pheno_name+"_X_test_match_to_pheno_cov_MinMax.pkl"


y_train_chunks_file = "/path/to/your/project/NN_with_SNP_selection/Y_files/rep"+rep+"/"+pheno_name+"_Y_train_1k_chunks_no_missing.pkl"
y_test_file = "/path/to/your/project/NN_with_SNP_selection/Y_files/rep"+rep+"/"+pheno_name+"_Y_test_1k_chunks_no_missing.pkl"

# train NN on train set
val_size = 1000
num_of_epochs=80

X_val, y_val = create_validation_set(X_train_chunks_file=X_train_chunks_file,
                                     y_train_chunks_file=y_train_chunks_file)
model = fit_NN(x_train_chunks_file=X_train_chunks_file,
               y_train_chunks_file=y_train_chunks_file,
               num_epochs_from=1,
               num_epochs_to=num_of_epochs,
               X_val=X_val,
               y_val=y_val,
               pheno_name=pheno_name,
               model=None)

# -----------for predict only----------
#load NN model
#model_name = "/path/to/your/project/NN_with_SNP_selection/predictions/rep"+rep+"/" + pheno_name + "/NN_0.2_0.1_dropout_0.1_relu_Adam0.0000001_batch_size_50_genes_MinMax_cov_MinMax/" + str(num_of_epochs)
#model = keras.models.load_model(model_name)


predictions = predict_pheno(model=model,
                            x_test_chunks_file=X_test_chunks_file)
predictions_df = pd.DataFrame(predictions, columns=[pheno_name])
predictions_df.to_csv(
    "/path/to/your/project/NN_with_SNP_selection/predictions/rep" +rep+"/"+ pheno_name +"/NN_0.2_0.1_dropout_0.1_relu_Adam0.0000001_batch_size_50_genes_MinMax_cov_MinMax/predictions_df_"+str(num_of_epochs)+"_epochs")

# predictions_df = predictions_df.rename(columns={"0":"height"})

y_test = pd.read_pickle(y_test_file)
y_test = y_test.drop(['FID'], axis=1)
print("mean_squared_error")
print(np.sqrt(mean_squared_error(y_test, predictions)))
print("r2 score")
print(r2_score(y_test, predictions))
