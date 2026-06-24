import numpy as np
from pandas_plink import read_plink
import pandas as pd

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

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer=Adam(learning_rate=0.0000001))
    model.summary()
    return model


def fit_NN(x_train_chunks_file, y_train_chunks_file, num_epochs_from, num_epochs_to, X_val, y_val, start_time,pheno_name, model=None):
    loss = []
    val_loss = []
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
                        y_batch.iloc[:,0] = y_batch.iloc[:,0].astype(int)
                        X_batch.iloc[:,0] = X_batch.iloc[:,0].astype(int)

                        if epoch == 1 and batch_num == 1 and model==None:
                            model = build_model(X_batch)
                        if batch_num == 1: #its the validation set
                            batch_num = batch_num + 1
                            continue
                        # fit
                        history = model.fit(X_batch, y_batch, epochs=1, batch_size=50, validation_data=(X_val, y_val))
                        batch_num = batch_num + 1
                    except EOFError:
                        break
        if epoch%10 == 0:
            print('Time fiting'+str(epoch)+'epochs:')
            time_in_minutes = float(time.time() - start_time)/float(60)
            print('--- %s minutes---' % time_in_minutes)
            save_to = output_path + str(epoch)
            model.save(save_to,num_epochs_to)
        loss.extend(history.history['loss'])
        val_loss.extend(history.history['val_loss'])
        
    plot_loss(loss, val_loss, num_epochs_to, num_epochs_from,pheno_name)
    return model
    
def plot_loss(loss, val_loss,num_epochs_to,num_epochs_from,pheno_name):
    fig = plt.figure()
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('PROP-DEEP '+pheno_name)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    temp = int(num_epochs_to-num_epochs_from + 5)
    plt.xlim(0, temp)
    plt.ylim(0, 1)
    name = output_path+str(num_epochs_from)+"-"+str(num_epochs_to)+"_epochs.png"
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
        y_val.iloc[:,0] = y_val.iloc[:,0].astype(int)
    return X_val, y_val
        
  
import time
import sys
import os

###################################begining################
start_time = time.time()
X_train_chunks_file = sys.argv[1]
X_test_chunks_file= sys.argv[2]
y_train_chunks_file= sys.argv[3]
y_test_file = sys.argv[4]
output_path = sys.argv[5]
pheno_name = sys.argv[6]


if not os.path.exists(output_path):
    os.makedirs(output_path)
else:
    print("predictions directory existed for rep" + rep)

# train NN on train set
val_size = 1000
num_of_epochs=80



X_val, y_val = create_validation_set(X_train_chunks_file = X_train_chunks_file,
                                     y_train_chunks_file = y_train_chunks_file)

model = fit_NN(x_train_chunks_file = X_train_chunks_file,
               y_train_chunks_file = y_train_chunks_file,
               num_epochs_from = 1,
               num_epochs_to = num_of_epochs,
               X_val = X_val,
               y_val = y_val,
               start_time = start_time,
               pheno_name=pheno_name,
               model = None)

#-----------for predict only----------
#load NN model
#model_name = output_path+"/NN_0.2_dropout_0.1_dropout_relu_Adam0.0000001_batch_size_50_genes_MinMax_cov_MinMax/80"
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

print('log_loss=', log_loss(y_test,predictions))
fpr, tpr, thresholds = roc_curve(y_test, predictions)
print('roc_auc_score=',roc_auc_score(y_test,predictions))
print('average_precision_score =',average_precision_score(y_test, predictions))

