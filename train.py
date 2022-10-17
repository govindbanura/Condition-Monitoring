# Importing Libraries
import numpy as np
import logging
import pickle
import keras
import json
import h5py
import sys
import os
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM 
from sklearn.metrics import confusion_matrix
from keras.callbacks import CSVLogger, LambdaCallback
from termcolor import colored
import datetime
import warnings
warnings.filterwarnings("ignore")

# define model
def model(no_of_classes:int):
    """
    This function defines a sequential model to train
    :param no_of_classes: Total no of classes for the model
    :return: model
    """
    try:
        model = Sequential() #sequential model
        model.add(LSTM(32, dropout=0.2)) #lstm layer with 32 nodes and dropout of 20%

        model.add(Dense(20, activation='relu')) #Dense layer
        model.add(Dense(no_of_classes, activation='softmax')) #output layer
        model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), 
                    metrics=[keras.metrics.Precision(), keras.metrics.Recall()]) #compiling model with precision and recall metrics
        logging.debug('Model Created')
        return model
    except Exception as e:
        logging.error(e)
        logging.debug('Pipeline Terminated')
        exit()


def train(model, X_train, y_train, columns_list_target:list, time_now:str):
    """
    This function train the model on the data
    :param model: model object
    :param X_train: input data for training the model
    :param y_train: target data for model training
    :param column_list_target: list of target values
    :param time_now: date and time to save with files 
    """
    logging.debug('Model Training Started')
    #file path and name of the model 
    filepath = 'models/model'+time_now+'.h5'
    # logs for model
    txt_log = open('Model_version_log.txt', mode='a+', buffering=1)
    txt_log.write('\n\nModel Path: '+filepath + '\n')
    txt_log.write('Model Final Result:' + '\n')
    #callback to save model's final parameters 
    save_op_callback = LambdaCallback(
        on_train_end = lambda logs: txt_log.write(
            'loss: '+ str(round(logs['loss'],4)) + '  |  ' + 
            'val_loss: '+ str(round(logs['val_loss'],4)) + '\n'+
            'precision: '+ str(round(logs['precision'],4)) + '  |  ' + 
            'val_precision: '+ str(round(logs['val_precision'],4)) + '\n'+
            'recall: '+ str(round(logs['recall'],4)) + '  |  ' + 
            'val_recall: '+ str(round(logs['val_recall'],4)) + '\n'
        )
    )
    try:
        #model training
        model.fit(X_train, y_train, epochs=1, verbose=1, 
                    validation_split=0.2, batch_size=16, shuffle = True, callbacks=[save_op_callback])
        model.save(filepath, save_format="h5") # saving the model
        f = h5py.File(filepath, mode='a')
        f.attrs['labels'] = columns_list_target
        f.close()
        txt_log.close()
        logging.debug('Training Completed')
        logging.info('Model Saved at: '+filepath)
        logging.info('Model logs saved at: Model_version_log.txt')
        # print(colored('Training Completed', 'green'))
        return True
    except Exception as e:
        logging.error("Model training failed due to: " + e)
        # print(colored("Model training failed due to:", 'red'),colored(e, 'red'))
        exit()