from pathlib import Path
import pandas as pd
import numpy as np
import logging
import pickle
import h5py
import glob
import os
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM 
import datetime
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
from termcolor import colored
import warnings
warnings.filterwarnings("ignore")


def get_model():

    logging.debug('Last trained model loading - Started')
    list_of_files = glob.glob('models/*.h5')
    model_name = max(list_of_files, key=os.path.getctime)
    scaler_name = model_name.replace('model', 'stand_scaler').replace('.h5','.pkl')
    try:
        model = load_model(model_name)
        f = h5py.File(model_name, mode='r')
        labels = None
        if 'labels' in f.attrs:
            labels = f.attrs.get('labels')
        f.close()
        logging.debug('Model loading - Completed')
    except Exception as e:
        print('Model load failed:', e)
        logging.error('Model load failed due to: '+ e)
        logging.error('Pipeline Terminated')
        exit()

    try:
        logging.debug('Loading Standard Scaler')
        with open(scaler_name, 'rb') as f:
            scaler = pickle.load(f)
        logging.debug('Standard Scalar Loaded Succesfully')
    except Exception as e:
        print('Scaler model load failed:', e)
        logging.error('Standard Scaler load failed due to: '+ e)
        logging.error('Pipeline Terminated')
        exit()

    return scaler, model, labels


def standardize_test_data(scaler, test_data):
    """
    This function standardize the test data
    :param scaler: standard scaler model
    :param test_data: test data features
    :return: standardize test data
    """
    try:
        logging.debug('Standardizing Test data')
        test_data = scaler.transform(test_data.reshape(-1, test_data.shape[-1])).reshape(test_data.shape)
        
        if len(test_data.shape) == 2:
            test_data = test_data.reshape(1, test_data.shape[0], test_data.shape[1])
        logging.debug('Standardizing Test data - Completed')
        return test_data    
    except Exception as e:
        logging.error(e)
        logging.debug('Pipeline Terminated')
        exit()


def model_predict(model, test_data):
    try:
        y_pred = model.predict(test_data)

        max_index = y_pred.argmax(axis=1)

        return max_index
    except Exception as e:
        logging.error(e)
        logging.error('Testing Interrupted')
        exit()

def test_model(model, X_test, y_test, columns_list_target:list, time_now):
    try:
        y_pred = model.predict(X_test)    
        _y_pred = np.argmax(y_pred, axis=1)
        _y_test = np.argmax(y_test, axis=1)
        cm = confusion_matrix(_y_test, _y_pred)

        cm_df = pd.DataFrame(cm, columns=columns_list_target, index=columns_list_target)
        cm_df.to_csv('confusion_matrix/conf_mat'+time_now+'.csv')
        logging.info('Confusion Matrix saved at: confusion_matrix/conf_mat'+time_now+'.csv')
        logging.debug('Testing Completed')
        print('Testing Completed')
    except Exception as e:
        logging.error(e)
        logging.error('Testing Interrupted')
        exit()
    
def predict_class(predict_file_path):
    logging.debug('Prediction Started')
    try:
        data_test = pd.read_csv(predict_file_path, index_col='Unnamed: 0')
    except:
        try:
            data_test = pd.read_csv(predict_file_path)
        except Exception as e:
            logging.error('Predict Failed Due to: '+e)
            # print(colored(e, 'red'))
            raise CustomError('Predict Failed due to: '+e)

    # getting input columns
    column_1 = data_test['0'] #tachometer_signal
    column_2 = data_test['1'] #underhang_bearing_accelerometer_axial
    column_3 = data_test['2'] #underhang_bearing_accelerometer_radiale 
    column_4 = data_test['3'] #underhang_bearing_accelerometer_tangential_direction
    column_5 = data_test['4'] #overhang_bearing_accelerometer_axial
    column_6 = data_test['5'] #overhang_bearing_accelerometer_radiale
    column_7 = data_test['6'] #overhang_bearing_accelerometer_tangential_direction
    column_8 = data_test['7'] #microphone

    if len(column_1) == len(column_2) == len(column_3) == len(column_4) == len(column_5) == len(column_6) == len(column_7) == len(column_8) == 5000:
        test_data = np.array([column_1, column_2, column_3, column_4, column_5, column_6, column_7, column_8])
        test_data = np.transpose(test_data)

        scaler, model, labels = get_model()
        test_data = standardize_test_data(scaler, test_data)
        result = model_predict(model, test_data)
        logging.info('Predicted Class: '+labels[result][0])
        print(colored('Predicted Class: '+labels[result][0], 'green'))
        return labels[result][0]
    else:
        logging.error('Test data shape should be 5000 * 8')
        print(colored('Error: test data shape should be 5000 * 8','red'))
        raise CustomError('Predict Failed due to: '+e)  