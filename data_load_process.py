import pandas as pd
import numpy as np
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from termcolor import colored
import warnings
warnings.filterwarnings("ignore")

def loading_data(dataset_path:str, columns_name:list):
    """
    This function loads the data and create a target column based on directory name
    param dataset_path: Path of the dataset directory
    param columns_name: list of column name
    return: Dataframe with data
    """
    # print(colored('Reading Data','blue'))
    logging.debug('Started Data Reading')
    sequence_row = []
    for r, d, f in os.walk(dataset_path):
        # if folder contains files
        if f:
            # print(r)
            logging.info('Reading Folder: '+r)
            # target column values
            target = r.replace(dataset_path, '').replace('\\','/').split('/')[0].split('_')[0].replace('-','_')
            for filename in sorted(f): 
                if filename.endswith('.csv'):
                    # reading files
                    try:
                        data = pd.read_csv(r +'/'+ filename, index_col='Unnamed: 0') 
                    except:
                        try:
                            data = pd.read_csv(r +'/'+ filename)
                        except Exception as e:
                            logging.error(e)
                            # print(e)
                            pass
                    #getting sample rows of data and append them into df dataframe
                    mp_seq = data[columns_name].values 
                    sequence_row.append([mp_seq, target])

    final_df = pd.DataFrame(sequence_row)
    final_df.columns = ['data','target']
    # print(colored('Data loading completed','green'))
    logging.debug('Data loading completed')
    return final_df


def one_hot_encoding(data):
    """
    This function converts target column to one hot encoding
    :param data: dataset
    :return: Dataframe with onehot encoded data, target column list
    """
    try:
        one_hot_encoded_data = pd.get_dummies(data, columns = ['target'])
        columns_list_target = one_hot_encoded_data.columns.tolist()
        columns_list_target.remove('data')
        logging.debug('Target Converted to One-Hot Encodings Successfully.')
        return one_hot_encoded_data, columns_list_target
    except Exception as e:
        logging.error(e)
        logging.debug('Pipeline Terminated')
        exit()

def feature_target_split(data, columns_list_target:list):
    """
    This function splits the features and targets columns
    :param data: dataset
    :param columns_list_target: list of target column names
    :return: Dataframe with onehot encoded data, target column list
    """
    try:
        ## setting Features and Target 
        x = np.array(data['data'].tolist())
        y = np.asarray(data[columns_list_target]).astype('float32')
        logging.debug('Data Divided to Features and Targets')
        return x, y
    except Exception as e:
        logging.error(e)
        logging.debug('Pipeline Terminated')
        exit()
    
def data_balance(X_train, y_train):
    """
    This function balances the imbalance data
    :param X_train: data features
    :param y_train: target data
    :return: balanced features and targets data
    """
    try:
        orig_shape = X_train.shape
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]*X_train.shape[2]))

        sampler = SMOTE(random_state=0)
        X_train, y_train = sampler.fit_resample(X_train, y_train)
        X_train = np.reshape(X_train, (X_train.shape[0],orig_shape[1], orig_shape[2]))
        logging.debug('Data balancing completed')
        return X_train, y_train
    except Exception as e:
        logging.error(e)
        logging.debug('Pipeline Terminated')
        exit()

def standardize_data(X_train, time_now):
    """
    This function standardize the data
    :param X_train: data features
    :param time_now: datetime in string for filename
    :return: standardize data and scaler model
    """
    try:
        # define standard scaler
        scaler = StandardScaler()
        # transform data
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        logging.debug('Data Standardization completed')
        return scaler, X_train
    except Exception as e:
        logging.error(e)
        logging.debug('Pipeline Terminated')
        exit()