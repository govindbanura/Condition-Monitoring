# importing libraries
from data_load_process import *
from train import *
from predict import *
import datetime
import yaml
import pickle
from yaml.loader import SafeLoader
from termcolor import colored
import logging
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    log_filename = './logs/' + 'log' + datetime.datetime.now().strftime("_%d_%m_%Y_%H_%M") + ".log"
    logging.basicConfig(filename=log_filename, filemode='w', level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')

    try:
        with open('train_config.yaml') as f:
            train_vars = yaml.load(f, Loader=SafeLoader)

        #user defined variables
        dataset_path = train_vars['dataset_path']
        columns_name = train_vars['columns_name']
        data_balanced = train_vars['data_balanced']
        time_now = datetime.datetime.now().strftime("_%d_%m_%Y_%H_%M_%S")
        logging.debug('Loading train_config successfully')
    except Exception as e:
        logging.error(e)
    
    data = loading_data(dataset_path, columns_name)
    data, columns_list_target = one_hot_encoding(data)

    X_train, y_train = feature_target_split(data, columns_list_target)

    if not data_balanced:
        X_train, y_train = data_balance(X_train, y_train)

    scaler, X_train = standardize_data(X_train, time_now)
    standard_scaler_path = 'stand_scalers/stand_scaler'+time_now+'.pkl'
    pickle.dump(scaler, open(standard_scaler_path, 'wb'))
    logging.info("Standard Scaler Saved at: "+ standard_scaler_path)
    # print(colored('Data Standardization completed','green'))

    model = model(len(columns_list_target))
    train(model, X_train, y_train, columns_list_target, time_now)

    if train_vars['test']:
        logging.debug('Testing Started')
        test_folder_path = train_vars['test_folder_path']

        data = loading_data(test_folder_path, columns_name)
        # print(colored(data['target'].value_counts(),'red'))
        data, columns_list_target = one_hot_encoding(data)
        X_test, y_test = feature_target_split(data, columns_list_target)
        test_data = standardize_test_data(scaler, X_test)

        result = test_model(model, X_test, y_test, columns_list_target, time_now)  
        logging.debug('Testing Completed')

    
    
    