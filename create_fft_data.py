from pathlib import Path
import pandas as pd
import numpy as np
import logging
import os
import sys
import glob
from numpy.fft import fft
from termcolor import colored

#give data path here
dataset = 'C:/Users/Govind.Banura/Desktop/condition_monitoring/data/' 
output_folder_path = "C:\\Users\\Govind.Banura\\Desktop\\condition_monitoring\\pipeline\\fft_data\\"

def plot_fft(data, sr):
    X = fft(data)
    N = len(X)
    n = np.arange(N)
    T = N/sr
    freq = n/T  

    return freq, np.abs(X)

for r, d, f in os.walk(dataset):
    if f:
        print(r)
        output = r.replace(dataset, '').replace('\\','/').replace('/','_').rsplit('_',1) # target column values
        category = output[0]
        try:
            subcategory = output[1]+'_'
        except:
            subcategory = ""

        output_path = output_folder_path + category + '/'
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        sequences = []
        for filename in f:
            if filename.endswith('.csv'):
                data = pd.read_csv(r +'/'+ filename, header=None)
                data.index = pd.date_range(start="2018-09-09 00:00:00", periods=250000, freq='20us').to_pydatetime().tolist()
                
                data = data.resample('1000us').mean()
                
                for column in data.columns:
                    freq, data[column] = plot_fft(data[column], 5000)
                
                data.index = freq
                if len(freq)!= 5000:
                    print(colored('|----- Error file:','red'),output_path +'/'+ filename)
                
                data.to_csv(output_path + subcategory + filename, index=True, header=True)
        del filename, output_path, sequences, data