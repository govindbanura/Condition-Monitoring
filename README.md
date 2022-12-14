# How to Use
### 1. Dataset
We can download the data from the link : [dataset](http://www02.smt.ufrj.br/~offshore/mfs/page_01.html#SEC2)

### 2. Convert Data to FFT
After downloading the data, we have to convert it into FFT using "create_fft_data.py".
In the file, add 'dataset' and 'output_folder_path' and run it using
```bash
python create_fft_data.py
```

### 3. Train Test Split
After converting data to FFT, we can split the data into train and test via a library called [splitfolder](https://pypi.org/project/split-folders/).

Using Python
```python
import splitfolders

splitfolders.ratio(
    <dataset_path>, 
    output=<output_folder_path>,
    seed=1337, 
    ratio=(.7, .3)
)
```
Using Terminal
```bash
splitfolders --ratio .7 .3 -- <dataset_path>
```
### 4. Train & Test
First, configure the **train_config.yaml** to train & test the model and then run the following command in the terminal.
```bash
python main.py
```
After training, all files will be saved in the following manner
> Standard Scaler
>>     Condition-Monitoring/
>>                       stand_scalers/
>>                                    <Standard-Scaler.pkl file>
           
> Models          
>>     Condition-Monitoring/
>>                         models/
>>                               <Model.h5py file>

> Confusion Matrix (from test data)
>>     Condition-Monitoring/
>>                         confusion_matrix/
>>                                         <Confusion-matrix.csv file>



### 5.Prediction
For prediction, we have created an API which can be started by running the following command in the terminal.
```bash
python app.py
```
The API server will run on the local host of the system. To predict, we have to send a post request to the API.

Here, I used [postman](https://www.postman.com/) to test it.

1. Use the below address in URL
    > http://127.0.0.1:5000/predict

2. Change request to POST
3. In the Body section, add one new key called 'file'. Change the type from 'Text' to 'File'
4. In 'Value', browse the CSV data file and sent the request.
5. Result will be shown in the response section.

 ![Postman](/images/api.png)