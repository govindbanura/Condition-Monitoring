from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
import datetime
from predict import predict_class
from termcolor import colored
import logging
# from werkzeug.utils import secure_filename
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'file_upload'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['csv'])

log_filename = './logs/' + 'api_log' + datetime.datetime.now().strftime("_%d_%m_%Y_%H_%M") + ".log"
logging.basicConfig(filename=log_filename, filemode='w', level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        logging.error('No file part in the request')
        resp = jsonify({'Error' : 'No file part in the request'})
        resp.status_code = 400
        return resp

    file = request.files['file']
    if file.filename == '':
        logging.error('No file selected for uploading')
        resp = jsonify({'Error' : 'No file selected for uploading'})
        resp.status_code = 400
        return resp

    if file and allowed_file(file.filename):
        filename = file.filename
        new_filename = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+'.'+filename.rsplit('.', 1)[1].lower()
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
        file.save(file_path)
        logging.info('Data file saved successful')
        try:
            label_class = predict_class(file_path)
            logging.info('Predicted Class: ' + label_class)
            resp = jsonify({'Predicted Class' : label_class})
            resp.status_code = 201
            return resp
        except Exception as e:
            logging.error(e)
            resp = jsonify({'Error' : str(e)})
            resp.status_code = 400
            return resp
    else:
        logging.error('Allowed file types are ["CSV"]')
        resp = jsonify({'Error' : 'Allowed file types are ["CSV"]'})
        resp.status_code = 400
        return resp

if __name__ == '__main__':
    try:
        app.run('0.0.0.0')
    except Exception as e:
        logging.error('Api run failed due to '+e)