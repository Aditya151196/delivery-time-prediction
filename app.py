from flask import Flask,render_template,request,jsonify
from src.pipeline.prediction_pipeline import PredictionPipeline,CustomData
import os
from werkzeug.utils import secure_filename

from prediction.batch_prediction import batch_prediction
from src.components.data_transformation import DataTransformationConfig
from src.pipeline.training_pipeline import Train
from src.logger import logging
from src.exception import CustomException
from src.config.configuration import *

feature_engineering_file_path = FEATURE_ENG_OBJ_PATH
transformer_file_path = PREPROCESSING_OBJ_PATH
model_file_path = MODEL_FILE_PATH

upload_folder = 'batch_prediction/uploaded_csv_file'

application = Flask(__name__,template_folder='templates')
app = application

ALLOWED_EXTENSIONS = {'csv'}

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    
    else:
        data = CustomData(
            Delivery_person_Age = int(request.form.get('Delivery_person_Age')),
            Delivery_person_Ratings = float(request.form.get('Delivery_person_Ratings')),
            Weather_conditions = request.form.get('Weather_conditions'),
            Road_traffic_density = request.form.get('Road_traffic_density'),
            Vehicle_condition = int(request.form.get('Vehicle_condition')),
            multiple_deliveries = int(request.form.get('Multiple_deliveries')),
            distance = float(request.form.get('Distance')),
            Type_of_order = request.form.get('Type_of_order'),
            Type_of_vehicle = request.form.get('Type_of_vehicle'),
            Festival = request.form.get('Festival'),
            City = request.form.get('City')
            
        )

        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictionPipeline()
        pred = predict_pipeline.predict(final_new_data)

        result = int(pred[0])

        return render_template('form.html',final_result=result)
    
@app.route("/batch",methods=['GET','POST'])
def perform_batch_prediction():
    if request.method == 'GET':
        return render_template('batch.html')
    else:
        file = request.files['csv_file']
        
        directory_path = upload_folder
        os.makedirs(directory_path,exist_ok=True)

        if file and '.' in file.filename and file.filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS:
            for filename in os.listdir(os.path.join(upload_folder)):
                file_path = os.path.join(upload_folder,filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

            # save the new file to the upload directory
            filename = secure_filename(file.filename)
            file_path = os.path.join(upload_folder,filename)
            file.save(file_path)
            print(file_path)

            # Performing batch prediction
            batch = batch_prediction(file_path,model_file_path,
                                     transformer_file_path,feature_engineering_file_path)
            batch.start_batch_prediction()

            output = "Batch Prediction Done"
            return render_template("batch.html",prediction_result = 'output', prediction_type='batch')
        else:
            return render_template("batch.html",prediction_type="batch",error="Invalid file type")
        
@app.route('/train',methods=['GET','POST'])
def train():
    if request.method == "GET":
        return render_template('train.html')
    else:
        try:
            pipeline = Train()
            pipeline.main()

            return  render_template('train_html',message = "Training complete")
        except Exception as e:
            logging.error(f"{e}")
            error_message = str(e)
            return render_template('index.html',error=error_message)
        
if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True,port="8888")
