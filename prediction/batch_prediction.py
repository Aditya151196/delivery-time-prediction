from src.constants import *
from src.config.configuration import *
from src.logger import logging
import os,sys
import pandas as pd
import numpy as np
from src.exception import CustomException
import pickle
from src.utils import load_model
from sklearn.pipeline import Pipeline

PREDICTION_FOLDER = "batch_prediction"
PREDICTION_CSV_FILE = "prediction.csv"
PREDICTION_OUTPUT_FILE = "prediction_out.csv"

FEATURE_ENG_FOLDER = 'feature_eng'

ROOT_DIR = os.getcwd()
FEATURE_ENG_PATH = os.path.join(ROOT_DIR,PREDICTION_FOLDER,FEATURE_ENG_FOLDER)
BATCH_PREDICTION = os.path.join(ROOT_DIR,PREDICTION_FOLDER,PREDICTION_CSV_FILE)

class batch_prediction:
    def __init__(self,input_file_path,
                 model_file_path,feature_engineering_file_path,
                 transformer_file_path) -> None :
        self.input_file_path = input_file_path
        self.model_file_path = model_file_path
        self.feature_engineering_file_path = feature_engineering_file_path
        self.transformer_file_path = transformer_file_path

    def start_batch_prediction(self):
        try:
            logging.info("Loading the saved pipeline")

            # Load the feature engineering pipeline
            with open(self.feature_engineering_file_path,'rb') as f:
                feature_pipeline = pickle.load(f)

            logging.info(f"feature engineering object accessed : {self.feature_engineering_file_path}")

            logging.info(f"Loading the transformation pipeline")
            with open(self.transformer_file_path,'rb') as f:
                preprocessor = pickle.load(f)

            logging.info(f"Preprocessor object accessed : {self.transformer_file_path}")

            logging.info(f"Loading model file object")
            with open(self.model_file_path,'rb') as f:
                model = load_model(file_path = self.model_file_path)

            logging.info(f"Model file object accessed : {self.model_file_path}")

            # create the feature engineering pipeline
            feature_engineering_pipeline = Pipeline([
                ('feature_engineering',feature_pipeline)
            ])

            # read the input file
            df = pd.read_csv(self.input_file_path)

            df.to_csv("df_delivery_time.csv") 

            # Applying feature engineering
            df = feature_engineering_pipeline.transform(df)

            df.to_csv("transformed_df.csv")    

            # Save the feature engineering data set
            FEATURE_ENG_PATH = FEATURE_ENG_PATH  # Specify the desired path for 
            os.makedirs(FEATURE_ENG_PATH,exist_ok=True)
            file_path = os.path.join(FEATURE_ENG_PATH,"batch_feature_eng.csv")
            df.to_csv(file_path,index=False)
            logging.info("Feature engineered batch saved in csv file")

            # Dropping target column
            df.drop("Time_taken (min)",axis=1,inplace=True)

            df.to_csv('delivery_time_pred_before_transfomration.csv')

            logging.info(f"Columns before transformation: {', '.join(f'{col}: {df[col].dtype}' for col in df.columns)}")
            # Transform the feature-engineered data using the preprocessor
            transformed_data = preprocessor.transform(df)
            logging.info(f"Transformed data shape: {transformed_data.shape}")

            file_path = os.path.join(FEATURE_ENG_PATH,'preprocessor.csv')

            logging.info(f"Model Data type : {type(model)}")
            predictions = model.predict(transformed_data)
            logging.info(f"Prediction completed on transformed data : {predictions}")

            # Creating dataframe from predictions
            df_predictions = pd.DataFrame(predictions,columns=['prediction'])

            # Save the predictions to a csv file
            BATCH_PREDICTION_PATH = BATCH_PREDICTION
            os.makedirs(BATCH_PREDICTION_PATH,exist_ok=True)

            predictions_csv = os.path.join(BATCH_PREDICTION_PATH,'predictions.csv')
            df_predictions.to_csv(predictions_csv,index=False)
            logging.info(f"Batch predictions saved to : {predictions_csv}")

        except Exception as e:
            raise CustomException(e,sys)




