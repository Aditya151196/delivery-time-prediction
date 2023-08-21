from src.constants import *
from src.config.configuration import *
from src.exception import CustomException
from src.logger import logging
import os,sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation,DataTransformationConfig
from src.components.model_trainer import ModelTrainer,ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    
    train_file_path:str=TRAIN_FILE_PATH
    test_file_path:str=TEST_FILE_PATH
    raw_file_path:str=RAW_FILE_PATH


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()


    def initiate_data_ingestion(self):
        try:
            logging.info("Reading dataset")
            df = pd.read_csv(DATASET_FILE_PATH)

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_file_path),exist_ok=True)

            df.to_csv(self.data_ingestion_config.raw_file_path,index=False)

            logging.info("Train-test split started")
            train_set,test_set = train_test_split(df,train_size=0.80,random_state=35)
            logging.info("Train-test split completed")

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_file_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.data_ingestion_config.test_file_path),exist_ok=True)

            train_set.to_csv(self.data_ingestion_config.train_file_path,header=True)
            test_set.to_csv(self.data_ingestion_config.test_file_path,header=True)

            return(
                self.data_ingestion_config.train_file_path,
                self.data_ingestion_config.test_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)


#if __name__ == "__main__":
 #   obj = DataIngestion()
 #   train_data, test_data = obj.iniitiate_data_ingestion()
 #   data_transformation = DataTransformation()
 #   train_arr, test_arr = data_transformation.inititate_data_transformation(train_data, test_data)

# Data Transformation

#if __name__ == "__main__":
 #   obj = DataIngestion()
  #  train_data_path,test_data_path=obj.iniitiate_data_ingestion()
   # data_transformation = DataTransformation()
    #train_arr,test_arr,_ = data_transformation.inititate_data_transformation(train_data_path,test_data_path)


# Model Training 

if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initaite_data_transformation(train_data_path,test_data_path)
    model_trainer = ModelTrainer()
    print(model_trainer.initate_model_training(train_arr,test_arr))

    