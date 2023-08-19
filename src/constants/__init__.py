import os,sys
from datetime import datetime

def get_current_timestamp():
    return f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

CURRENT_TIME_STAMP=get_current_timestamp()

ROOT_DIR_KEY=os.getcwd()
DATASET_DIR="Data"
DATASET_NAME="finalTrain.csv"

#Artifact dir
ARTIFACT_DIR_NAME="Artifact"

#Data ingestion related variables
DATA_INGESTION_DIR="data_ingestion"
RAW_DATA_DIR_NAME="raw_data_dir"
RAW_DATA_FILE_NAME="raw.csv"
INGESTED_DATA_DIR_NAME="ingested_data"
TRAIN_DATA_FILE_NAME="train.csv"
TEST_DATA_FILE_NAME="test.csv"


# Data transformation related variables
DATA_TRANSFORMATION_ARTIFACT = "data_transformation"
DATA_PREPROCESSED_DIR = "preprocessed"
DATA_TRANSFORMATION_PREPROCESSING_OBJ = "preprocessor.pkl"
DATA_TRANSFORMED_DIR = "transformed_data"
TRANSFORMED_TRAIN_DIR_KEY = "train.csv"
TRANSFORMED_TEST_DIR_KEY = "test.csv"
