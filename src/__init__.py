from src.logger import logging
from src.exception import CustomException
import os,sys
import pickle

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_path)
    
    except Exception as e:
        logging.info("Exception occured while saving an object")
        raise CustomException(e,sys)