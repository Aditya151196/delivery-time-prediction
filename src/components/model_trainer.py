from src.constants import *
from src.logger import logging
from src.exception import CustomException
from src.config.configuration import *
import os,sys
from dataclasses import dataclass
from sklearn.base import BaseEstimator,TransformerMixin
import numpy as np
import pandas as pd
from src.utils import save_obj,evaluate_model

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression,Lasso,Ridge

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = MODEL_FILE_PATH

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("Splitting dependent and independent data from train and test data")
            X_train, y_train,X_test,y_test = (train_array[: ,:-1],train_array[:,-1],
                                             test_array[: , :-1], test_array[:,-1] )
            
            models = {
                'Linear Regression': LinearRegression(),
                'Decision Tree' : DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Random Forest' : RandomForestRegressor(),
                'XGB Regressor' : XGBRegressor()
                }
            
            model_report:dict = evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)

            print("\n==============================================================\n")
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f'Best Model Found, Model Name : {best_model_name}, R2 Score : {best_model_score}')
            print("\n======================================================================\n")
            logging.info(f'Best Model Found, Model Name : {best_model_name}, R2 Score : {best_model_score}')

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
        except Exception as e:
            logging.info("Exception occured while training the model")
            raise CustomException(e,sys)
        
