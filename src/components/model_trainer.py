from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


from src.exception import CustomException
from src.logger import logging
from src.config import Config
from src.utils import save_object, tune_hyperparameters
from dataclasses import dataclass, field

import os
import sys

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = Config.MODEL_FILE_PATH
    preprocessor_obj_file_path: str = Config.PREPROCESSOR_FILE_PATH

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def train_model(self, train_array, test_array):
        try:
            logging.info("Splitting Dependent and Independent variables from train and test data")
            X_train, y_train, X_test, y_test = train_array[:, :-1], train_array[:, -1], test_array[:, :-1], test_array[:, -1]     
            hyperparams_report = tune_hyperparameters(X_train, y_train, 
                                                      X_test, y_test, 
                                                      Config.MODELS, Config.MODEL_PARAMS)

            best_model_info = max(hyperparams_report, key=lambda x: x['test_score'])
            best_model_name = best_model_info['model_name']
            best_params = best_model_info['best_params']

            best_model = Config.MODELS[best_model_name]
            best_model.set_params(**best_params)

            best_model.fit(X_train, y_train)
            logging.info(f'Successfully trained the best model which is {best_model_name.upper()} model')

            logging.info("Saving the best model to file")
            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            logging.info('Successfully saved the best model')

            return X_test, y_test # RETURN THESE FOR EVALUATION PHASE

        except Exception as e:
            raise CustomException(e, sys)