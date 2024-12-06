from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd

import yaml
import pickle
import os
import sys
import importlib

from src.config import Config
from src.logger import logging
from src.exception import CustomException

def save_object(file_path, obj):
    logging.info("Entered the save_object method of utils")
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info("Exited the save_object method of utils")
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    logging.info("Entered the load_object method of utils")
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)

        logging.info("Exited the load_object method of utils")
        return obj

    except Exception as e:
        raise CustomException(e, sys)
    
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_model_instance(model_config):
    """
    Dynamically loads the model class and initializes it.
    """
    module_name, class_name = model_config['class'].rsplit('.', 1)
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    return model_class()

def tune_hyperparameters(x_train, y_train, x_test, y_test, config_path):
    logging.info("Entered the tune_hyperparameters method of utils")
    try:
        report = []

        # Load models and parameters from YAML config
        config = load_config(config_path)
        models_config = config['models']

        for model_name, model_config in models_config.items():
            model = get_model_instance(model_config)
            model_params = model_config.get('params', {})

            print(f"Training model: {model_name}")
            print(f"Using parameters: {model_params}")

            grid_search = GridSearchCV(model, model_params, cv=3, n_jobs=-1)
            grid_search.fit(x_train, y_train)

            # Use the best estimator and evaluate it
            best_model = grid_search.best_estimator_
            y_test_pred = best_model.predict(x_test)
            test_model_score = r2_score(y_test, y_test_pred)

            print(f"Successfully trained model: {best_model}")
            print('\n')
            report.append({
                'model': best_model,
                'best_params': grid_search.best_params_,
                'train_score_cv': grid_search.best_score_,
                'test_score': test_model_score,
            })

        pd.DataFrame(report).to_csv("grid_search_results.csv", index=False)
        logging.info("Exited the tune_hyperparameters method of utils")
        return report

    except Exception as e:
        raise Exception(f"Error in tune_hyperparameters: {e}")