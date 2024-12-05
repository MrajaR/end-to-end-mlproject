from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd

import pickle
import os
import sys

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
    
def evaluate_model(x_train, y_train, x_test, y_test, models):
    logging.info("Entered the evaluate_model method of utils")
    try:
        report = {}

        for i in range(len(models.keys())):
            model = list(models.values())[i]
            params = Config.MODEL_PARAMS[list(models.keys())[i]]

            grid_search = GridSearchCV(model, params, cv=3, n_jobs=-1)
            grid_search.fit(x_train, y_train)

            model.set_params(**grid_search.best_params_)

            model.fit(x_train, y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
            
        logging.info("Exited the evaluate_model method of utils")
        return report

    except Exception as e:
        raise CustomException(e, sys)
def tune_hyperparameters(x_train, y_train, x_test, y_test, models, params):
    logging.info("Entered the tune_hyperparameters method of utils")
    try:
        report = []

        # Extract keys and values once
        model_keys = list(models.keys())
        model_values = list(models.values())

        for i in range(len(model_keys)):
            model_name = model_keys[i]
            model = model_values[i]

            # Retrieve parameters for the current model
            model_params = params.get(model_name)
            if model_params is None:
                logging.error(f"No parameters found for model: {model_name}")
                raise ValueError(f"No parameters found for model: {model_name}")


            print(f"Training model: {model_name}")
            print(f"Using parameters: {model_params}")

            # Perform GridSearchCV
            grid_search = GridSearchCV(model, model_params, cv=3, n_jobs=-1)
            grid_search.fit(x_train, y_train)

            # Update the model with the best parameters
            model.set_params(**grid_search.best_params_)
            model.fit(x_train, y_train)

            # Evaluate on the test set
            y_test_pred = model.predict(x_test)
            test_model_score = r2_score(y_test, y_test_pred)

            print(f"Successfully trained model: {model_name}")
            print('\n')
            report.append({
                'model_name': model_name,
                'best_params': grid_search.best_params_,
                'train_score_cv': grid_search.best_score_,
                'test_score': test_model_score
            })

        # Save the results to a CSV file
        pd.DataFrame(report).to_csv(Config.GRID_SEARCH_RESULT_PATH, index=False)
        logging.info("Exited the tune_hyperparameters method of utils")
        return report

    except Exception as e:
        raise CustomException(e, sys)
