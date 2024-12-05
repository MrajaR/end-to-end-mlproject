from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import numpy as np
import pickle
import sys

from dataclasses import dataclass

from src.config import Config
from src.logger import logging
from src.exception import CustomException

@dataclass
class ModelEvaluationConfig:
    model_path: str = Config.MODEL_FILE_PATH


class ModelEvaluation:
    def __init__(self):
        self.config = ModelEvaluationConfig()

    def evaluate_model(self, x_test, y_test):
        logging.info("Entered the evaluate_model method of Model Evaluation class")
        try:
            model = pickle.load(open(self.config.model_path, "rb"))
            y_test_pred = model.predict(x_test)

            mae = mean_absolute_error(y_test, y_test_pred)
            mse = mean_squared_error(y_test, y_test_pred)
            mape = mean_absolute_percentage_error(y_test, y_test_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_test_pred)

            logging.info("Exited the evaluate_model method of Model Evaluation class")
            return f"Best Model Metrics:\nMAE: {mae}\nMSE: {mse}\nRMSE: {rmse}\nMAPE: {mape}\nR2: {r2}"

        except Exception as e:
            raise CustomException(e, sys)
