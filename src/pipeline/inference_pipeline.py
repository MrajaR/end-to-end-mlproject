import numpy as np
import pandas as pd

import os
import sys

from src.utils import load_object
from src.config import Config
from src.logger import logging
from src.exception import CustomException

class InferencePipeline:
    def __init__(self):
        logging.info("Entered the predict method of InferencePipeline: class in inference pipeline")
        try:
            self.model = load_object(Config.MODEL_FILE_PATH)
            self.preprocessor = load_object(Config.PREPROCESSOR_FILE_PATH)
        except Exception as e:
            logging.error(f"Error occured in initialization of InferencePipeline: class in inference pipeline {e}")
            raise CustomException(e, sys)

    def predict(self, input_feature):
        logging.info("Entered the predict method of InferencePipeline: class in inference pipeline")
        try:
            processed_input = self.preprocessor.transform(input_feature)
            return self.model.predict(processed_input)
        except Exception as e:
            logging.error(f"Error occured in predict method of InferencePipeline: class in inference pipeline {e}")
            raise CustomException(e, sys)
        

class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):
        try:
            logging.info('Entered the CustomData class in inference pipeline')
            self.gender = gender
            self.race_ethnicity = race_ethnicity
            self.parental_level_of_education = parental_level_of_education
            self.lunch = lunch
            self.test_preparation_course = test_preparation_course
            self.reading_score = reading_score
            self.writing_score = writing_score
        except Exception as e:
            logging.error(f"Error occured in initialization of CustomData class in inference pipeline {e}")
            raise CustomException(e, sys)

    def get_data_as_dataframe(self):
        logging.info("Entered the get_data_as_dataframe method of CustomData class in inference pipeline")
        try:
            input_df = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score" : [self.reading_score],
                "writing_score": [self.writing_score]
            }

            return pd.DataFrame(input_df)
        except Exception as e:
            logging.error(f"Error occured in in get_data_as_dataframe method of CustomData class in inference pipeline {e}")
            raise CustomException(e, sys)