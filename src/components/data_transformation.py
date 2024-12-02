import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.utils import save_object
from src.logger import logging
from src.exception import CustomException
from src.config import Config

from dataclasses import dataclass, field
import os
import sys

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = Config.PREPROCESSOR_FILE_PATH
    train_path: str = Config.TRAIN_PATH
    test_path: str = Config.TEST_PATH

    target_column: str = Config.TARGET_COL
    cont_cols: list = field(default_factory=lambda: Config.CONT_COLS)
    cat_cols: list = field(default_factory=lambda: Config.CAT_COLS)

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):        
        logging.info("get transformer object initiated")
        try:
            cont_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder())
            ])

            preprocessor = ColumnTransformer([
                ('cont_pipeline', cont_pipeline, self.data_transformation_config.cont_cols),
                ('cat_pipeline', cat_pipeline, self.data_transformation_config.cat_cols)
            ])

            logging.info("Preprocessor initiated")

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def transform_data(self, train_path, test_path):
        logging.info("Data Transformation Initiated")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            feature_train_df = train_df.drop(columns=[self.data_transformation_config.target_column], axis=1)
            target_train_df = train_df[self.data_transformation_config.target_column]

            feature_test_df = test_df.drop(columns=[self.data_transformation_config.target_column], axis=1)
            target_test_df = test_df[self.data_transformation_config.target_column]

            preprocessor = self.get_data_transformer_object()

            feature_train_array = preprocessor.fit_transform(feature_train_df) # fit the transformation with the data and then transform it
            feature_test_array = preprocessor.transform(feature_test_df) # only transform the test data, not the fit since the fit is already done

            train_arr = np.c_[feature_train_array, np.array(target_train_df)]
            test_arr = np.c_[feature_test_array, np.array(target_test_df)]

            logging.info("Data Transformation Completed")

            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessor)

            logging.info("Preprocessor pickle file saved")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)