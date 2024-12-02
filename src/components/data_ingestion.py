import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from dataclasses import dataclass
import os
import sys

from src.exception import CustomException
from src.logger import logging
from src.config import Config

@dataclass
class DataIngestionConfig:
    train_data_path: str = Config.TRAIN_PATH
    test_data_path: str = Config.TEST_PATH
    raw_data_path: str = Config.RAW_PATH


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion methods Starts")
        try:
            df = pd.read_csv(Config.SOURCE_PATH)
            logging.info("Dataset read as pandas dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split begins") 
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of Data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)