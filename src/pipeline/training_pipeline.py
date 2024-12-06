from src.components import (DataIngestion,
                            DataTransformation,
                            ModelTrainer,
                            ModelEvaluation)
from src.exception import CustomException
from src.logger import logging
from src.config import Config
import os
import sys

class TrainingPipeline:
    def __init__(self):
        """Initialize the TrainingPipeline with configurations and components."""
        self.config = Config()
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
        self.model_evaluation = ModelEvaluation()

    def run_pipeline(self):
        """
        Execute the entire machine learning pipeline:
        - Data Ingestion
        - Data Transformation
        - Model Training
        - Model Evaluation
        """
        try:
            logging.info("Starting the training pipeline.")
            
            # Step 1: Data Ingestion
            logging.info("Step 1: Data Ingestion started.")
            train_path, test_path, raw_data_path = self.data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion completed. train data saved at {train_path}, test data saved at {test_path}, raw data saved at {raw_data_path}.")
            
            # Step 2: Data Transformation
            logging.info("Step 2: Data Transformation started.")
            train_array, test_array, preprocessor_obj_file_path = self.data_transformation.transform_data(train_path, test_path)
            logging.info(f"Data Transformation completed and Processor object saved at {preprocessor_obj_file_path}.")
            
            # Step 3: Model Training
            logging.info("Step 3: Model Training started.")
            X_test, y_test = self.model_trainer.train_model(train_array, test_array)
            logging.info(f"Model Training completed. Model saved at {self.config.MODEL_FILE_PATH}.")
            
            # Step 4: Model Evaluation
            logging.info("Step 4: Model Evaluation started.")
            evaluation_metrics = self.model_evaluation.evaluate_model(X_test, y_test)
            logging.info(f"Model Evaluation completed. {evaluation_metrics}.")
            
            logging.info("Training pipeline executed successfully.")
        
        except CustomException as e:
            logging.error(f"CustomException occurred: {str(e)}")
            raise e
        except Exception as e:
            logging.error(f"An unexpected error occurred: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    training_pipeline = TrainingPipeline()
    training_pipeline.run_pipeline()
