import pickle
import os
import sys

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