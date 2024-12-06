from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import os


class Config:
    # Columns
    CONT_COLS = ['reading_score', 'writing_score']
    CAT_COLS = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
    TARGET_COL = 'math_score'

    # Artifacts Paths
    ARTIFACTS_DIR = "artifacts"
    PREPROCESSOR_FILE_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")
    MODEL_FILE_PATH = os.path.join("artifacts", "model.pkl")
    GRID_SEARCH_RESULT_PATH = os.path.join("artifacts", "grid_search.csv")

    # Dataset Paths
    SOURCE_PATH = "notebook/data/stud.csv"
    TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "train.csv")
    TEST_PATH = os.path.join(ARTIFACTS_DIR, "test.csv")
    RAW_PATH = os.path.join(ARTIFACTS_DIR, "data.csv")

    YAML_CONFIG_PATH = "src/config/config.yaml"
   
    CONT_PIPELINE = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
    
    CAT_PIPELINE = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder())
            ])

    PREPROCESSOR = ColumnTransformer([
            ('cont_pipeline', CONT_PIPELINE, CONT_COLS),
            ('cat_pipeline', CAT_PIPELINE, CAT_COLS)
        ])
    
