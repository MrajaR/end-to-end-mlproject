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

    # Models
    MODELS = {
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest Regressor": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "Linear Regression": LinearRegression(),
        "XGBRegressor": XGBRegressor(),
        "CatBoost Regressor": CatBoostRegressor(verbose=False),
        "AdaBoost Regressor": AdaBoostRegressor(),
    }

    # Model Parameters
    MODEL_PARAMS = {
        "Decision Tree": {
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            # Uncomment if needed
            # 'splitter': ['best', 'random'],
            # 'max_features': ['sqrt', 'log2'],
        },
        "Random Forest Regressor": {
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            # Uncomment if needed
            # 'max_features': ['sqrt', 'log2', None],
            'n_estimators': [8, 16, 32, 64, 128, 256]
        },
        "Gradient Boosting": {
            # Uncomment if needed
            # 'loss': ['squared_error', 'huber', 'absolute_error', 'quantile'],
            'learning_rate': [0.1, 0.01, 0.05, 0.001],
            'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
            # 'criterion': ['squared_error', 'friedman_mse'],
            # 'max_features': ['auto', 'sqrt', 'log2'],
            'n_estimators': [8, 16, 32, 64, 128, 256]
        },
        "Linear Regression": {},  # No parameters for Linear Regression
        "XGBRegressor": {
            'learning_rate': [0.1, 0.01, 0.05, 0.001],
            'n_estimators': [8, 16, 32, 64, 128, 256]
        },
        "CatBoost Regressor": {
            'depth': [6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'iterations': [30, 50, 100]
        },
        "AdaBoost Regressor": {
            'learning_rate': [0.1, 0.01, 0.5, 0.001],
            # Uncomment if needed
            # 'loss': ['linear', 'square', 'exponential'],
            'n_estimators': [8, 16, 32, 64, 128, 256]
        }
    }


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
    
