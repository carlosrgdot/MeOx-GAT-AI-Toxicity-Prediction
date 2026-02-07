import sys,os
import pandas as pd
import numpy as np

"""
Defining common constants variables for training pipeline
"""
TARGET_COLUM = 'Toxicity' # Target column for prediction
PIPELINE_NAME: str = 'MeOx'
ARTIFACT_DIR:str = 'artifacts'
FILE_NAME:str = 'MeOx_data.csv'
TRAIN_FILE_NAME: str = 'train.csv'
TEST_FILE_NAME: str = 'test.csv'

SCHEMA_FILE_PATH = os.path.join('data_schema','schema.yaml')
SAVED_MODEL_DIR=os.path.join('saved_models')
DATA_FILE_NAME: str = "data.csv"


"""
Data Ingestion related constants start with DATA INGESTION_
"""
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2
DATA_INGESTION_S3_KEY: str = "Metal_Oxide_cytotoxicityCSV.csv"
DATA_INGESTION_BUCKET_NAME: str = "meoxdockeredition"



"""
Data Validation related constants start with DATA VALIDATION_
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"



"""
Data Transformation related constants start with DATA TRANSFORMATION_
"""

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessing.pkl"
TRANSFORMED_DATA_FILE_NAME: str = "toxic_graph_data.pt"

DATA_TRANSFORMATION_NUMERICAL_COLS = [
    "Exposure dose (ug/mL)",
    "Exposure time"
]
DATA_TRANSFORMATION_CATEGORICAL_COLS = [
    "Material type"
]
TARGET_COLUMN = "Toxicity"
GRAPH_DBSCAN_METRIC = 'cosine'
GRAPH_MIN_SAMPLES = 3
GRAPH_EPS_QUANTILE = 0.98
GRAPH_K_INTRA_CLUSTER = 5


"""
Model Trainer related constants start with MODEL TRAINER_
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_FILE_NAME: str = "model.pth"


MODEL_TRAINER_RANDOM_SEED: int = 43
MODEL_TRAINER_OPTUNA_TRIALS: int = 100
MODEL_TRAINER_CV_FOLDS: int = 5
MODEL_TRAINER_TEST_SIZE: float = 0.2
MODEL_TRAINER_MAX_EPOCHS_OPTUNA: int = 60
MODEL_TRAINER_PATIENCE_OPTUNA: int = 10
MODEL_TRAINER_MAX_EPOCHS_FINAL: int = 300
MODEL_TRAINER_PATIENCE_FINAL: int = 30
MODEL_TRAINER_EXPECTED_BALANCED_ACCURACY: float = 0.95
MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD : float = 0.05

