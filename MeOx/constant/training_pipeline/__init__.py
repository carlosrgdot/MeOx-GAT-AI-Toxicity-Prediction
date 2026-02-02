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

#knn imputer to replace nan values
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": "uniform",
}