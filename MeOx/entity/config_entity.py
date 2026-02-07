from datetime import datetime
import os,sys
from MeOx.constant import training_pipeline
from MeOx.exception.exception import MeOxException
from MeOx.logging.logger import logging

class TrainingPipelineConfig:
    def __init__(self,timestamp=datetime.now()):
        timestamp = timestamp.strftime("%Y%m%d%H%M%S")
        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifact_name = training_pipeline.ARTIFACT_DIR
        self.artifact_dir = os.path.join(self.artifact_name, timestamp)
        self.timestamp: str = timestamp

class DataIngestionConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:
            self.data_ingestion_dir:str = os.path.join(training_pipeline_config.artifact_dir,training_pipeline.DATA_INGESTION_DIR_NAME)
            self.feature_store_file_path:str = os.path.join(self.data_ingestion_dir,training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR, training_pipeline.FILE_NAME)
            self.data_file_path: str = os.path.join(self.data_ingestion_dir,training_pipeline.DATA_INGESTION_INGESTED_DIR,training_pipeline.DATA_FILE_NAME)
            self.bucket_name: str = training_pipeline.DATA_INGESTION_BUCKET_NAME
            self.s3_key: str = training_pipeline.DATA_INGESTION_S3_KEY
        except Exception as e:
            raise MeOxException(e,sys) from e

class DataValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:
            self.data_validation_dir: str = os.path.join( training_pipeline_config.artifact_dir, training_pipeline.DATA_VALIDATION_DIR_NAME)
            self.valid_data_dir: str = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_VALID_DIR)
            self.invalid_data_dir: str = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_INVALID_DIR)
            self.valid_data_file_path: str = os.path.join(self.valid_data_dir,training_pipeline.DATA_FILE_NAME)
            self.invalid_data_file_path: str = os.path.join(self.invalid_data_dir,training_pipeline.DATA_FILE_NAME)
            self.drift_report_file_path: str = os.path.join(self.data_validation_dir,training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,)
        except Exception as e:
            raise MeOxException(e,sys) from e


class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            self.data_transformation_dir: str = os.path.join(
                training_pipeline_config.artifact_dir,
                training_pipeline.DATA_TRANSFORMATION_DIR_NAME
            )

            self.transformed_data_file_path: str = os.path.join(
                self.data_transformation_dir,
                training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                training_pipeline.TRANSFORMED_DATA_FILE_NAME)

            self.transformed_object_file_path: str = os.path.join(
                self.data_transformation_dir,
                training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                training_pipeline.PREPROCESSING_OBJECT_FILE_NAME,
            )
        except Exception as e:
            raise MeOxException(e, sys) from e


class ModelTrainerConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir, training_pipeline.MODEL_TRAINER_DIR_NAME)
        self.trained_model_file_path: str = os.path.join(self.model_trainer_dir, training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR, training_pipeline.MODEL_FILE_NAME)
        self.expected_balanced_accuracy: float = training_pipeline.MODEL_TRAINER_EXPECTED_BALANCED_ACCURACY
        self.overfitting_underfitting_threshold: float = training_pipeline.MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD