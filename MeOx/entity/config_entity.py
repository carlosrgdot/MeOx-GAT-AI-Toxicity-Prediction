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
            self.train_file_path:str = os.path.join(self.data_ingestion_dir,training_pipeline.DATA_INGESTION_INGESTED_DIR, training_pipeline.TRAIN_FILE_NAME)
            self.test_file_path:str = os.path.join(self.data_ingestion_dir,training_pipeline.DATA_INGESTION_INGESTED_DIR, training_pipeline.TEST_FILE_NAME)
            self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
            self.bucket_name: str = training_pipeline.DATA_INGESTION_BUCKET_NAME
            self.s3_key: str = training_pipeline.DATA_INGESTION_S3_KEY
        except Exception as e:
            raise MeOxException(e,sys) from e
