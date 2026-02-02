from MeOx.components.data_ingestion import DataIngestion
from MeOx.components.data_validation import DataValidation
from MeOx.components.data_transformation import DataTransformation
from MeOx.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig, DataValidationConfig,DataTransformationConfig
from MeOx.exception.exception import MeOxException
from MeOx.logging.logger import logging
import sys,os




if __name__ == '__main__':
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info('Starting data ingestion process')
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact)
        logging.info('Data ingestion process completed')

        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_config,data_validation_config)
        logging.info('Starting data validation process')
        data_validation_artifact=data_validation.initiate_data_validation()
        print(data_validation_artifact)
        logging.info('Data validation process completed')





    except Exception as e:
        raise MeOxException(e,sys) from e