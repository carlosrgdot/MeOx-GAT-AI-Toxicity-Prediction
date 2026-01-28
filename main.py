from MeOx.components.data_ingestion import DataIngestion
from MeOx.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig
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

    except Exception as e:
        raise MeOxException(e,sys) from e