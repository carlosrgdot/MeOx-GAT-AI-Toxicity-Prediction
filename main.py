from MeOx.components.data_ingestion import DataIngestion
from MeOx.components.data_validation import DataValidation
from MeOx.components.data_transformation import DataTransformation
from MeOx.components.model_trainer import ModelTrainer
from MeOx.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig, DataValidationConfig, \
    DataTransformationConfig, ModelTrainerConfig
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

        if data_validation_artifact.validation_status:
            logging.info("Starting data transformation process")

            data_transformation_config = DataTransformationConfig(training_pipeline_config)
            data_transformation = DataTransformation(
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=data_transformation_config
            )

            data_transformation_artifact = data_transformation.initiate_data_transformation()
            print(f"Transformation Artifact: {data_transformation_artifact}")
            logging.info("Data transformation process completed")

            #logging.info("Starting model training process")

            model_trainer_config = ModelTrainerConfig(training_pipeline_config)
            model_trainer = ModelTrainer(
                model_trainer_config=model_trainer_config,
                data_transformation_artifact=data_transformation_artifact
            )

            model_trainer_artifact = model_trainer.initiate_model_trainer()
            print(f"Model Trainer Artifact: {model_trainer_artifact}")
            logging.info("Model training process completed")

        else:
            logging.error("Pipeline stopped. Data Validation failed.")
            raise Exception("Data Validation failed. Check your data schema.")


    except Exception as e:
        raise MeOxException(e,sys) from e