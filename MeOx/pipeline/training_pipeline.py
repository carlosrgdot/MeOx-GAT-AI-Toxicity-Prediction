import os,sys
import json
from MeOx.exception.exception import MeOxException
from MeOx.logging.logger import logging
from MeOx.components.data_ingestion import DataIngestion
from MeOx.components.data_validation import DataValidation
from MeOx.components.data_transformation import DataTransformation
from MeOx.components.model_trainer import ModelTrainer
from MeOx.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from MeOx.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, ModelTrainerArtifact





class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

    def start_data_ingestion(self):
        try:
            logging.info(">>>>> STAGE 1: Data Ingestion Started <<<<<")

            data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)

            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            logging.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")
            logging.info(">>>>> STAGE 1: Data Ingestion Completed <<<<<\n")
            return data_ingestion_artifact

        except Exception as e:
            raise MeOxException(e,sys) from e

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact):
        try:
            logging.info(">>>>> STAGE 2: Data Validation Started <<<<<")

            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=data_validation_config
            )

            data_validation_artifact = data_validation.initiate_data_validation()

            logging.info(f"Data Validation Artifact: {data_validation_artifact}")
            logging.info(">>>>> STAGE 2: Data Validation Completed <<<<<\n")
            return data_validation_artifact
        except Exception as e:
            raise MeOxException(e,sys) from e

    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact):
        try:
            if not data_validation_artifact.validation_status:
                raise Exception("Data Validation failed. Pipeline stopped.")

            logging.info(">>>>> STAGE 3: Data Transformation Started <<<<<")

            data_transformation_config = DataTransformationConfig(
                training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=data_transformation_config
            )

            data_transformation_artifact = data_transformation.initiate_data_transformation()

            logging.info(f"Data Transformation Artifact: {data_transformation_artifact}")
            logging.info(">>>>> STAGE 3: Data Transformation Completed <<<<<\n")
            return data_transformation_artifact

        except Exception as e:
            raise MeOxException(e,sys) from e

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact)-> ModelTrainerArtifact:
        try:
            logging.info(">>>>> STAGE 4: Model Trainer Started <<<<<")

            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(
                model_trainer_config=model_trainer_config,
                data_transformation_artifact=data_transformation_artifact
            )

            model_trainer_artifact = model_trainer.initiate_model_trainer()

            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            logging.info(">>>>> STAGE 4: Model Trainer Completed <<<<<\n")
            return model_trainer_artifact
        except Exception as e:
            raise MeOxException(e,sys) from e

    def run_pipeline(self):
        try:
            logging.info("================ PIPELINE STARTED ================")

            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)

            try:
                real_score = model_trainer_artifact.test_metric_artifact.balanced_accuracy

                metrics_data = {
                    "accuracy": float(real_score),
                    "materials": 15,
                    "status": "Active (Retrained)"
                }

                json_path = "metrics.json"
                with open(json_path, "w") as f:
                    json.dump(metrics_data, f)

                logging.info(f" Dashboard metrics updated successfully: {metrics_data}")

            except Exception as e:
                logging.error(f" Could not update dashboard metrics: {str(e)}")
            # =================================================================

            logging.info("================ PIPELINE COMPLETED ================")
            return model_trainer_artifact

        except Exception as e:
            raise MeOxException(e, sys) from e