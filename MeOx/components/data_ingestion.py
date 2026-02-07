import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from MeOx.exception.exception import MeOxException
from MeOx.logging.logger import logging
from MeOx.entity.config_entity import DataIngestionConfig
from MeOx.entity.artifact_entity import DataIngestionArtifact
from MeOx.cloud.s3_connector import S3Connector


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            self.s3 = S3Connector()
        except Exception as e:
            raise MeOxException(e, sys)

    def download_data_from_s3(self) -> pd.DataFrame:
        try:
            bucket_name = self.data_ingestion_config.bucket_name
            s3_key = self.data_ingestion_config.s3_key
            local_file_path = self.data_ingestion_config.feature_store_file_path
            logging.info(f"Downloaded from s3: {bucket_name}/{s3_key}")
            self.s3.download_file(
                bucket_name=bucket_name,
                s3_key=s3_key,
                local_file_path=local_file_path
            )
            logging.info(f"Data downloaded in: {local_file_path}")
            dataframe = pd.read_csv(local_file_path)

            return dataframe

        except Exception as e:
            raise MeOxException(e, sys)

    def save_full_data(self, dataframe: pd.DataFrame) -> str:
        try:
            dir_path = os.path.dirname(self.data_ingestion_config.data_file_path)
            os.makedirs(dir_path, exist_ok=True)

            data_file_path = os.path.join(dir_path, "data.csv")

            dataframe.to_csv(
                data_file_path, index=False, header=True
            )
            logging.info(f"Full dataset saved at: {data_file_path}")
            return data_file_path

        except Exception as e:
            raise MeOxException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            dataframe = self.download_data_from_s3()
            data_file_path = self.save_full_data(dataframe)

            data_ingestion_artifact = DataIngestionArtifact(
                data_file_path=data_file_path
            )
            return data_ingestion_artifact

        except Exception as e:
            raise MeOxException(e, sys)