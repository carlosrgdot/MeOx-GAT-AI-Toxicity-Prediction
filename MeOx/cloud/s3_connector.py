import os,sys
import boto3
from MeOx.logging.logger import logging
from MeOx.exception.exception import MeOxException
import certifi

class S3Connector:
    def __init__(self):
        try:
            self.s3_client = boto3.client(
                's3',
                verify=certifi.where()
            )
        except Exception as e:
            raise MeOxException(e, sys)

    def download_file(self, bucket_name, s3_key, local_file_path):
        try:
            logging.info(f'Downloading file from s3://{bucket_name}/{s3_key} to {local_file_path}')
            dir_path = os.path.dirname(local_file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            self.s3_client.download_file(bucket_name, s3_key, local_file_path)
            logging.info(f'File downloaded successfully to {local_file_path}')

        except Exception as e:
            raise MeOxException(e, sys)

    def upload_file_to_s3(self, local_file_path, bucket_name, s3_key):
        try:
            logging.info(f"Uploading file {local_file_path} to S3 bucket {bucket_name} with key {s3_key}")
            self.s3_client.upload_file(local_file_path, bucket_name, s3_key)
            logging.info(f"File {local_file_path} uploaded successfully to S3 bucket {bucket_name} with key {s3_key}")
        except Exception as e:
            raise MeOxException(e, sys)