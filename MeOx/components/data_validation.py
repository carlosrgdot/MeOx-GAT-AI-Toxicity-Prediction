import os,sys
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split

from MeOx.constant.training_pipeline import SCHEMA_FILE_PATH
from MeOx.exception.exception import MeOxException
from MeOx.logging.logger import logging
from MeOx.entity.config_entity import DataValidationConfig,DataIngestionConfig
from MeOx.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from MeOx.utils.main_utils.utils import read_yaml_file, write_yaml_file


class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact, data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise MeOxException(e,sys) from e

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MeOxException(e,sys) from e

    def validate_columns(self,dataframe:pd.DataFrame) -> bool:
        try:
            number_of_columns = len(self._schema_config['columns'])
            logging.info(f'Required number of columns: {number_of_columns}')
            logging.info(f'Present number of columns: {len(dataframe.columns)}')
            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        except Exception as e:
            raise MeOxException(e,sys) from e

    def is_column_exist(self, dataframe: pd.DataFrame) -> bool:

        try:
            dataframe_columns = dataframe.columns
            schema_columns = self._schema_config["columns"].keys()

            missing_columns = []
            for col in schema_columns:
                if col not in dataframe_columns:
                    missing_columns.append(col)

            if len(missing_columns) > 0:
                logging.info(f"Missing columns: {missing_columns}")
                return False
            return True
        except Exception as e:
            raise MeOxException(e, sys)

    def data_drift(self,base_df,current_df,threshold=0.05) -> bool:
        try:
            status=True
            report={}
            numerical_columns = self._schema_config["numerical_columns"]
            for column in numerical_columns:
                d1=base_df[column]
                d2=current_df[column]
                missing_ratio = d2.isnull().sum() / len(d2)

                if missing_ratio > 0.3:
                    is_found = True
                    status = False
                    logging.info(f"CRITICAL DRIFT: Column '{column}' has {missing_ratio:.3%} missing values.")

                    report.update({
                        column: {
                            "p_value": 0.0,
                            "drift_status": True,
                            "message": "Too many missing values"
                        }
                    })
                    continue

                d1 = d1.dropna()
                d2 = d2.dropna()

                if d1.empty or d2.empty:
                    continue
                is_sample_dist = ks_2samp(d1,d2)
                if is_sample_dist.pvalue < threshold:
                    is_found = True
                    status = False
                    logging.info(f"Drift detected in column: {column} p_value: {is_sample_dist.pvalue}")
                else:
                    is_found = False

                report.update({
                    column: {
                        "p_value": float(is_sample_dist.pvalue),
                        "drift_status": is_found
                    }
                })

            drift_report_file_path = self.data_validation_config.drift_report_file_path
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)
            return status
        except Exception as e:
            raise MeOxException(e,sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # Read data from train and test
            train_dataframe=DataValidation.read_data(train_file_path)
            test_dataframe=DataValidation.read_data(test_file_path)
            # Validate columns
            status=self.validate_columns(dataframe=train_dataframe)
            if status:
                status = self.is_column_exist(dataframe=train_dataframe)

                # Variables para guardar las rutas finales
            valid_train_path = None
            valid_test_path = None
            invalid_train_path = None
            invalid_test_path = None

            if status:
                drift_status = self.data_drift(base_df=train_dataframe, current_df=test_dataframe)

                dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
                os.makedirs(dir_path, exist_ok=True)

                train_dataframe.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
                test_dataframe.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)

                valid_train_path = self.data_validation_config.valid_train_file_path
                valid_test_path = self.data_validation_config.valid_test_file_path

                message = "Data Validation successful."

            else:
                dir_path = os.path.dirname(self.data_validation_config.invalid_train_file_path)
                os.makedirs(dir_path, exist_ok=True)

                train_dataframe.to_csv(self.data_validation_config.invalid_train_file_path, index=False, header=True)
                test_dataframe.to_csv(self.data_validation_config.invalid_test_file_path, index=False, header=True)

                invalid_train_path = self.data_validation_config.invalid_train_file_path
                invalid_test_path = self.data_validation_config.invalid_test_file_path

                message = "Data Validation failed (Schema mismatch)."

            # --- Crear el Artefacto con tus campos exactos ---
            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=valid_train_path,
                valid_test_file_path=valid_test_path,
                invalid_train_file_path=invalid_train_path,
                invalid_test_file_path=invalid_test_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            logging.info(f"Data validation artifact created: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise MeOxException(e,sys) from e