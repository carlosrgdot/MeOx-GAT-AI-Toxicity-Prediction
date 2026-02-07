import os, sys
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split

from MeOx.constant.training_pipeline import SCHEMA_FILE_PATH
from MeOx.exception.exception import MeOxException
from MeOx.logging.logger import logging
from MeOx.entity.config_entity import DataValidationConfig, DataIngestionConfig
from MeOx.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from MeOx.utils.main_utils.utils import read_yaml_file, write_yaml_file


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise MeOxException(e, sys) from e

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MeOxException(e, sys) from e

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            number_of_columns = len(self._schema_config['columns'])
            logging.info(f'Required number of columns: {number_of_columns}')
            logging.info(f'Present number of columns: {len(dataframe.columns)}')
            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        except Exception as e:
            raise MeOxException(e, sys) from e

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

    def validate_data_types(self, dataframe: pd.DataFrame) -> bool:
        try:
            validation_status = True
            for col, rules in self._schema_config['columns'].items():
                expected_type = rules.get('type', 'object')

                if expected_type in ['float', 'int']:
                    if col in dataframe.columns:
                        if not pd.api.types.is_numeric_dtype(dataframe[col]):
                            logging.info(
                                f"Data Type Error: Column '{col}' expected {expected_type} but found non-numeric.")
                            validation_status = False
            return validation_status
        except Exception as e:
            raise MeOxException(e, sys)

    def validate_critical_variables(self, dataframe: pd.DataFrame) -> bool:
        critical_vars = ["Exposure dose (ug/mL)", "Material type", "Exposure time"]
        try:
            validation_status = True
            for col in critical_vars:
                if col in dataframe.columns:
                    if dataframe[col].isnull().sum() > 0:
                        logging.info(f"Critical Error: Variable '{col}' has null values.")
                        validation_status = False
                else:
                    logging.info(f"Critical Error: Critical variable '{col}' missing from dataframe.")
                    validation_status = False
            return validation_status
        except Exception as e:
            raise MeOxException(e, sys)

    def data_drift(self, base_df, current_df, threshold=0.05) -> bool:
        try:
            status = True
            report = {}
            numerical_columns = self._schema_config["numerical_columns"]

            for column in numerical_columns:
                if column not in base_df.columns or column not in current_df.columns:
                    continue

                d1 = base_df[column]
                d2 = current_df[column]

                missing_ratio = d2.isnull().sum() / len(d2)
                if missing_ratio > 0.3:
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

                is_sample_dist = ks_2samp(d1, d2)

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
            raise MeOxException(e, sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            data_file_path = self.data_ingestion_artifact.data_file_path
            dataframe = DataValidation.read_data(data_file_path)

            logging.info("Starting validation sequence (Single File)...")

            status = True
            error_message = ""

            if not self.validate_number_of_columns(dataframe=dataframe):
                status = False
                error_message = "Validation Failed: Column count mismatch."

            elif not self.is_column_exist(dataframe=dataframe):
                status = False
                error_message = "Validation Failed: Missing required columns."

            elif not self.validate_data_types(dataframe=dataframe):
                status = False
                error_message = "Validation Failed: Data type mismatch (Numeric check)."

            elif not self.validate_critical_variables(dataframe=dataframe):
                status = False
                error_message = "Validation Failed: Critical variables contain nulls."

            drift_report_path = self.data_validation_config.drift_report_file_path
            dir_path_report = os.path.dirname(drift_report_path)
            os.makedirs(dir_path_report, exist_ok=True)
            write_yaml_file(file_path=drift_report_path, content={"drift_status": "skipped_single_file_strategy"})

            valid_data_path = None

            if status:
                logging.info("Data Validation Successful.")

                dir_path = os.path.dirname(self.data_validation_config.valid_data_file_path)
                os.makedirs(dir_path, exist_ok=True)

                valid_data_path = os.path.join(dir_path, "data.csv")
                dataframe.to_csv(valid_data_path, index=False, header=True)

            else:
                logging.info(f"Data Validation Failed. Error: {error_message}")
                raise Exception(error_message)

            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_data_file_path=valid_data_path,
                drift_report_file_path=drift_report_path
            )

            logging.info(f"Data validation artifact created: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise MeOxException(e, sys) from e