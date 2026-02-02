import os,sys
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from MeOx.exception.exception import MeOxException
from MeOx.logging.logger import logging
from MeOx.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from MeOx.entity.config_entity import DataTransformationConfig
from MeOx.utils.main_utils.utils import save_object, save_numpy_array_data
from MeOx.constant.training_pipeline import TARGET_COLUM, DATA_TRANSFORMATION_IMPUTER_PARAMS


class DataTransformation:

    def __init__(self,data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact:DataValidationArtifact = data_validation_artifact
            self.data_transformation_config:DataTransformationConfig = data_transformation_config

        except Exception as e:
            raise MeOxException(e,sys) from e

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MeOxException(e,sys) from e

    def get_data_transformer_object(cls) -> Pipeline:
        logging.info('Entered get_data_transformer_object method of Data_Transformation class')

        try:
            imputer:KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info(f'Initialise KNNImputer with {DATA_TRANSFORMATION_IMPUTER_PARAMS}')
            processor:Pipeline= Pipeline(steps=[('imputer', imputer),])
            return processor
        except Exception as e:
            raise MeOxException(e,sys) from e



    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Initiating Data Transformation")
            train_df=DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df=DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            #Trainig dataframe
            input_feature_train_df = train_df.drop(columns=['TARGET_COLUM'],axis=1)
            target_feature_train_df = train_df[TARGET_COLUM]
            target_feature_train_df = target_feature_train_df.replace(-1,0)


            #Testing dataframe
            input_feature_test_df = test_df.drop(columns=['TARGET_COLUM'],axis=1)
            target_feature_test_df = test_df[TARGET_COLUM]
            target_feature_test_df = target_feature_test_df.replace(-1,0)

            #Data transformation
            preprocessor = self.get_data_transformer_object()
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature=preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature=preprocessor_object.transform(input_feature_test_df)

            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]

            #save numpy array
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array= train_arr,)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array= test_arr,)
            save_object(self.data_validation_config.transformed_object_file_path, obj= preprocessor_object,)

            #prepare artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )




        except Exception as e:
            raise MeOxException(e,sys) from e