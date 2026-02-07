import sys
from MeOx.entity.artifact_entity import ClassificationMetricArtifact
from MeOx.exception.exception import MeOxException
from MeOx.logging.logger import logging
from MeOx.constant.training_pipeline import SAVED_MODEL_DIR,MODEL_FILE_NAME


class MeOxModel:
    def __init__(self,processor,model):
        try:
            self.processor=processor
            self.model=model
        except Exception as e:
            raise MeOxException(e,sys) from e

    def predict(self,x):
        try:
            x_transform = self.processor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise MeOxException(e,sys) from e