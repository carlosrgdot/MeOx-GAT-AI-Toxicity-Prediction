import sys
from MeOx.entity.artifact_entity import ClassificationMetricArtifact
from MeOx.exception.exception import MeOxException
from MeOx.logging.logger import logging
from sklearn.metrics import f1_score,accuracy_score,balanced_accuracy_score

def get_classification_scores(y_true, y_pred) -> ClassificationMetricArtifact:
    try:
        model_f1_score = f1_score(y_true, y_pred)
        model_accuracy_score = accuracy_score(y_true, y_pred)
        model_balanced_accuracy_score = balanced_accuracy_score(y_true, y_pred)

        classification_metric = ClassificationMetricArtifact(f1_score=model_f1_score,accuracy=model_accuracy_score,balanced_accuracy=model_balanced_accuracy_score)
        return classification_metric
    except Exception as e:
        raise MeOxException(e, sys)