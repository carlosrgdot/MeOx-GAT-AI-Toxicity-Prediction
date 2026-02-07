from dataclasses import dataclass



@dataclass
class DataIngestionArtifact:
    data_file_path: str


@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_data_file_path: str
    drift_report_file_path: str


@dataclass
class DataTransformationArtifact:
    transformed_data_file_path: str
    transformed_object_file_path: str


@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    accuracy: float
    balanced_accuracy: float

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact

