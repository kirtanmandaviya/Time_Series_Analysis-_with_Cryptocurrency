from dataclasses import dataclass

@dataclass
class DataIngestionArtifacts:
    feature_store_file_path: str # Path to the raw live data

@dataclass
class DataValidationArtifacts:
    validation_status: bool
    valid_data_file_path: str # Path to validated live data
    drift_report_file_path: str

@dataclass
class DataTransformationArtifacts:
    transformed_data_file_path: str # Path to the scaled CSV
    transformed_object_file_path: str # Path to the loaded scaler

@dataclass
class PredictionArtifact:
    prediction_file_path: str # CSV containing Date + Predicted Price
    prediction_status: bool
    message: str