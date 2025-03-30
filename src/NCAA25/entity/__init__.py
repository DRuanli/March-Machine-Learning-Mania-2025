from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataPreparationConfig:
    data_path: Path
    teams_file: Path
    seeds_file: Path
    regular_season_results_file: Path
    tourney_results_file: Path
    detailed_results_file: Path
    output_dir: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    trained_model_path: Path
    model_params: dict
    train_data_path: Path
    test_size: float
    random_state: int
    calibration_method: str
    feature_columns_file: Path

@dataclass(frozen=True)
class ModelAnalysisConfig:
    root_dir: Path
    model_path: Path
    submission_path: Path
    train_data_path: Path
    feature_columns_file: Path
    calibrator_path: Path
    analysis_reports_dir: Path