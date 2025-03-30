from src.NCAA25.constants import *
from src.NCAA25.utils.common import read_yaml, create_directories
from src.NCAA25.entity import DataIngestionConfig, DataPreparationConfig, ModelTrainerConfig, ModelAnalysisConfig
from pathlib import Path


class ConfigurationManager:
    def __init__(
            self,
            config_filepath=CONFIG_FILE_PATH,
            params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config

    def get_data_preparation_config(self) -> DataPreparationConfig:
        """
        Get the configuration for data preparation

        Returns:
            DataPreparationConfig: Configuration for data preparation
        """
        config = self.config.data_ingestion

        data_preparation_config = DataPreparationConfig(
            data_path=config.unzip_dir,
            teams_file=Path(config.unzip_dir) / "MTeams.csv",
            seeds_file=Path(config.unzip_dir) / "MNCAATourneySeeds.csv",
            regular_season_results_file=Path(config.unzip_dir) / "MRegularSeasonCompactResults.csv",
            tourney_results_file=Path(config.unzip_dir) / "MNCAATourneyCompactResults.csv",
            detailed_results_file=Path(config.unzip_dir) / "MRegularSeasonDetailedResults.csv",
            output_dir=Path(config.root_dir) / "processed"
        )

        create_directories([data_preparation_config.output_dir])

        return data_preparation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """
        Get the configuration for model training
        """
        config = self.config.model_trainer
        create_directories([config.root_dir])

        return ModelTrainerConfig(
            root_dir=config.root_dir,
            trained_model_path=config.trained_model_path,
            model_params=self.params.random_forest,
            train_data_path=config.train_data_path,
            test_size=config.test_size,
            random_state=self.params.random_forest.random_state,
            calibration_method=self.params.calibration.method,
            feature_columns_file=config.feature_columns_file
        )

    def get_model_analysis_config(self) -> ModelAnalysisConfig:
        """Get configuration for model analysis"""
        config = self.config.model_analysis
        create_directories([config.root_dir, config.analysis_reports_dir])

        return ModelAnalysisConfig(
            root_dir=config.root_dir,
            model_path=config.model_path,
            submission_path=config.submission_path,
            train_data_path=config.train_data_path,
            feature_columns_file=config.feature_columns_file,
            calibrator_path=config.calibrator_path,
            analysis_reports_dir=config.analysis_reports_dir
        )