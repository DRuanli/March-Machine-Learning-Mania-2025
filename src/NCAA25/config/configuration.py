from src.NCAA25.constants import *
from src.NCAA25.utils.common import read_yaml, create_directories
from src.NCAA25.entity import DataIngestionConfig, DataPreparationConfig
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