from src.NCAA25 import logger
from src.NCAA25.config.configuration import ConfigurationManager
from src.NCAA25.components.data_ingestion import DataIngestion
from src.NCAA25.components.data_preparation import DataPreparation


def main():
    try:
        logger.info("Stage 1: Data Ingestion and Preparation Started")

        # Configuration
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()

        # Data Ingestion
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()

        # Data Preparation
        data_preparation_config = config.get_data_preparation_config()
        data_preparation = DataPreparation(config=data_preparation_config)
        data_preparation.run()

        logger.info("Stage 1: Data Ingestion and Preparation Completed")
    except Exception as e:
        logger.error(f"Error in Stage 1: {e}")
        raise e