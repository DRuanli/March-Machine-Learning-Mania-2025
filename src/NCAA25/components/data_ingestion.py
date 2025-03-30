import os
import zipfile
import gdown
from src.NCAA25 import logger
from src.NCAA25.entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> None:
        """
        Download the data from Google Drive URL
        """
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs(os.path.dirname(self.config.local_data_file), exist_ok=True)

            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            # Extract file ID from Google Drive URL
            file_id = dataset_url.split("/")[-2]

            # Use gdown to download from Google Drive
            gdown.download(
                id=file_id,
                output=zip_download_dir,
                quiet=False
            )

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")
        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            raise e

    def extract_zip_file(self) -> None:
        """
        Extract the downloaded zip file
        """
        try:
            os.makedirs(self.config.unzip_dir, exist_ok=True)

            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(self.config.unzip_dir)

            logger.info(f"Extracted data to {self.config.unzip_dir}")
        except Exception as e:
            logger.error(f"Error extracting zip file: {e}")
            raise e