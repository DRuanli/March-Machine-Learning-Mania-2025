import os
import yaml
import json
import joblib
import pandas as pd
from box import ConfigBox
from pathlib import Path
from typing import Any, Dict, List
import logging

from src.NCAA25 import logger


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Read a YAML file and return a ConfigBox object

    Args:
        path_to_yaml (Path): Path to the YAML file

    Returns:
        ConfigBox: ConfigBox object containing the YAML data

    Raises:
        ValueError: If the YAML file is empty
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except Exception as e:
        logger.error(f"Error reading YAML file {path_to_yaml}: {e}")
        raise e


def create_directories(path_to_directories: List[Path], verbose=True):
    """
    Create directories if they don't exist

    Args:
        path_to_directories (List[Path]): List of paths to directories
        verbose (bool, optional): Whether to log the creation. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Directory {path} created successfully")


def save_dataframe(dataframe: pd.DataFrame, path: Path):
    """
    Save a pandas DataFrame to a CSV file

    Args:
        dataframe (pd.DataFrame): DataFrame to save
        path (Path): Path to save the DataFrame to
    """
    dataframe.to_csv(path, index=False)
    logger.info(f"DataFrame saved to {path}")


def load_dataframe(path: Path) -> pd.DataFrame:
    """
    Load a pandas DataFrame from a CSV file

    Args:
        path (Path): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    return pd.read_csv(path)


def save_model(model: Any, path: Path):
    """
    Save a model to a file

    Args:
        model (Any): Model to save
        path (Path): Path to save the model to
    """
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


def load_model(path: Path) -> Any:
    """
    Load a model from a file

    Args:
        path (Path): Path to the model file

    Returns:
        Any: Loaded model
    """
    return joblib.load(path)