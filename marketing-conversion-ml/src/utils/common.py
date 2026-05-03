# common.py - Utility helper functions used across the ML pipeline
# Contains functions for saving/loading objects, reading configs, etc.

import os  # Import os for file and directory operations
import sys  # Import sys for system-specific parameters and exception handling
import pickle  # Import pickle for serializing and deserializing Python objects
import yaml  # Import yaml for reading YAML configuration files
import pandas as pd  # Import pandas for data manipulation and analysis
import numpy as np  # Import numpy for numerical operations

from src.utils.logger import logger  # Import the custom logger for logging messages
from src.utils.exception import CustomException  # Import custom exception for error handling


def save_object(file_path: str, obj: object) -> None:
    """
    Save a Python object to a file using pickle serialization.

    Args:
        file_path (str): The full path where the object will be saved
        obj (object): The Python object to serialize and save
    """
    try:
        # Extract the directory path from the full file path
        dir_path = os.path.dirname(file_path)

        # Create the directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)

        # Open the file in write-binary mode and dump the object using pickle
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)  # Serialize and write the object to file

        # Log the successful save operation
        logger.info(f"Object saved successfully at: {file_path}")

    except Exception as e:
        # Raise a custom exception with detailed error information
        raise CustomException(e, sys)


def load_object(file_path: str) -> object:
    """
    Load a Python object from a pickle file.

    Args:
        file_path (str): The full path to the pickle file to load

    Returns:
        object: The deserialized Python object
    """
    try:
        # Open the file in read-binary mode and load the object using pickle
        with open(file_path, "rb") as file_obj:
            loaded_obj = pickle.load(file_obj)  # Deserialize and read the object from file

        # Log the successful load operation
        logger.info(f"Object loaded successfully from: {file_path}")

        # Return the loaded object
        return loaded_obj

    except Exception as e:
        # Raise a custom exception with detailed error information
        raise CustomException(e, sys)


def read_yaml(file_path: str) -> dict:
    """
    Read a YAML configuration file and return its contents as a dictionary.

    Args:
        file_path (str): The path to the YAML file

    Returns:
        dict: The contents of the YAML file as a Python dictionary
    """
    try:
        # Open the YAML file in read mode with UTF-8 encoding
        with open(file_path, "r", encoding="utf-8") as yaml_file:
            # Parse the YAML content and return it as a dictionary
            config = yaml.safe_load(yaml_file)

        # Log the successful read operation
        logger.info(f"YAML file loaded successfully from: {file_path}")

        # Return the parsed configuration dictionary
        return config

    except Exception as e:
        # Raise a custom exception with detailed error information
        raise CustomException(e, sys)


def get_project_root() -> str:
    """
    Get the root directory path of the project.

    Returns:
        str: The absolute path to the project root directory
    """
    # Navigate up from utils/ to src/ to project root
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
