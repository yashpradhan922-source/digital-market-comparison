# data_ingestion.py - Data Ingestion component for the ML pipeline
# Responsible for reading raw data, splitting into train/test sets, and saving them

import os  # Import os for file and directory path operations
import sys  # Import sys for exception handling and system info
import pandas as pd  # Import pandas for reading and manipulating CSV data
from sklearn.model_selection import train_test_split  # Import for splitting data into train/test

from src.utils.logger import logger  # Import custom logger for logging pipeline events
from src.utils.exception import CustomException  # Import custom exception for error handling
from src.utils.common import read_yaml, get_project_root  # Import helper functions


class DataIngestion:
    """
    Data Ingestion class handles reading raw CSV data, performing
    train-test split, and saving the resulting datasets to disk.
    """

    def __init__(self):
        """
        Initialize the DataIngestion class by loading configuration
        from the YAML config file.
        """
        # Get the project root directory path
        self.project_root = get_project_root()

        # Load the configuration from config.yaml
        config_path = os.path.join(self.project_root, "src", "config", "config.yaml")
        self.config = read_yaml(config_path)  # Parse the YAML config into a dictionary

        # Extract data ingestion specific configuration
        self.ingestion_config = self.config["data_ingestion"]

        # Log that DataIngestion has been initialized
        logger.info("DataIngestion component initialized successfully")

    def initiate_data_ingestion(self) -> tuple:
        """
        Execute the data ingestion process:
        1. Read the raw CSV data
        2. Create necessary directories
        3. Split data into training and testing sets
        4. Save the split datasets

        Returns:
            tuple: Paths to the training and testing CSV files
        """
        # Log the start of the data ingestion process
        logger.info("Data ingestion process started")

        try:
            # Construct the full path to the raw data CSV file
            raw_data_path = os.path.join(self.project_root, self.ingestion_config["raw_data_path"])

            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(raw_data_path)

            # Log the shape of the loaded dataset (rows, columns)
            logger.info(f"Dataset loaded successfully. Shape: {df.shape}")

            # Construct full paths for train and test data output files
            train_data_path = os.path.join(self.project_root, self.ingestion_config["train_data_path"])
            test_data_path = os.path.join(self.project_root, self.ingestion_config["test_data_path"])

            # Create the output directories if they don't exist
            os.makedirs(os.path.dirname(train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(test_data_path), exist_ok=True)

            # Save a copy of the raw data to the raw data directory
            raw_save_path = os.path.join(self.project_root, "data", "raw", "digital_marketing_campaign_dataset.csv")
            os.makedirs(os.path.dirname(raw_save_path), exist_ok=True)

            # Only copy if raw data is not already in the expected location
            if not os.path.exists(raw_save_path):
                df.to_csv(raw_save_path, index=False)  # Save raw data without row indices
                logger.info(f"Raw data saved to: {raw_save_path}")

            # Split the dataset into training and testing sets
            train_set, test_set = train_test_split(
                df,  # The full dataset to split
                test_size=self.ingestion_config["test_size"],  # Proportion for test set (0.2)
                random_state=self.ingestion_config["random_state"],  # Seed for reproducibility
            )

            # Save the training set to CSV file
            train_set.to_csv(train_data_path, index=False)  # Save without row indices
            logger.info(f"Training data saved to: {train_data_path}. Shape: {train_set.shape}")

            # Save the testing set to CSV file
            test_set.to_csv(test_data_path, index=False)  # Save without row indices
            logger.info(f"Testing data saved to: {test_data_path}. Shape: {test_set.shape}")

            # Log the successful completion of data ingestion
            logger.info("Data ingestion completed successfully")

            # Return the paths to the saved train and test files
            return train_data_path, test_data_path

        except Exception as e:
            # Log the error and raise a custom exception with details
            logger.error(f"Error during data ingestion: {str(e)}")
            raise CustomException(e, sys)
