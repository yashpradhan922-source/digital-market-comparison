# training_pipeline.py - End-to-end training pipeline for the ML project
# Orchestrates data ingestion, transformation, and model training in sequence

import sys  # Import sys for exception handling

from src.components.data_ingestion import DataIngestion  # Import data ingestion component
from src.components.data_transformation import DataTransformation  # Import data transformation component
from src.components.model_trainer import ModelTrainer  # Import model training component
from src.utils.logger import logger  # Import custom logger for logging pipeline events
from src.utils.exception import CustomException  # Import custom exception for error handling


class TrainingPipeline:
    """
    Training Pipeline orchestrates the entire ML training workflow:
    1. Data Ingestion: Read and split the raw data
    2. Data Transformation: Preprocess features (scale, encode)
    3. Model Training: Train models and select the best one
    """

    def __init__(self):
        """
        Initialize the TrainingPipeline class.
        No specific initialization needed as components are created during execution.
        """
        # Log that the training pipeline has been initialized
        logger.info("TrainingPipeline initialized")

    def run_pipeline(self) -> dict:
        """
        Execute the full training pipeline from data ingestion to model training.

        Returns:
            dict: A dictionary containing the best model name, metrics, and full report
        """
        try:
            # Log the start of the training pipeline
            logger.info("=" * 50)
            logger.info("TRAINING PIPELINE STARTED")
            logger.info("=" * 50)

            # ====== STEP 1: Data Ingestion ======
            # Create an instance of the DataIngestion component
            data_ingestion = DataIngestion()

            # Execute data ingestion to read CSV and split into train/test
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

            # Log the completion of Step 1
            logger.info("Step 1 - Data Ingestion completed")

            # ====== STEP 2: Data Transformation ======
            # Create an instance of the DataTransformation component
            data_transformation = DataTransformation()

            # Execute data transformation to preprocess features
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
                train_data_path, test_data_path  # Pass the paths from Step 1
            )

            # Log the completion of Step 2
            logger.info("Step 2 - Data Transformation completed")

            # ====== STEP 3: Model Training ======
            # Create an instance of the ModelTrainer component
            model_trainer = ModelTrainer()

            # Execute model training with the transformed data
            best_model_name, best_metrics, model_report = model_trainer.initiate_model_training(
                train_arr, test_arr  # Pass the transformed arrays from Step 2
            )

            # Log the completion of Step 3
            logger.info("Step 3 - Model Training completed")

            # Log the overall pipeline completion with results
            logger.info("=" * 50)
            logger.info(f"TRAINING PIPELINE COMPLETED - Best Model: {best_model_name}")
            logger.info(f"Best Model Metrics: {best_metrics}")
            logger.info("=" * 50)

            # Build and return the results dictionary
            results = {
                "best_model": best_model_name,  # Name of the best performing model
                "best_metrics": best_metrics,  # Metrics of the best model
                "model_report": model_report,  # Full comparison report of all models
            }

            # Return the results dictionary
            return results

        except Exception as e:
            # Log the error and raise a custom exception
            logger.error(f"Error in training pipeline: {str(e)}")
            raise CustomException(e, sys)
