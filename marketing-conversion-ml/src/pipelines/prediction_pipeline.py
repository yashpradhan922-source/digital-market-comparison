# prediction_pipeline.py - Prediction pipeline for the ML project
# Handles loading the trained model and preprocessor to make predictions on new data

import os  # Import os for file path operations
import sys  # Import sys for exception handling
import numpy as np  # Import numpy for numerical operations
import pandas as pd  # Import pandas for creating DataFrames from input

from src.utils.logger import logger  # Import custom logger
from src.utils.exception import CustomException  # Import custom exception
from src.utils.common import load_object, read_yaml, get_project_root  # Import helpers


class PredictionPipeline:
    """
    Prediction Pipeline handles making predictions on new data:
    - Loads the trained model from disk
    - Loads the fitted preprocessor from disk
    - Transforms new input data using the preprocessor
    - Returns predictions using the trained model
    """

    def __init__(self):
        """
        Initialize the PredictionPipeline by loading the model and preprocessor.
        """
        try:
            # Get the project root directory path
            self.project_root = get_project_root()

            # Load the main configuration from config.yaml
            config_path = os.path.join(self.project_root, "src", "config", "config.yaml")
            self.config = read_yaml(config_path)

            # Construct the full path to the trained model file
            model_path = os.path.join(
                self.project_root, self.config["model_trainer"]["trained_model_file_path"]
            )

            # Construct the full path to the fitted preprocessor file
            preprocessor_path = os.path.join(
                self.project_root, self.config["data_transformation"]["preprocessor_obj_file_path"]
            )

            # Load the trained model object from the pickle file
            self.model = load_object(model_path)

            # Load the fitted preprocessor object from the pickle file
            self.preprocessor = load_object(preprocessor_path)

            # Log successful initialization
            logger.info("PredictionPipeline initialized - model and preprocessor loaded")

        except Exception as e:
            # Log the error and raise a custom exception
            logger.error(f"Error initializing PredictionPipeline: {str(e)}")
            raise CustomException(e, sys)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new input data.

        Args:
            features (pd.DataFrame): A DataFrame containing input features
                                     (must match the training data columns)

        Returns:
            np.ndarray: Array of predicted class labels (0 or 1)
        """
        try:
            # Log the prediction request with the input shape
            logger.info(f"Making prediction for input shape: {features.shape}")

            # Transform the input features using the fitted preprocessor
            data_scaled = self.preprocessor.transform(features)

            # Make predictions using the trained model
            predictions = self.model.predict(data_scaled)

            # Log the prediction results
            logger.info(f"Predictions completed: {predictions}")

            # Return the predicted class labels
            return predictions

        except Exception as e:
            # Log the error and raise a custom exception
            logger.error(f"Error during prediction: {str(e)}")
            raise CustomException(e, sys)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities for new input data.

        Args:
            features (pd.DataFrame): A DataFrame containing input features

        Returns:
            np.ndarray: Array of predicted probabilities for each class
        """
        try:
            # Log the probability prediction request
            logger.info(f"Making probability prediction for input shape: {features.shape}")

            # Transform the input features using the fitted preprocessor
            data_scaled = self.preprocessor.transform(features)

            # Get prediction probabilities from the trained model
            pred_proba = self.model.predict_proba(data_scaled)

            # Log the completion of probability prediction
            logger.info("Probability predictions completed")

            # Return the probability predictions
            return pred_proba

        except Exception as e:
            # Log the error and raise a custom exception
            logger.error(f"Error during probability prediction: {str(e)}")
            raise CustomException(e, sys)


class CustomData:
    """
    Custom Data class for creating a DataFrame from individual feature values.
    Used to convert Streamlit form inputs into the format expected by the pipeline.
    """

    def __init__(
        self,
        Age: int,  # Customer's age (18-69)
        Gender: str,  # Customer's gender (Male/Female)
        Income: float,  # Customer's annual income
        CampaignChannel: str,  # Marketing campaign channel (Email, PPC, SEO, etc.)
        CampaignType: str,  # Type of campaign (Awareness, Consideration, etc.)
        AdSpend: float,  # Amount spent on advertising
        ClickThroughRate: float,  # Click-through rate (0-0.3)
        ConversionRate: float,  # Conversion rate (0-0.2)
        WebsiteVisits: int,  # Number of website visits
        PagesPerVisit: float,  # Average pages per visit
        TimeOnSite: float,  # Average time spent on site (minutes)
        SocialShares: int,  # Number of social media shares
        EmailOpens: int,  # Number of email opens
        EmailClicks: int,  # Number of email clicks
        PreviousPurchases: int,  # Number of previous purchases
        LoyaltyPoints: int,  # Customer loyalty points
    ):
        """
        Initialize CustomData with individual feature values from user input.
        """
        # Store each feature value as an instance attribute
        self.Age = Age
        self.Gender = Gender
        self.Income = Income
        self.CampaignChannel = CampaignChannel
        self.CampaignType = CampaignType
        self.AdSpend = AdSpend
        self.ClickThroughRate = ClickThroughRate
        self.ConversionRate = ConversionRate
        self.WebsiteVisits = WebsiteVisits
        self.PagesPerVisit = PagesPerVisit
        self.TimeOnSite = TimeOnSite
        self.SocialShares = SocialShares
        self.EmailOpens = EmailOpens
        self.EmailClicks = EmailClicks
        self.PreviousPurchases = PreviousPurchases
        self.LoyaltyPoints = LoyaltyPoints

    def get_data_as_dataframe(self) -> pd.DataFrame:
        """
        Convert the stored feature values into a pandas DataFrame.

        Returns:
            pd.DataFrame: A single-row DataFrame with all features as columns
        """
        try:
            # Create a dictionary mapping column names to their values (as lists for DataFrame)
            custom_data_input_dict = {
                "Age": [self.Age],  # Wrap each value in a list for DataFrame construction
                "Gender": [self.Gender],
                "Income": [self.Income],
                "CampaignChannel": [self.CampaignChannel],
                "CampaignType": [self.CampaignType],
                "AdSpend": [self.AdSpend],
                "ClickThroughRate": [self.ClickThroughRate],
                "ConversionRate": [self.ConversionRate],
                "WebsiteVisits": [self.WebsiteVisits],
                "PagesPerVisit": [self.PagesPerVisit],
                "TimeOnSite": [self.TimeOnSite],
                "SocialShares": [self.SocialShares],
                "EmailOpens": [self.EmailOpens],
                "EmailClicks": [self.EmailClicks],
                "PreviousPurchases": [self.PreviousPurchases],
                "LoyaltyPoints": [self.LoyaltyPoints],
            }

            # Create and return a DataFrame from the dictionary
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            # Raise a custom exception if DataFrame creation fails
            raise CustomException(e, sys)
