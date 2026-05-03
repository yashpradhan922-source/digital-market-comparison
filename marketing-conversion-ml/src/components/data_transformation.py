# data_transformation.py - Data Transformation component for the ML pipeline
# Handles feature engineering, encoding categorical variables, and scaling numerical features

import os  # Import os for file and directory path operations
import sys  # Import sys for exception handling
import numpy as np  # Import numpy for numerical array operations
import pandas as pd  # Import pandas for data manipulation
from imblearn.over_sampling import SMOTE  # Import SMOTE for oversampling the minority class
from sklearn.compose import ColumnTransformer  # Import for applying different transformers to different columns
from sklearn.pipeline import Pipeline  # Import for creating sequential transformation pipelines
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # Import scaler and encoder
from sklearn.impute import SimpleImputer  # Import for handling missing values

from src.utils.logger import logger  # Import custom logger
from src.utils.exception import CustomException  # Import custom exception
from src.utils.common import save_object, read_yaml, get_project_root  # Import helper functions


class DataTransformation:
    """
    Data Transformation class handles preprocessing of features:
    - Imputing missing values
    - Scaling numerical features using StandardScaler
    - Encoding categorical features using OneHotEncoder
    - Saving the fitted preprocessing pipeline for later use
    """

    def __init__(self):
        """
        Initialize the DataTransformation class by loading configuration
        from the YAML config file.
        """
        # Get the project root directory path
        self.project_root = get_project_root()

        # Load the configuration from config.yaml
        config_path = os.path.join(self.project_root, "src", "config", "config.yaml")
        self.config = read_yaml(config_path)  # Parse the YAML config

        # Extract transformation and feature configuration sections
        self.transformation_config = self.config["data_transformation"]
        self.feature_config = self.config["features"]

        # Log successful initialization
        logger.info("DataTransformation component initialized successfully")

    def get_data_transformer_object(self):
        """
        Create and return a ColumnTransformer preprocessing pipeline that:
        - Imputes missing values and scales numerical columns
        - Imputes missing values and one-hot encodes categorical columns

        Returns:
            ColumnTransformer: The configured preprocessing pipeline object
        """
        try:
            # Get the lists of numerical and categorical column names from config
            numerical_columns = self.feature_config["numerical_columns"]
            categorical_columns = self.feature_config["categorical_columns"]

            # Log the columns being processed
            logger.info(f"Numerical columns: {numerical_columns}")
            logger.info(f"Categorical columns: {categorical_columns}")

            # Create a pipeline for numerical features:
            # Step 1: Impute missing values with the median value
            # Step 2: Scale features to have zero mean and unit variance
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),  # Fill NaN with column median
                    ("scaler", StandardScaler()),  # Standardize features (z-score normalization)
                ]
            )

            # Create a pipeline for categorical features:
            # Step 1: Impute missing values with the most frequent category
            # Step 2: One-hot encode categorical variables into binary columns
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),  # Fill NaN with mode
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),  # One-hot encode
                ]
            )

            # Combine both pipelines using ColumnTransformer
            # This applies the numerical pipeline to numerical columns
            # and the categorical pipeline to categorical columns
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),  # Apply num_pipeline to numerical cols
                    ("cat_pipeline", cat_pipeline, categorical_columns),  # Apply cat_pipeline to categorical cols
                ],
                remainder="drop",  # Drop any columns not specified in the transformers
            )

            # Log the successful creation of the preprocessor
            logger.info("Data transformer object created successfully")

            # Return the configured preprocessor pipeline
            return preprocessor

        except Exception as e:
            # Log the error and raise a custom exception
            logger.error(f"Error creating data transformer: {str(e)}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str) -> tuple:
        """
        Execute the data transformation process:
        1. Read training and testing data
        2. Drop unnecessary columns
        3. Separate features and target variable
        4. Fit the preprocessor on training data and transform both sets
        5. Save the fitted preprocessor

        Args:
            train_path (str): Path to the training CSV file
            test_path (str): Path to the testing CSV file

        Returns:
            tuple: (transformed_train_array, transformed_test_array, preprocessor_path)
        """
        try:
            # Log the start of data transformation
            logger.info("Data transformation process started")

            # Read the training and testing CSV files into DataFrames
            train_df = pd.read_csv(train_path)  # Load training data
            test_df = pd.read_csv(test_path)  # Load testing data

            # Log the shapes of loaded datasets
            logger.info(f"Train DataFrame shape: {train_df.shape}")
            logger.info(f"Test DataFrame shape: {test_df.shape}")

            # Get the preprocessing pipeline object
            preprocessing_obj = self.get_data_transformer_object()

            # Get the target column name and columns to drop from config
            target_column_name = self.feature_config["target_column"]
            drop_columns = self.feature_config["drop_columns"]

            # Create the list of all columns to remove (drop columns + target)
            columns_to_drop = drop_columns + [target_column_name]

            # Separate features (X) and target (y) for training data
            input_feature_train_df = train_df.drop(columns=columns_to_drop)  # Drop target and ID columns
            target_feature_train_df = train_df[target_column_name]  # Extract target column

            # Separate features (X) and target (y) for testing data
            input_feature_test_df = test_df.drop(columns=columns_to_drop)  # Drop target and ID columns
            target_feature_test_df = test_df[target_column_name]  # Extract target column

            # Log the feature and target shapes
            logger.info(f"Training features shape: {input_feature_train_df.shape}")
            logger.info(f"Testing features shape: {input_feature_test_df.shape}")

            # Fit the preprocessor on training data and transform training features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)

            # Transform testing features using the already fitted preprocessor
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Log the shapes of transformed arrays
            logger.info(f"Transformed training array shape: {input_feature_train_arr.shape}")
            logger.info(f"Transformed testing array shape: {input_feature_test_arr.shape}")

            # ====== SMOTE: Handle Class Imbalance ======
            # Apply SMOTE (Synthetic Minority Over-sampling Technique) to the training data
            # This generates synthetic samples for the minority class to balance the dataset
            smote = SMOTE(random_state=42)  # Initialize SMOTE with a fixed random seed

            # Log class distribution before SMOTE
            unique, counts = np.unique(target_feature_train_df, return_counts=True)
            logger.info(f"Class distribution BEFORE SMOTE: {dict(zip(unique, counts))}")

            # Apply SMOTE to resample the training features and target
            input_feature_train_arr, target_feature_train_resampled = smote.fit_resample(
                input_feature_train_arr,  # Transformed training features
                np.array(target_feature_train_df),  # Training target labels
            )

            # Log class distribution after SMOTE
            unique, counts = np.unique(target_feature_train_resampled, return_counts=True)
            logger.info(f"Class distribution AFTER SMOTE: {dict(zip(unique, counts))}")
            logger.info(f"Training data shape after SMOTE: {input_feature_train_arr.shape}")

            # Combine transformed features with target variable into single arrays
            # np.c_ concatenates arrays column-wise
            train_arr = np.c_[input_feature_train_arr, target_feature_train_resampled]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Construct the full path for saving the preprocessor object
            preprocessor_path = os.path.join(
                self.project_root, self.transformation_config["preprocessor_obj_file_path"]
            )

            # Save the fitted preprocessing pipeline to disk for reuse during prediction
            save_object(file_path=preprocessor_path, obj=preprocessing_obj)

            # Log the successful completion of transformation
            logger.info("Data transformation completed successfully")

            # Return the transformed arrays and the path to the saved preprocessor
            return train_arr, test_arr, preprocessor_path

        except Exception as e:
            # Log the error and raise a custom exception
            logger.error(f"Error during data transformation: {str(e)}")
            raise CustomException(e, sys)
