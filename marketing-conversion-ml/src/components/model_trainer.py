# model_trainer.py - Model Training component for the ML pipeline
# Trains multiple ML models, evaluates them, and saves the best performing model

import os  # Import os for file and directory path operations
import sys  # Import sys for exception handling
import json  # Import json for saving model evaluation reports
import numpy as np  # Import numpy for numerical operations

from sklearn.ensemble import (
    RandomForestClassifier,  # Import Random Forest classifier
    GradientBoostingClassifier,  # Import Gradient Boosting classifier
)
from sklearn.linear_model import LogisticRegression  # Import Logistic Regression classifier
from xgboost import XGBClassifier  # Import XGBoost classifier
from sklearn.metrics import (
    accuracy_score,  # Import accuracy metric
    precision_score,  # Import precision metric
    recall_score,  # Import recall metric
    f1_score,  # Import F1-score metric
    roc_auc_score,  # Import ROC-AUC metric
    classification_report,  # Import detailed classification report
    confusion_matrix,  # Import confusion matrix
)

from src.utils.logger import logger  # Import custom logger
from src.utils.exception import CustomException  # Import custom exception
from src.utils.common import save_object, read_yaml, get_project_root  # Import helpers


class ModelTrainer:
    """
    Model Trainer class handles:
    - Training multiple classification models with configured hyperparameters
    - Evaluating models using various metrics
    - Selecting and saving the best performing model
    - Generating a model comparison report
    """

    def __init__(self):
        """
        Initialize the ModelTrainer class by loading configuration files.
        """
        # Get the project root directory path
        self.project_root = get_project_root()

        # Load the main config file for paths and settings
        config_path = os.path.join(self.project_root, "src", "config", "config.yaml")
        self.config = read_yaml(config_path)

        # Load the params file for model hyperparameters
        params_path = os.path.join(self.project_root, "src", "config", "params.yaml")
        self.params = read_yaml(params_path)

        # Extract model trainer configuration (output paths)
        self.model_trainer_config = self.config["model_trainer"]

        # Log successful initialization
        logger.info("ModelTrainer component initialized successfully")

    def evaluate_model(self, y_true, y_pred, y_pred_proba=None) -> dict:
        """
        Evaluate a model using multiple classification metrics.

        Args:
            y_true: Array of true target values
            y_pred: Array of predicted target values
            y_pred_proba: Array of predicted probabilities (optional, for AUC)

        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        # Calculate accuracy: ratio of correct predictions to total predictions
        accuracy = accuracy_score(y_true, y_pred)

        # Calculate precision: ratio of true positives to predicted positives
        precision = precision_score(y_true, y_pred, average="binary")

        # Calculate recall: ratio of true positives to actual positives
        recall = recall_score(y_true, y_pred, average="binary")

        # Calculate F1-score: harmonic mean of precision and recall
        f1 = f1_score(y_true, y_pred, average="binary")

        # Calculate ROC-AUC score if probability predictions are available
        if y_pred_proba is not None:
            roc_auc = roc_auc_score(y_true, y_pred_proba)  # Area under ROC curve
        else:
            roc_auc = None  # Set to None if probabilities not available

        # Compute the confusion matrix (TP, FP, FN, TN breakdown)
        cm = confusion_matrix(y_true, y_pred)

        # Build and return the metrics dictionary
        metrics = {
            "accuracy": round(accuracy, 4),  # Round to 4 decimal places
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4) if roc_auc else None,
            "confusion_matrix": cm.tolist(),  # Convert numpy array to list for JSON serialization
        }

        # Return the metrics dictionary
        return metrics

    def initiate_model_training(self, train_array: np.ndarray, test_array: np.ndarray) -> tuple:
        """
        Train multiple models, evaluate them, select the best one, and save it.

        Args:
            train_array (np.ndarray): Transformed training data (features + target in last column)
            test_array (np.ndarray): Transformed testing data (features + target in last column)

        Returns:
            tuple: (best_model_name, best_model_metrics, model_report)
        """
        try:
            # Log the start of model training
            logger.info("Model training process started")

            # Split the arrays into features (X) and target (y)
            # All columns except the last are features
            X_train = train_array[:, :-1]  # Training features
            y_train = train_array[:, -1]  # Training target (last column)
            X_test = test_array[:, :-1]  # Testing features
            y_test = test_array[:, -1]  # Testing target (last column)

            # Log the shapes of the splits
            logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

            # Define the dictionary of models to train with their hyperparameters
            models = {
                "Random Forest": RandomForestClassifier(
                    n_estimators=self.params["random_forest"]["n_estimators"],  # Number of trees
                    max_depth=self.params["random_forest"]["max_depth"],  # Max tree depth
                    min_samples_split=self.params["random_forest"]["min_samples_split"],  # Min samples to split
                    min_samples_leaf=self.params["random_forest"]["min_samples_leaf"],  # Min samples in leaf
                    random_state=self.params["random_forest"]["random_state"],  # Random seed
                    n_jobs=self.params["random_forest"]["n_jobs"],  # Parallel processing
                ),
                "Gradient Boosting": GradientBoostingClassifier(
                    n_estimators=self.params["gradient_boosting"]["n_estimators"],  # Boosting stages
                    learning_rate=self.params["gradient_boosting"]["learning_rate"],  # Step size
                    max_depth=self.params["gradient_boosting"]["max_depth"],  # Max tree depth
                    min_samples_split=self.params["gradient_boosting"]["min_samples_split"],  # Min samples to split
                    min_samples_leaf=self.params["gradient_boosting"]["min_samples_leaf"],  # Min samples in leaf
                    random_state=self.params["gradient_boosting"]["random_state"],  # Random seed
                ),
                "Logistic Regression": LogisticRegression(
                    C=self.params["logistic_regression"]["C"],  # Regularization strength
                    max_iter=self.params["logistic_regression"]["max_iter"],  # Max solver iterations
                    random_state=self.params["logistic_regression"]["random_state"],  # Random seed
                ),
                "XGBoost": XGBClassifier(
                    n_estimators=self.params["xgboost"]["n_estimators"],  # Number of trees
                    learning_rate=self.params["xgboost"]["learning_rate"],  # Step size
                    max_depth=self.params["xgboost"]["max_depth"],  # Max tree depth
                    min_child_weight=self.params["xgboost"]["min_child_weight"],  # Min child weight
                    subsample=self.params["xgboost"]["subsample"],  # Training instance subsample
                    colsample_bytree=self.params["xgboost"]["colsample_bytree"],  # Column subsample
                    random_state=self.params["xgboost"]["random_state"],  # Random seed
                    use_label_encoder=self.params["xgboost"]["use_label_encoder"],  # Disable label encoder
                    eval_metric=self.params["xgboost"]["eval_metric"],  # Loss function metric
                ),
            }

            # Initialize a dictionary to store the evaluation report for all models
            model_report = {}

            # Initialize variables to track the best model
            best_model_score = 0  # Track the highest accuracy score
            best_model_name = ""  # Track the name of the best model
            best_model = None  # Track the best model object

            # Iterate through each model, train it, and evaluate its performance
            for model_name, model in models.items():
                # Log the current model being trained
                logger.info(f"Training model: {model_name}")

                # Fit the model on the training data
                model.fit(X_train, y_train)

                # Generate predictions on the test set
                y_pred = model.predict(X_test)  # Predicted class labels

                # Generate probability predictions for ROC-AUC calculation
                y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class

                # Evaluate the model using multiple metrics
                metrics = self.evaluate_model(y_test, y_pred, y_pred_proba)

                # Store the metrics in the model report dictionary
                model_report[model_name] = metrics

                # Log the model's accuracy
                logger.info(f"{model_name} - Accuracy: {metrics['accuracy']}, F1: {metrics['f1_score']}")

                # Check if this model has the best accuracy so far
                if metrics["accuracy"] > best_model_score:
                    best_model_score = metrics["accuracy"]  # Update best score
                    best_model_name = model_name  # Update best model name
                    best_model = model  # Update best model object

            # Log the best model selection
            logger.info(f"Best model: {best_model_name} with accuracy: {best_model_score}")

            # Construct the full path for saving the trained model
            model_path = os.path.join(
                self.project_root, self.model_trainer_config["trained_model_file_path"]
            )

            # Save the best model to disk using pickle
            save_object(file_path=model_path, obj=best_model)
            logger.info(f"Best model saved to: {model_path}")

            # Save the model comparison report as a JSON file
            report_path = os.path.join(
                self.project_root, self.model_trainer_config["model_report_path"]
            )
            os.makedirs(os.path.dirname(report_path), exist_ok=True)  # Create directory if needed

            # Write the report to a JSON file with indentation for readability
            with open(report_path, "w") as f:
                json.dump(model_report, f, indent=4)  # Pretty-print the JSON
            logger.info(f"Model report saved to: {report_path}")

            # Log the completion of model training
            logger.info("Model training completed successfully")

            # Return the best model name, its metrics, and the full report
            return best_model_name, model_report[best_model_name], model_report

        except Exception as e:
            # Log the error and raise a custom exception
            logger.error(f"Error during model training: {str(e)}")
            raise CustomException(e, sys)
