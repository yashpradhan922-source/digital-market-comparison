# test_pipeline.py - Unit tests for the ML pipeline components
import os
import sys
import pytest
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.common import get_project_root, read_yaml
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.pipelines.prediction_pipeline import CustomData


class TestDataIngestion:
    """Tests for the DataIngestion component."""

    def test_config_loading(self):
        """Test that configuration loads correctly."""
        ingestion = DataIngestion()
        assert ingestion.config is not None  # Config should not be None
        assert "data_ingestion" in ingestion.config  # Should contain ingestion section

    def test_data_ingestion_returns_paths(self):
        """Test that data ingestion returns valid file paths."""
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        assert os.path.exists(train_path)  # Train file should exist
        assert os.path.exists(test_path)  # Test file should exist


class TestDataTransformation:
    """Tests for the DataTransformation component."""

    def test_transformer_creation(self):
        """Test that the data transformer object is created."""
        transformation = DataTransformation()
        transformer = transformation.get_data_transformer_object()
        assert transformer is not None  # Transformer should not be None


class TestCustomData:
    """Tests for the CustomData class."""

    def test_custom_data_to_dataframe(self):
        """Test that CustomData creates a valid DataFrame."""
        data = CustomData(
            Age=30, Gender="Male", Income=50000,
            CampaignChannel="Email", CampaignType="Awareness",
            AdSpend=1000, ClickThroughRate=0.1, ConversionRate=0.05,
            WebsiteVisits=10, PagesPerVisit=3.0, TimeOnSite=5.0,
            SocialShares=20, EmailOpens=5, EmailClicks=2,
            PreviousPurchases=3, LoyaltyPoints=1500,
        )
        df = data.get_data_as_dataframe()
        assert isinstance(df, pd.DataFrame)  # Should be a DataFrame
        assert df.shape == (1, 16)  # Should have 1 row and 16 columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
