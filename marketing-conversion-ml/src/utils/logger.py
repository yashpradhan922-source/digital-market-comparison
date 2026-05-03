# logger.py - Custom logging module for the ML pipeline
# Provides a configured logger instance for consistent logging across the project

import logging  # Import the built-in logging module for log management
import os  # Import os module for file path and directory operations
from datetime import datetime  # Import datetime to generate timestamped log filenames

# Create a unique log filename using the current date and time
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the directory path where log files will be stored
logs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")

# Create the logs directory if it does not already exist
os.makedirs(logs_path, exist_ok=True)

# Construct the full file path for the log file
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure the basic logging settings for the entire application
logging.basicConfig(
    filename=LOG_FILE_PATH,  # Set the log file location
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Define log message format
    level=logging.INFO,  # Set the minimum logging level to INFO
)

# Create a logger instance named 'MarketingConversionML' for use throughout the project
logger = logging.getLogger("MarketingConversionML")
