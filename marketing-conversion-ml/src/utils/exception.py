# exception.py - Custom exception handler for the ML pipeline
# Provides detailed error messages including file name and line number

import sys  # Import sys module to access exception details and traceback information


def error_message_detail(error, error_detail: sys):
    """
    Extracts detailed error information from the exception traceback.

    Args:
        error: The original exception object
        error_detail: sys module reference for extracting traceback info

    Returns:
        str: A formatted error message string with file name, line number, and error message
    """
    # Extract traceback information from the exception details
    _, _, exc_tb = error_detail.exc_info()

    # Get the filename where the exception occurred
    file_name = exc_tb.tb_frame.f_code.co_filename

    # Format the error message with file name, line number, and error description
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name,  # Name of the file where the error occurred
        exc_tb.tb_lineno,  # Line number where the error occurred
        str(error),  # The actual error message
    )

    # Return the formatted error message
    return error_message


class CustomException(Exception):
    """
    Custom exception class that provides detailed error information.
    Inherits from the built-in Exception class.
    """

    def __init__(self, error_message, error_detail: sys):
        """
        Initialize the CustomException with detailed error information.

        Args:
            error_message: The original error message or exception
            error_detail: sys module reference for extracting traceback info
        """
        # Call the parent Exception class constructor with the error message
        super().__init__(error_message)

        # Store the detailed error message with file and line information
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        """Return the detailed error message when the exception is printed."""
        return self.error_message
