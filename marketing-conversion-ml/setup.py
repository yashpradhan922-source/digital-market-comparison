# setup.py - Package setup configuration for Marketing Conversion ML
# Enables the project to be installed as a Python package

from setuptools import setup, find_packages  # Import setup tools for package configuration

# Read the requirements from requirements.txt
with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Configure the package setup
setup(
    name="marketing-conversion-ml",  # Package name
    version="1.0.0",  # Package version
    author="Your Name",  # Author name
    author_email="your.email@example.com",  # Author email
    description="ML pipeline for predicting digital marketing campaign conversions",  # Short description
    packages=find_packages(),  # Automatically discover all packages in the project
    install_requires=requirements,  # Install dependencies from requirements.txt
    python_requires=">=3.9",  # Minimum Python version required
)
