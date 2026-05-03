# Dockerfile - Container configuration for the Marketing Conversion ML app
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Set the entrypoint to run the Streamlit app
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
