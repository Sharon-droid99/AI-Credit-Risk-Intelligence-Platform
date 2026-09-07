FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if required by any python package
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Extract the features dataset and clean up the zip file
RUN unzip -j models/application_features.zip -d models/ && \
    rm models/application_features.zip

# Initialize DuckDB database
RUN python -m src.talk_to_data.query_runner

# Expose Streamlit port
EXPOSE 8501

# Run the Streamlit application
CMD ["python", "-m", "streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
