# Use the lightweight Python 3.10.14 base image on Debian Bullseye slim
FROM python:3.10.14-slim-bullseye

# Metadata for the container to identify the maintainer and version
LABEL maintainer="jorge.abrego@gmail.com"
LABEL version="1.0"
LABEL description="Dockerfile to configure an MLflow server, using a database as backend store and a storage for artifacts"

# Update package list, install and upgrade procps, then clean up to reduce image size
RUN apt-get update && apt-get -y upgrade \
    && apt-get install -y procps \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version 
RUN pip install --no-cache-dir --upgrade pip \
    && pip --version    

# Create a group and user named 'mlflow' to avoid running as root
RUN groupadd mlflow && useradd --create-home -g mlflow mlflow

# Add the local binary directory of the 'mlflow' user to the PATH
ENV PATH /home/mlflow/.local/bin:${PATH}

# Set the working directory to /home/mlflow
WORKDIR /home/mlflow

# Copy the requirements file into the image
COPY mlflow_requeriments.txt mlflow_requeriments.txt

# Install dependencies from the requirements file and then remove the file
RUN pip install --no-cache-dir -r mlflow_requeriments.txt \
    && rm mlflow_requeriments.txt

# Switch to the 'mlflow' user
USER mlflow

# Expose port 5000 for the MLflow server
EXPOSE 5000

# Command to run the MLflow server with specified environment variables
CMD mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri ${BACKEND_STORE_URI} \
    --artifacts-destination s3://${MLFLOW_BUCKET_NAME} \
    --serve-artifacts