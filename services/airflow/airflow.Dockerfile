FROM apache/airflow:slim-2.9.2-python3.10

# Copy the requirements file into the Docker container
COPY ["airflow_requirements.txt", "./"] 

# Install dependencies using pip (set global.trusted-host to solve ssl conflicts)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-3.10.txt" -r airflow_requirements.txt 

ENV SHELL /bin/bash