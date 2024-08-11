# Use the lightweight Python 3.10.14 base image on Debian Bullseye slim
FROM python:3.10.14-slim-bullseye

# Set the working directory
WORKDIR /app

# Copy the requirements file into the image
COPY requirements.txt .

# Install dependencies from the requirements file and then remove the file
RUN pip install --no-cache-dir -r requirements.txt \
    && rm requirements.txt

# Copy the API code
COPY . .

# Create a non-root user to run the application
RUN adduser --disabled-password --gecos '' appuser && chown -R appuser /app
USER appuser

# Expose the port on which the application will run
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
