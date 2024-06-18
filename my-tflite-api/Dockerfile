# Phyton image to use.
FROM python:3.10-slim

# Environment variable to ensure the output is flushed directly
ENV PYTHONUNBUFFERED True

# Set the working directory in the container
ENV APP_HOME /app
WORKDIR $APP_HOME

# Copy the local code to the container image
COPY . ./

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install all required packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run web service on container using gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app