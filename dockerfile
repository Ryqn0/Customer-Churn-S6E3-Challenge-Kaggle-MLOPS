# Use official Python image as the base image
FROM python:3.13-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application code into the container
COPY . .

# Copy the pre-trained model into the container
COPY models/voting_model.pkl .

# Copy models metadata and artifacts from MLFlow into the container
COPY mlruns/894875891458910561/models .

# src/ importable : from data.preprocess, from features.build_features, etc.
ENV PYTHONPATH=/app/src:/app

# Expose the port that the FastAPI app will run on
EXPOSE 8000

# Command to run the FastAPI app using Uvicorn
CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
