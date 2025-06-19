# Use an official lightweight Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_ONEDNN_OPTS=0

# Create a working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Upgrade pip and install dependencies
RUN pip install --upgrade pip

# Install required Python libraries (avoid Keras 3 issue)
RUN pip install tensorflow==2.12.0 \
    tf-keras \
    transformers==4.30.2 \
    fastapi \
    uvicorn \
    pydantic

# Expose port
EXPOSE 8000

# Start the FastAPI app with uvicorn
CMD ["uvicorn", "API_sent_main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
