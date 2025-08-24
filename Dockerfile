FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Copy only requirements first (layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

 # Create necessary directories
RUN mkdir -p /app/data /app/db

EXPOSE 8000

# Start the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]