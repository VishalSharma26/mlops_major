# Use Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt requirements.txt
COPY src/ src/
COPY tests/ tests/
COPY model/ model/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run training by default
CMD ["python", "src/train.py"]
