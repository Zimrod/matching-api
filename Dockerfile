FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire application code
COPY . .

# Add root to Python path so app/ can import config
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Start the application (main.py is in app/ folder)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]