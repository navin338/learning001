FROM python:3.9-slim

WORKDIR /app

# Install system dependencies that some ML libs need
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set PYTHONPATH to current directory
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "main_v1:app", "--host", "0.0.0.0", "--port", "8000"]