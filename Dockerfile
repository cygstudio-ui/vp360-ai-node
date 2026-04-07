FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    cmake build-essential libopenblas-dev \
    liblapack-dev libjpeg-dev python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi "uvicorn[standard]" \
    python-multipart requests \
    numpy dlib face_recognition

COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
