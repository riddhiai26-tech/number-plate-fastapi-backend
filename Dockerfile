FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/best_license_plate_model.pt \
    DEVICE=cpu \
    IMAGE_SIZE=320 \
    CONFIDENCE_THRESHOLD=0.25 \
    IOU_THRESHOLD=0.7 \
    MAX_UPLOAD_MB=10

# Tesseract OCR + runtime libraries needed by OpenCV/Pillow.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       tesseract-ocr \
       tesseract-ocr-eng \
       libglib2.0-0 \
       libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY app ./app
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
