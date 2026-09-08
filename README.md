# Number Plate Detection FastAPI Backend

Backend for the project **Deep Learning–Based Number Plate Detection Using YOLOv8**.

It uses the supplied trained checkpoint:

`models/best_license_plate_model.pt`

and follows the notebook's two-stage workflow:

1. YOLOv8 detects the license-plate bounding box.
2. The detected region is cropped.
3. Tesseract OCR attempts to read the crop.

The backend never fabricates a plate number or converts model confidence into an "accuracy" claim.

## API

### `GET /health`

Returns model/OCR availability.

### `POST /predict`

Multipart form upload:

- field: `file`
- accepted: JPEG, PNG, WEBP
- default maximum: 10 MB

The response contains:

- `success`
- `detections[]`
- bounding box coordinates
- YOLO confidence
- cropped plate as a data URL
- OCR text/status
- optional OCR confidence returned by Tesseract
- annotated image as a data URL
- processing timings
- image dimensions

## Local setup

Python 3.11 is recommended because the project pins the same major Ultralytics release used in the supplied notebook.

Install Tesseract OCR separately, then:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open:

`http://localhost:8000/docs`

## Docker

```bash
docker build -t number-plate-backend .
docker run --rm -p 8000:8000 \
  -e ALLOWED_ORIGINS=http://localhost:5173 \
  number-plate-backend
```

Then test:

`http://localhost:8000/health`

## Frontend integration

Send a `POST` request to:

`https://YOUR-BACKEND-DOMAIN/predict`

with `multipart/form-data` and the field name `file`.

For the Lovable Detection Lab:

- use `annotated_image_base64` for the finished detection visualization;
- use `detections[0].bbox` to draw a responsive box over the original image;
- use `detections[0].crop_base64` for the cropped plate panel;
- use `detections[0].ocr.text` for the OCR result;
- use `detections[0].confidence` as detection confidence;
- use `processing.*` for technical timing metadata.

Do not label YOLO confidence as model accuracy.

## Important deployment note

The `.pt` file is a Python/Ultralytics model checkpoint. It is intended to run in the Python backend, not directly in the browser.

For a public deployment, keep the model and server-side dependencies on the backend. Restrict `ALLOWED_ORIGINS` to the actual website origin.

The default Docker configuration is CPU inference for compatibility. A GPU deployment can be faster but requires a compatible CUDA/PyTorch environment.

## Failure behavior

If no plate is found, `/predict` returns HTTP 200 with:

`success=false` and `reason="no_detection"`.

If the model is unavailable, `/predict` returns HTTP 503.

If OCR cannot read the crop, the detection remains valid and the OCR status is `unreadable`; the backend does not guess the text.

## Source alignment

The supplied notebook trained `yolov8n.pt` for 100 epochs at 320px and later saved the trained checkpoint as `best_license_plate_model.pt`. Its OCR example uses Tesseract with `--psm 6`. This backend preserves those core choices while adding production API/error-handling structure.
