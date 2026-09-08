import shutil
import tempfile
from pathlib import Path

import pytesseract
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import ALLOWED_ORIGINS, MODEL_PATH, DEVICE
from .image_utils import load_image
from .inference import PlateDetector
from .schemas import (
    BoundingBox,
    Detection,
    HealthResponse,
    ImageInfo,
    OCRResult,
    PredictResponse,
    ProcessingInfo,
)

app = FastAPI(
    title="License Plate Detection API",
    description="YOLOv8 license-plate detection followed by Tesseract OCR.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

detector = None
model_load_error = None


@app.on_event("startup")
def load_model() -> None:
    global detector, model_load_error
    try:
        detector = PlateDetector(MODEL_PATH)
        model_load_error = None
    except Exception as exc:
        detector = None
        model_load_error = str(exc)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        pytesseract.get_tesseract_version()
        ocr_available = True
    except Exception:
        ocr_available = False

    return HealthResponse(
        status="ok" if detector is not None else "degraded",
        model=MODEL_PATH.name,
        model_loaded=detector is not None,
        ocr_available=ocr_available,
        device=DEVICE,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "model_unavailable",
                "message": "The YOLO model could not be loaded.",
                "reason": model_load_error,
            },
        )

    content_type = (file.content_type or "").lower()
    allowed_mimes = {"image/jpeg", "image/png", "image/webp"}
    if content_type and content_type not in allowed_mimes:
        raise HTTPException(
            status_code=415,
            detail="Unsupported media type. Upload JPEG, PNG, or WEBP.",
        )

    raw = await file.read()

    try:
        image, image_format = load_image(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        output = detector.predict(image)
    except Exception as exc:
        # Do not expose internal stack traces or model paths to the client.
        raise HTTPException(
            status_code=500,
            detail="Inference failed. Check the backend logs.",
        ) from exc

    detections = [
        Detection(
            class_name=item["class_name"],
            class_id=item["class_id"],
            confidence=item["confidence"],
            bbox=BoundingBox(**item["bbox"]),
            crop_base64=item["crop_base64"],
            ocr=OCRResult(**item["ocr"]),
        )
        for item in output["detections"]
    ]

    if not detections:
        return PredictResponse(
            success=False,
            reason="no_detection",
            message="No license plate was detected in the supplied image.",
            detections=[],
            image=ImageInfo(width=image.width, height=image.height, format=image_format),
            processing=ProcessingInfo(
                detection_ms=output["detection_ms"],
                ocr_ms=output["ocr_ms"],
                total_ms=output["total_ms"],
                device=DEVICE,
                image_size=320,
            ),
            annotated_image_base64=output["annotated_image_base64"],
        )

    return PredictResponse(
        success=True,
        reason=None,
        message=f"{len(detections)} license plate detection(s) found.",
        detections=detections,
        image=ImageInfo(width=image.width, height=image.height, format=image_format),
        processing=ProcessingInfo(
            detection_ms=output["detection_ms"],
            ocr_ms=output["ocr_ms"],
            total_ms=output["total_ms"],
            device=DEVICE,
            image_size=320,
        ),
        annotated_image_base64=output["annotated_image_base64"],
    )
