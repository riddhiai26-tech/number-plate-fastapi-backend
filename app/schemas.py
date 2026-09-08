from typing import List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class OCRResult(BaseModel):
    text: Optional[str] = None
    confidence: Optional[float] = None
    status: str = "unreadable"


class Detection(BaseModel):
    class_name: str
    class_id: int
    confidence: float = Field(ge=0, le=1)
    bbox: BoundingBox
    crop_base64: Optional[str] = None
    ocr: OCRResult


class ImageInfo(BaseModel):
    width: int
    height: int
    format: Optional[str] = None


class ProcessingInfo(BaseModel):
    detection_ms: float
    ocr_ms: float
    total_ms: float
    device: str
    image_size: int


class PredictResponse(BaseModel):
    success: bool
    reason: Optional[str] = None
    message: Optional[str] = None
    detections: List[Detection] = []
    image: Optional[ImageInfo] = None
    processing: Optional[ProcessingInfo] = None
    annotated_image_base64: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model: str
    model_loaded: bool
    ocr_available: bool
    device: str
