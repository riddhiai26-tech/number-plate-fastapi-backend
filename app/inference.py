import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from .config import CONFIDENCE_THRESHOLD, DEVICE, IMAGE_SIZE, IOU_THRESHOLD
from .image_utils import pil_to_base64
from .ocr import read_plate


class PlateDetector:
    def __init__(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model_path = model_path
        self.model = YOLO(str(model_path))
        self.names = self.model.names

    @property
    def device(self) -> str:
        return DEVICE

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        start_total = time.perf_counter()

        detection_start = time.perf_counter()
        results = self.model.predict(
            source=image,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=DEVICE,
            verbose=False,
        )
        detection_ms = (time.perf_counter() - detection_start) * 1000

        result = results[0]
        boxes = result.boxes

        detections: List[Dict[str, Any]] = []
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)

        # Keep all detections above the configured confidence threshold.
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.detach().cpu().tolist()
            confs = boxes.conf.detach().cpu().tolist()
            classes = boxes.cls.detach().cpu().tolist()

            # Highest-confidence detections first.
            ordered = sorted(
                zip(xyxy, confs, classes),
                key=lambda item: item[1],
                reverse=True,
            )

            ocr_start = time.perf_counter()

            for bbox, confidence, class_id in ordered:
                x1, y1, x2, y2 = bbox
                x1 = max(0, min(int(round(x1)), image.width - 1))
                y1 = max(0, min(int(round(y1)), image.height - 1))
                x2 = max(x1 + 1, min(int(round(x2)), image.width))
                y2 = max(y1 + 1, min(int(round(y2)), image.height))

                crop = image.crop((x1, y1, x2, y2))
                text, ocr_confidence, ocr_status = read_plate(crop)

                class_name = self.names.get(int(class_id), str(int(class_id))) if isinstance(self.names, dict) else self.names[int(class_id)]

                detections.append(
                    {
                        "class_name": class_name,
                        "class_id": int(class_id),
                        "confidence": float(confidence),
                        "bbox": {
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2),
                        },
                        "crop_base64": pil_to_base64(crop, "JPEG"),
                        "ocr": {
                            "text": text,
                            "confidence": ocr_confidence,
                            "status": ocr_status,
                        },
                    }
                )

                label = f"{class_name} {confidence * 100:.2f}%"
                draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 80), width=max(2, image.width // 400))
                # Draw a filled label background without requiring a specific font.
                try:
                    bbox_label = draw.textbbox((0, 0), label)
                    label_h = bbox_label[3] - bbox_label[1]
                    label_w = bbox_label[2] - bbox_label[0]
                except AttributeError:
                    label_w, label_h = draw.textsize(label)
                top = max(0, y1 - label_h - 8)
                draw.rectangle((x1, top, x1 + label_w + 10, top + label_h + 8), fill=(0, 0, 0))
                draw.text((x1 + 5, top + 4), label, fill=(255, 255, 255))

            ocr_ms = (time.perf_counter() - ocr_start) * 1000
        else:
            ocr_ms = 0.0

        total_ms = (time.perf_counter() - start_total) * 1000

        return {
            "detections": detections,
            "annotated_image_base64": pil_to_base64(annotated, "JPEG"),
            "detection_ms": detection_ms,
            "ocr_ms": ocr_ms,
            "total_ms": total_ms,
        }
