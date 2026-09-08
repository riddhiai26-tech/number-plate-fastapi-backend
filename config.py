from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "best_license_plate_model.pt")))

DEVICE = os.getenv("DEVICE", "cpu")
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "320"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.7"))
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "10"))

# Comma-separated origins, e.g.:
# ALLOWED_ORIGINS=https://your-lovable-site.lovable.app,http://localhost:5173
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
    if origin.strip()
]

TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")
TESSERACT_LANG = os.getenv("TESSERACT_LANG", "eng")
TESSERACT_CONFIG = os.getenv("TESSERACT_CONFIG", "--psm 6")

# Prevent accidentally processing very large decoded images.
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "25000000"))
