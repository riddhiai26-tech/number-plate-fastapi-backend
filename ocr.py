import re
from typing import Optional, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image

from .config import TESSERACT_CMD, TESSERACT_CONFIG, TESSERACT_LANG

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

PLATE_CHARS = re.compile(r"[^A-Z0-9]")


def _clean_text(text: str) -> str:
    # Keep the OCR result faithful to Tesseract while normalizing whitespace.
    # We do NOT invent or "correct" characters into a plausible plate number.
    text = re.sub(r"\s+", " ", text).strip().upper()
    return text


def _ocr_once(image: Image.Image) -> Tuple[Optional[str], Optional[float]]:
    text = pytesseract.image_to_string(
        image,
        lang=TESSERACT_LANG,
        config=TESSERACT_CONFIG,
    )
    cleaned = _clean_text(text)
    if not cleaned:
        return None, None

    try:
        data = pytesseract.image_to_data(
            image,
            lang=TESSERACT_LANG,
            config=TESSERACT_CONFIG,
            output_type=pytesseract.Output.DICT,
        )
        confidences = [
            float(c)
            for c in data.get("conf", [])
            if c not in ("", None, "-1") and float(c) >= 0
        ]
        confidence = (sum(confidences) / len(confidences) / 100.0) if confidences else None
    except Exception:
        confidence = None

    return cleaned, confidence


def read_plate(crop: Image.Image) -> Tuple[Optional[str], Optional[float], str]:
    """
    Primary OCR behavior follows the supplied notebook:
    Tesseract with --psm 6 on the YOLO crop.

    A small grayscale/upscale fallback is used only when the primary OCR
    returns no text. No character substitution or plate-format guessing is done.
    """
    text, confidence = _ocr_once(crop)
    if text:
        return text, confidence, "ok"

    # Conservative fallback for difficult crops.
    arr = np.array(crop)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    scale = 3
    upscaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    fallback = Image.fromarray(upscaled)
    text, confidence = _ocr_once(fallback)

    if text:
        return text, confidence, "ok_fallback"

    return None, None, "unreadable"
