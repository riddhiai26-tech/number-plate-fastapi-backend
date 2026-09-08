import base64
import io
from typing import Tuple

from PIL import Image, ImageOps, UnidentifiedImageError

from .config import MAX_IMAGE_PIXELS, MAX_UPLOAD_MB


ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


def load_image(raw: bytes) -> Tuple[Image.Image, str]:
    if not raw:
        raise ValueError("The uploaded file is empty.")

    max_bytes = int(MAX_UPLOAD_MB * 1024 * 1024)
    if len(raw) > max_bytes:
        raise ValueError(f"Image exceeds the {MAX_UPLOAD_MB:g} MB upload limit.")

    try:
        with Image.open(io.BytesIO(raw)) as probe:
            image_format = probe.format or "UNKNOWN"
            if image_format not in ALLOWED_FORMATS:
                raise ValueError("Unsupported image format. Use JPEG, PNG, or WEBP.")
            image = ImageOps.exif_transpose(probe).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc

    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError("Image dimensions are too large.")

    return image, image_format


def pil_to_base64(image: Image.Image, image_format: str = "JPEG", quality: int = 90) -> str:
    buffer = io.BytesIO()
    fmt = image_format.upper()
    if fmt == "JPG":
        fmt = "JPEG"

    save_kwargs = {}
    if fmt == "JPEG":
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True

    image.save(buffer, format=fmt, **save_kwargs)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "image/jpeg" if fmt == "JPEG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{encoded}"
