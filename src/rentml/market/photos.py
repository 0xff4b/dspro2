"""Validation and re-encoding of uploaded listing photos.

A browser upload arrives as a data URL (``data:image/png;base64,...``). Every photo is parsed
with Pillow, rotated according to its EXIF orientation, scaled down and saved again as JPEG. The
re-encoding drops all metadata (camera, GPS position) and guarantees that only pixel data that
Pillow could parse is stored and served.
"""

import base64
import binascii
import io
from collections.abc import Sequence

from PIL import Image, ImageOps, UnidentifiedImageError

from rentml.market.models import MarketError

MAX_PHOTOS = 6
MAX_UPLOAD_BYTES = 6 * 1024 * 1024
MAX_PIXELS = 40_000_000
MAX_SIDE = 1600
JPEG_QUALITY = 82
ALLOWED_FORMATS: tuple[str, ...] = ("JPEG", "PNG", "WEBP")
_UNREADABLE = "Ein Foto konnte nicht gelesen werden (erlaubt sind JPG, PNG und WebP)."


class PhotoError(MarketError):
    """An uploaded photo is too large, not an image or in an unsupported format."""


def decode_data_url(content: str) -> bytes:
    """Raw bytes of a base64 image data URL as sent by ``dcc.Upload``.

    Args:
        content: ``data:image/<type>;base64,<payload>``.

    Returns:
        The decoded bytes.

    Raises:
        PhotoError: If it is no base64 image data URL or larger than ``MAX_UPLOAD_BYTES``.
    """
    header, sep, payload = str(content).partition(",")
    if not sep or not header.startswith("data:image/") or not header.endswith(";base64"):
        raise PhotoError(_UNREADABLE)
    if len(payload) * 3 // 4 > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise PhotoError(f"Ein Foto ist grösser als {limit_mb} MB.")
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise PhotoError(_UNREADABLE) from exc


def _flatten(image: Image.Image) -> Image.Image:
    if image.mode not in ("RGBA", "LA", "P"):
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    background = Image.new("RGB", rgba.size, "white")  # JPEG has no transparency
    background.paste(rgba, mask=rgba.getchannel("A"))
    return background


def to_jpeg(raw: bytes) -> bytes:
    """Re-encode an uploaded image as a metadata-free JPEG of at most ``MAX_SIDE`` pixels.

    Args:
        raw: Bytes of a JPEG, PNG or WebP image.

    Returns:
        JPEG bytes.

    Raises:
        PhotoError: If the bytes are no supported image or the image is too large.
    """
    try:
        with Image.open(io.BytesIO(raw)) as source:
            if source.format not in ALLOWED_FORMATS:
                raise PhotoError(_UNREADABLE)
            if source.width * source.height > MAX_PIXELS:  # checked before the pixels are decoded
                raise PhotoError("Ein Foto hat zu viele Bildpunkte (max. 40 Megapixel).")
            image = _flatten(ImageOps.exif_transpose(source))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise PhotoError(_UNREADABLE) from exc
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return out.getvalue()


def prepare(contents: Sequence[str] | None) -> list[bytes]:
    """Validate and re-encode all photos of one listing.

    Args:
        contents: Data URLs from ``dcc.Upload`` (``None`` or empty = no photos).

    Returns:
        JPEG bytes in upload order.

    Raises:
        PhotoError: For more than ``MAX_PHOTOS`` photos or an invalid photo.
    """
    items = list(contents or [])
    if len(items) > MAX_PHOTOS:
        raise PhotoError(f"Höchstens {MAX_PHOTOS} Fotos pro Inserat.")
    return [to_jpeg(decode_data_url(item)) for item in items]
