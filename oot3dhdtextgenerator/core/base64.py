#  Copyright 2020-2026 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Helper functions for AssignmentDataset persistence."""

from __future__ import annotations

from base64 import b64decode, b64encode
from binascii import Error as BinasciiError
from io import BytesIO

import numpy as np
from PIL import Image


def array_to_raw_base64_png(char_array: np.ndarray) -> str:
    """Convert an array into a raw base64-encoded PNG payload.

    Arguments:
        char_array: character array to encode
    Returns:
        raw base64 PNG payload
    """
    with BytesIO() as png_bytes:
        Image.fromarray(char_array, mode="L").save(png_bytes, format="PNG")
        return b64encode(png_bytes.getvalue()).decode("ascii")


def raw_base64_png_to_array(raw_base64_png: str) -> np.ndarray:
    """Convert a raw base64-encoded PNG payload into a grayscale array.

    Arguments:
        raw_base64_png: raw base64 PNG payload
    Returns:
        grayscale array
    Raises:
        ValueError: if payload cannot be decoded as PNG
    """
    try:
        png_bytes = b64decode(raw_base64_png, validate=True)
        with Image.open(BytesIO(png_bytes)) as image:
            return np.array(image.convert("L"), dtype=np.uint8)
    except (BinasciiError, OSError, ValueError) as exc:
        raise ValueError("Invalid base64 PNG payload") from exc
