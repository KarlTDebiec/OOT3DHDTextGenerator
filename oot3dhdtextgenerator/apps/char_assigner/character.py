#  Copyright 2020-2025 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Character and its associated metadata."""

from __future__ import annotations

from base64 import b64encode
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO

import numpy as np
from PIL import Image
from PIL.ImageOps import invert


@dataclass(slots=True)
class Character:
    """Character and its associated metadata."""

    id: int
    """Integer identifier for the character."""
    array: np.ndarray
    """Array representation of the character image."""
    assignment: str | None = None
    """Assigned value, if any."""
    predictions: list[str] | None = None
    """Potential assignments, if any."""

    @cached_property
    def image(self) -> str:
        """Base64 encoded PNG representation of the character.

        Returns:
            Base64 encoded image string.
        """
        img = Image.fromarray(self.array)
        img = invert(img)

        image_io = BytesIO()
        img.save(image_io, format="PNG")
        b64_image = b64encode(image_io.getvalue()).decode("ascii")

        return f"data:image/png;base64,{b64_image}"
