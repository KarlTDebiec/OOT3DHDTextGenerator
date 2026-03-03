#  Copyright 2020-2026 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Character and its associated metadata."""

from __future__ import annotations

from base64 import b64encode
from dataclasses import dataclass, field
from io import BytesIO
from typing import TYPE_CHECKING

from PIL import Image
from PIL.ImageOps import invert

if TYPE_CHECKING:
    import numpy as np


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
    score: np.ndarray | None = field(default=None, repr=False)
    """Raw model score vector."""
    _image: str | None = field(default=None, init=False, repr=False)
    """Cached base64 encoded PNG representation of the character."""

    @property
    def image(self) -> str:
        """Base64 encoded PNG representation of the character.

        Returns:
            base64 encoded image string.
        """
        if self._image is not None:
            return self._image

        img = Image.fromarray(self.array)
        img = invert(img)

        image_io = BytesIO()
        img.save(image_io, format="PNG")
        b64_image = b64encode(image_io.getvalue()).decode("ascii")
        self._image = f"data:image/png;base64,{b64_image}"
        return self._image
