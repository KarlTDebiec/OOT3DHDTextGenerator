#  Copyright 2020-2025 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Character and its associated metadata."""

from __future__ import annotations

from base64 import b64encode
from io import BytesIO

import numpy as np
from PIL import Image
from PIL.ImageOps import invert


class Character:
    """Character and its associated metadata."""

    def __init__(
        self,
        character_id: int,
        array: np.ndarray,
        assignment: str | None = None,
        predictions: list[str] | None = None,
    ) -> None:
        """Initialize.

        Arguments:
            character_id: Integer identifier for the character.
            array: Array representation of the character image.
            assignment: Assigned value, if any.
            predictions: Potential assignments, if any.
        Returns:
            None
        """
        self.id = character_id
        self.array = array
        self.assignment = assignment
        self._image = None
        self.predictions = predictions

    @property
    def image(self) -> str:
        """Base64 encoded PNG representation of the character.

        Returns:
            Base64 encoded image string.
        """
        if self._image is None:
            image = Image.fromarray(self.array)
            image = invert(image)

            image_io = BytesIO()
            image.save(image_io, format="PNG")
            b64_image = b64encode(image_io.getvalue()).decode("ascii")

            self._image = f"data:image/png;base64,{b64_image}"

        return self._image
