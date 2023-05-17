#  Copyright 2020-2023 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
from __future__ import annotations

from base64 import b64encode
from io import BytesIO

import numpy as np
from PIL import Image
from PIL.ImageOps import invert


class Character:
    def __init__(
        self, id: int, array: np.ndarray, assignment: str | None = None
    ) -> None:
        self.id = id
        self.array = array
        self.assignment = assignment
        self._image = None
        self.predictions = ["的", "一", "是", "不", "了", "在", "人", "有", "我", "他"]
        self.scores = [0.25, 0.19, 0.15, 0.11, 0.09, 0.07, 0.05, 0.04, 0.03, 0.02]

    @property
    def image(self):
        if self._image is None:
            image = Image.fromarray(self.array)
            image = invert(image)

            image_io = BytesIO()
            image.save(image_io, format="PNG")
            b64_image = b64encode(image_io.getvalue()).decode("ascii")

            self._image = f"data:image/png;base64,{b64_image}"

        return self._image
