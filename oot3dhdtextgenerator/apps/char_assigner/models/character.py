#  Copyright 2020-2023 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
from __future__ import annotations

from base64 import b64encode
from io import BytesIO

import numpy as np
from PIL import Image
from PIL.ImageOps import invert


class Character:
    def __init__(self, array: np.ndarray, assignment: str | None = None) -> None:
        self.array = array
        self.assignment = assignment

    @property
    def image(self):
        image = Image.fromarray(self.array)
        image = invert(image)

        image_io = BytesIO()
        image.save(image_io, format="PNG")
        yat = image_io.getvalue()
        eee = b64encode(yat)
        sam = eee.decode("ascii")
        say = f"data:image/png;base64,{sam}"
        return say
