#!/usr/bin/env python
#   Copyright (C) 2020 Karl T Debiec
#   All rights reserved. This software may be modified and distributed under
#   the terms of the BSD license. See the LICENSE file for details.
""""""
from __future__ import annotations

from typing import List

import numpy as np
from PIL import Image
from pipescaler.core.pipelines import Sorter


class OOT3DSorter(Sorter):
    """Sorts images."""

    def __call__(self, infile: str) -> str:
        data = np.array(Image.open(infile))

        if data.shape in [(128, 256, 4), (128, 512, 4), (256, 256, 4)]:
            if data[:, :, :3].sum() == 0:
                return "text"
            y, x, _ = np.where(data == 255)
            if x.size > 0 and y.size > 0 and y.max() == 15 and x.max() < 128:
                return "time_text"
        elif data.shape == (32, 256, 4) or data.shape == (64, 256, 4):
            if (
                np.all(data[:, :, 0] == data[:, :, 1])
                and np.all(data[:, :, 0] == data[:, :, 2])
                and (data[:, :, 3] != 255).sum() != 0
            ):
                return "large_text"
        elif data.shape == (64, 64, 4):
            bincount = np.where(np.bincount(data.flatten()) != 0)[0]
            if (
                bincount.size == 4 and np.all(bincount == np.array([0, 34, 136, 170]))
            ) or (
                bincount.size == 10
                and np.all(
                    bincount == np.array([0, 34, 51, 68, 85, 102, 119, 136, 153, 170])
                )
            ):
                return "shadow"
        return "default"

    @property
    def outlets(self) -> List[str]:
        return ["default", "large_text", "shadow", "text", "time_text"]
