#!/usr/bin/env python
#  Copyright 2020-2022 Karl T Debiec
#  All rights reserved. This software may be modified and distributed under
#  the terms of the BSD license. See the LICENSE file for details.
"""Hanzi character dataset."""
from pathlib import Path
from typing import Callable, Optional

import h5py
import numpy as np
from torchvision.datasets import VisionDataset


class HanziDataset(VisionDataset):
    """Hanzi character dataset."""

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

    @staticmethod
    def save_hdf5(
        images: np.ndarray, specifications: np.ndarray, outfile: Path
    ) -> None:
        with h5py.File(outfile, "w") as h5_file:
            if "images" in h5_file:
                del h5_file["images"]
            h5_file.create_dataset(
                "images/images",
                data=images,
                dtype=np.uint8,
                chunks=True,
                compression="gzip",
            )
            encoded_dtypes = [
                ("character", "S1"),
                ("font", "S255"),
                ("size", "uint8"),
                ("x_offset", "int8"),
                ("y_offset", "int8"),
                ("fill", "uint8"),
                ("rotation", "float32"),
            ]
            encoded_specifications = np.array(
                [
                    (
                        s["character"].encode("utf8"),
                        s["font"].encode("utf8"),
                        s["size"],
                        s["x_offset"],
                        s["y_offset"],
                        s["fill"],
                        s["rotation"],
                    )
                    for s in specifications
                ],
                dtype=encoded_dtypes,
            )

            h5_file.create_dataset(
                "images/specifications",
                data=encoded_specifications,
                dtype=encoded_dtypes,
                chunks=True,
                compression="gzip",
            )
