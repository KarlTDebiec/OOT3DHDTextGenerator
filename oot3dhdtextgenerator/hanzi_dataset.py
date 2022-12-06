#!/usr/bin/env python
#  Copyright 2020-2022 Karl T Debiec
#  All rights reserved. This software may be modified and distributed under
#  the terms of the BSD license. See the LICENSE file for details.
"""Hanzi character dataset."""
from pathlib import Path
from typing import Callable, Optional, Union

import h5py
import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset

from oot3dhdtextgenerator import character_to_index
from oot3dhdtextgenerator.common import validate_input_file


class HanziDataset(VisionDataset):
    """Hanzi character dataset."""

    specification_dtypes = [
        ("character", "U1"),
        ("font", "U256"),
        ("size", "uint8"),
        ("x_offset", "int8"),
        ("y_offset", "int8"),
        ("fill", "uint8"),
        ("rotation", "float32"),
    ]
    encoded_specification_dtypes = [
        ("character", "S4"),
        ("font", "S1024"),
        ("size", "uint8"),
        ("x_offset", "int8"),
        ("y_offset", "int8"),
        ("fill", "uint8"),
        ("rotation", "float32"),
    ]

    def __init__(
        self,
        infile: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        infile = validate_input_file(infile)

        super().__init__(
            str(infile.parent),
            transform=transform,
            target_transform=target_transform,
        )

        self.images, self.specifications = self.load_hdf5(infile)
        print(self.images)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        image = Image.fromarray(self.images[index])
        target = character_to_index(self.specifications[index]["character"])

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self) -> int:
        return len(self.images)

    @classmethod
    def decode_specification(cls, encoded_specifications: np.ndarray) -> np.ndarray:
        specifications = np.array(
            [
                (
                    s["character"].decode("utf8"),
                    s["font"].decode("utf8"),
                    s["size"],
                    s["x_offset"],
                    s["y_offset"],
                    s["fill"],
                    s["rotation"],
                )
                for s in encoded_specifications
            ],
            dtype=cls.specification_dtypes,
        )

        return specifications

    @classmethod
    def encode_specifications(cls, specifications: np.ndarray) -> np.ndarray:
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
            dtype=cls.encoded_specification_dtypes,
        )

        return encoded_specifications

    @classmethod
    def load_hdf5(cls, infile: Path) -> tuple[np.ndarray, np.ndarray]:
        """Load images and specifications from an HDF5 file.

        Arguments:
            infile: Path to HDF5 infile
        Returns:
            Images and specification
        Raises:
            ValueError: If infile does not contain images and specifications
        """
        with h5py.File(infile, "r") as h5_file:
            if not ("images/images" in h5_file and "images/specifications" in h5_file):
                raise ValueError(
                    f"HDF5{infile} does not contain images and specifications"
                )

            images = np.array(h5_file["images/images"])
            encoded_specifications = np.array(h5_file["images/specifications"])
            specifications = cls.decode_specification(encoded_specifications)

        return images, specifications

    @classmethod
    def save_hdf5(
        cls, images: np.ndarray, specifications: np.ndarray, outfile: Path
    ) -> None:
        """Save images and specifications to an HDF5 file.

        Arguments:
            images: Images
            specifications: Specifications
            outfile: Path to HDF5 outfile
        """
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

            h5_file.create_dataset(
                "images/specifications",
                data=cls.encode_specifications(specifications),
                dtype=cls.encoded_specification_dtypes,
                chunks=True,
                compression="gzip",
            )
