#  Copyright 2020-2023 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Learning dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import h5py
import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset

from oot3dhdtextgenerator.common import PathLike, validate_input_file
from oot3dhdtextgenerator.data import character_to_index


class LearningDataset(VisionDataset):
    """Learning dataset."""

    specification_dtypes = [
        ("character", "U1"),
        ("font", "U256"),
        ("size", "uint8"),
        ("x_offset", "int8"),
        ("y_offset", "int8"),
        ("fill", "uint8"),
        ("rotation", "float32"),
    ]
    """Specification dtypes"""
    encoded_specification_dtypes = [
        ("character", "S4"),
        ("font", "S1024"),
        ("size", "uint8"),
        ("x_offset", "int8"),
        ("y_offset", "int8"),
        ("fill", "uint8"),
        ("rotation", "float32"),
    ]
    """Encoded specification dtypes for HDF5"""

    def __init__(
        self,
        infile: PathLike,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        """Initialize.

        Arguments:
            infile: Path to HDF5 file
            transform: Transform to apply to images
            target_transform: Transform to apply to targets
        """
        infile = validate_input_file(infile)
        super().__init__(
            str(infile.parent),
            transform=transform,
            target_transform=target_transform,
        )
        self.images, self.specifications = self.load_hdf5(infile)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        """Get image and target at index."""
        image = Image.fromarray(self.images[index])
        target = character_to_index(self.specifications[index]["character"])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target

    def __len__(self) -> int:
        """Number of images in the dataset."""
        return len(self.images)

    @classmethod
    def decode_specification(cls, encoded_specifications: np.ndarray) -> np.ndarray:
        """Decode specifications from HDF5 file.

        Arguments:
            encoded_specifications: Specifications to decode
        Returns:
            Decoded specifications
        """
        specifications = np.array(
            [
                (
                    s["character"].decode("utf-8"),
                    s["font"].decode("utf-8"),
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
        """Encode specifications for storage in HDF5 file.

        Arguments:
            specifications: Specifications to encode
        Returns:
            Encoded specifications
        """
        encoded_specifications = np.array(
            [
                (
                    s["character"].encode("utf-8"),
                    s["font"].encode("utf-8"),
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
    def load_hdf5(
        cls,
        infile: Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load images and specifications from an HDF5 file.

        Arguments:
            infile: Path to HDF5 infile
        Returns:
            Train images, train specifications, test images, and test specifications
        Raises:
            ValueError: If infile does not contain images and specifications
        """
        with h5py.File(infile, "r") as h5_file:
            if "images" not in h5_file or "specifications" not in h5_file:
                raise ValueError(
                    f"HDF5{infile} does not contain images and specifications"
                )

            images = np.array(h5_file["images"])
            encoded_specifications = np.array(h5_file["specifications"])
            specifications = cls.decode_specification(encoded_specifications)

        return images, specifications

    @classmethod
    def save_hdf5(
        cls,
        images: np.ndarray,
        specifications: np.ndarray,
        outfile: Path,
    ) -> None:
        """Save images and specifications to an HDF5 file.

        Arguments:
            images: Train images
            specifications: Train image specifications
            outfile: Path to HDF5 outfile
        """
        with h5py.File(outfile, "w") as h5_file:
            if "images" in h5_file:
                del h5_file["images"]
            h5_file.create_dataset(
                "images",
                data=images,
                dtype=np.uint8,
                chunks=True,
                compression="gzip",
            )

            if "specifications" in h5_file:
                del h5_file["specifications"]
            h5_file.create_dataset(
                "specifications",
                data=cls.encode_specifications(specifications),
                dtype=cls.encoded_specification_dtypes,
                chunks=True,
                compression="gzip",
            )
