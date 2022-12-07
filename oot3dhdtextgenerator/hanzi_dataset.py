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

# TODO: Do not load entire dataset into memory
#   Keep h5py file open and read from it as needed
#   Need to update __len__ and __getitem__
#   May also add new save_h5py and load_h5py methods
#   Not clear if the old methods ought to be kept around


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
        infile: Union[str, Path],
        name: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """Initialize.

        Arguments:
            infile: Path to HDF5 file
            name: Name of dataset within HDF5 file
            transform: Transform to apply to images
            target_transform: Transform to apply to targets
        """
        infile = validate_input_file(infile)
        super().__init__(
            str(infile.parent),
            transform=transform,
            target_transform=target_transform,
        )
        self.images, self.specifications = self.load_hdf5(infile, name)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        """Get image and target at index."""
        image = Image.fromarray(self.images[index])
        target = character_to_index(self.specifications[index]["character"])

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
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
        """Encode specifications for storage in HDF5 file.

        Arguments:
            specifications: Specifications to encode
        Returns:
            Encoded specifications
        """
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
    def load_hdf5(
        cls,
        infile: Path,
        name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load images and specifications from an HDF5 file.

        Arguments:
            infile: Path to HDF5 infile
            name: Name of dataset within HDF5 file
        Returns:
            Train images, train specifications, test images, and test specifications
        Raises:
            ValueError: If infile does not contain images and specifications
        """
        with h5py.File(infile, "r") as h5_file:
            if (
                f"{name}/images" not in h5_file
                or f"{name}/specifications" not in h5_file
            ):
                raise ValueError(
                    f"HDF5{infile} does not contain '{name}' images and specifications"
                )

            images = np.array(h5_file[f"{name}/images"])
            encoded_specifications = np.array(h5_file[f"{name}/specifications"])
            specifications = cls.decode_specification(encoded_specifications)

        return images, specifications

    @classmethod
    def save_hdf5(
        cls,
        images: np.ndarray,
        specifications: np.ndarray,
        outfile: Path,
        name: str,
    ) -> None:
        """Save images and specifications to an HDF5 file.

        Arguments:
            images: Train images
            specifications: Train image specifications
            outfile: Path to HDF5 outfile
            name: Name of dataset within HDF5 file
        """
        with h5py.File(outfile, "w") as h5_file:
            if name in h5_file:
                del h5_file[name]

            h5_file.create_dataset(
                f"{name}/images",
                data=images,
                dtype=np.uint8,
                chunks=True,
                compression="gzip",
            )

            h5_file.create_dataset(
                f"{name}/specifications",
                data=cls.encode_specifications(specifications),
                dtype=cls.encoded_specification_dtypes,
                chunks=True,
                compression="gzip",
            )
