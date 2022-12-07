#!/usr/bin/env python
#  Copyright 2020-2022 Karl T Debiec
#  All rights reserved. This software may be modified and distributed under
#  the terms of the BSD license. See the LICENSE file for details.
"""Assignment dataset."""
from logging import info
from pathlib import Path
from typing import Iterable, Optional

import h5py
import numpy as np

from oot3dhdtextgenerator.common import validate_output_file


class AssignmentDataset:
    """Assignment dataset."""

    def __init__(self, infile: Path) -> None:
        """Initialize."""
        infile = validate_output_file(infile, exists_ok=True)
        if infile.exists():
            assigned_chars, unassigned_chars = self.load_hdf5(infile)
        else:
            assigned_chars, unassigned_chars = {}, set()

        self.assigned_chars = assigned_chars
        """Dictionary whose keys are image hashes and values are characters"""
        self.unassigned_chars = unassigned_chars
        """Set of image hashes of unassigned characters"""

    def __getitem__(self, image: np.ndarray) -> Optional[str]:
        """Get assignment for image.

        Arguments:
            image: Image to assign
        """
        if image.shape in ((128, 256, 4), (256, 256, 4)):
            all_chars_assigned = True
            chars = []

            breaking = False
            for x in range(0, image.shape[0], 16):
                for y in range(0, image.shape[1], 16):
                    char_array = image[x : x + 16, y : y + 16]
                    if char_array.sum() == 0:
                        breaking = True
                        break
                    char = self[char_array]
                    if char is None:
                        all_chars_assigned = False
                    else:
                        chars.append(char)
                if breaking:
                    break
            if all_chars_assigned:
                return chars
            return None

        char_bytes = image.tobytes()
        if char_bytes in self.assigned_chars:
            info(f"Assigned character {char_bytes} retrieved")
            return self.assigned_chars[char_bytes]
        elif char_bytes not in self.unassigned_chars:
            info(f"Unassigned character added")
            self.unassigned_chars.add(char_bytes)
        return None

    @classmethod
    def decode_assignments(cls, assignments: list[bytes]) -> list[str]:
        """Decode assignments from HDF5 file.

        Arguments:
            assignments: Assignments to decode
        Returns:
            Decoded assignments
        """
        return [assignment.decode("utf-8") for assignment in assignments]

    @classmethod
    def decode_images(cls, images: Iterable[bytes]) -> np.ndarray:
        """Decode images for storage in HDF5 file.

        Arguments:
            images: Images to decode
        Returns:
            Decoded images
        """
        return np.array(
            [np.frombuffer(image, dtype=np.uint8) for image in images]
        ).reshape((-1, 16, 16))

    @classmethod
    def encode_assignments(cls, assignments: Iterable[str]) -> list[bytes]:
        """Encode assignments for storage in HDF5 file.

        Arguments:
            assignments: Assignments to encode
        Returns:
            Encoded assignments
        """
        return [assignment.encode("utf-8") for assignment in assignments]

    @classmethod
    def encode_images(cls, images: np.ndarray) -> list[bytes]:
        """Encode images from HDF5 file.

        Arguments:
            images: Images to encode
        Returns:
            Encoded images
        """
        return [image.tobytes() for image in images]

    @classmethod
    def load_hdf5(cls, infile: Path) -> tuple[dict[bytes, str], set[bytes]]:
        """Load assignments from HDF5 file.

        Arguments:
            infile: Path to HDF5 file
        Returns:
            Assigned images in dictionary whose keys are image bytes and values are
            assigned characters, and unassigned images in set of bytes
        """
        with h5py.File(infile, "r") as h5_file:
            if (
                "assigned" not in h5_file
                or "assignments" not in h5_file
                or "unassigned" not in h5_file
            ):
                raise ValueError(
                    f"{infile} does not contain assigned and unassigned images"
                )
            assigned = cls.encode_images(h5_file["assigned"])
            assignments = cls.decode_assignments(h5_file["assignments"])
            unassigned = cls.encode_images(h5_file["unassigned"])

        return dict(zip(assigned, assignments)), set(unassigned)

    @classmethod
    def save_hdf5(
        cls,
        assigned_images: dict[bytes, str],
        unassigned_images: set[bytes],
        outfile: Path,
    ) -> None:
        """Save images and assignments to an HDF5 file.

        Arguments:
            assigned_images: Assigned images
            unassigned_images: Unassigned images
            outfile: Path to HDF5 outfile
        """
        with h5py.File(outfile, "w") as h5_file:
            if "assigned" in h5_file:
                del h5_file["assigned"]
            h5_file.create_dataset(
                f"assigned",
                data=cls.decode_images(assigned_images.keys()),
                dtype=np.uint8,
                chunks=True,
                compression="gzip",
            )

            if "assignments" in h5_file:
                del h5_file["assignments"]
            h5_file.create_dataset(
                f"assignments",
                data=cls.encode_assignments(assigned_images.values()),
                dtype="S4",
                chunks=True,
                compression="gzip",
            )

            if "unassigned" in h5_file:
                del h5_file["unassigned"]
            h5_file.create_dataset(
                f"unassigned",
                data=cls.decode_images(unassigned_images),
                dtype=np.uint8,
                chunks=True,
                compression="gzip",
            )
