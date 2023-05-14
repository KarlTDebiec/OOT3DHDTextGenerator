#  Copyright 2020-2023 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Assignment project."""
from __future__ import annotations

from logging import info
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from PIL import Image
from torch import Tensor
from torchvision.datasets import VisionDataset
from torchvision.transforms import Compose, Normalize, ToTensor

from oot3dhdtextgenerator.common import validate_input_file


class AssignmentDataset(VisionDataset):
    """Assignment project."""

    multi_char_array_shapes = ((128, 256), (128, 512), (256, 256))
    char_array_shape = (16, 16)

    def __init__(self, infile: Path) -> None:
        """Initialize."""
        infile = validate_input_file(infile, must_exist=False)

        transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
        super().__init__(str(infile.parent), transform=transform)

        assigned_char_bytes, unassigned_char_bytes = {}, []
        if infile.exists():
            assigned_char_bytes, unassigned_char_bytes = self.load_hdf5(infile)

        self.assigned_char_bytes = assigned_char_bytes
        """Dictionary whose keys are char bytes and values are char strs"""
        self.unassigned_char_bytes = unassigned_char_bytes
        """List of unassigned char bytes"""

    def __getitem__(self, index: int) -> Tensor:
        """Get unassigned char Tensor at index."""
        char_bytes = self.unassigned_char_bytes[index]
        char_array = self.bytes_to_array(char_bytes)
        char_image = Image.fromarray(char_array)
        char_tensor = self.transform(char_image)

        return char_tensor

    def __len__(self) -> int:
        """Number of images in the dataset."""
        return len(self.unassigned_char_bytes)

    def __str__(self) -> str:
        """String representation."""
        return f"<{self.__class__.__name__}>"

    def assign(self, char_array: np.ndarray, char: str) -> None:
        """Assign char to char array.

        Arguments:
            char_array: Char array to assign
            char: Char to assign
        """
        char_bytes = self.array_to_bytes(char_array)
        if char_bytes not in self.unassigned_char_bytes:
            raise ValueError(f"Character not unassigned")
        if char_bytes in self.assigned_char_bytes:
            raise ValueError(
                "Character already assigned to "
                f"{self.assigned_char_bytes[char_bytes]}, cannot assign to {char}"
            )
        if len(char) != 1:
            raise ValueError(f"Character {char} must be a single character")

        self.unassigned_char_bytes.pop(self.unassigned_char_bytes.index(char_bytes))
        self.assigned_char_bytes[char_bytes] = char

    def get_chars_for_multi_char_array(
        self, multi_char_array: np.ndarray
    ) -> str | None:
        """Get chars for a multi-char array, if all assigned, or None otherwise.

        Arguments:
            multi_char_array: Multi-char array whose chars to retrieve
        Returns:
            Chars, or None if not all chars are assigned
        """
        if multi_char_array.shape not in self.multi_char_array_shapes:
            raise ValueError(
                f"Invalid array shape {multi_char_array.shape}, "
                f"expected one of {self.multi_char_array_shapes}"
            )

        # Extract each character and check if it is assigned
        all_chars_assigned = True
        chars = []
        breaking = False
        for x in range(0, multi_char_array.shape[0], self.char_array_shape[0]):
            for y in range(0, multi_char_array.shape[1], self.char_array_shape[1]):
                char_array = multi_char_array[
                    x : x + self.char_array_shape[0], y : y + self.char_array_shape[1]
                ]
                if char_array.sum() == 0:
                    breaking = True
                    break
                char = self.get_char_for_char_array(char_array)
                if char is None:
                    all_chars_assigned = False
                else:
                    chars.append(char)
            if breaking:
                break

        # Return assignments, if all characters are assigned, or None otherwise
        if all_chars_assigned:
            return "".join(chars)
        return None

    def get_char_for_char_array(self, char_array: np.ndarray) -> str | None:
        """Get char for a char array, if assigned, or None otherwise.

        If char_array is not assigned, it is added to self.unassigned_char_arrays.

        Arguments:
            char_array: Char array whose char to retrieve
        Returns:
            Char, or None if not assigned
        """
        if char_array.shape != self.char_array_shape:
            raise ValueError(
                f"Invalid array shape {char_array.shape}, "
                f"expected {self.char_array_shape}"
            )

        char_bytes = self.array_to_bytes(char_array)
        if char_bytes in self.assigned_char_bytes:
            char = self.assigned_char_bytes[char_bytes]
            info(f"Assigned character {char} retrieved")
            return char
        elif char_bytes not in self.unassigned_char_bytes:
            self.unassigned_char_bytes.append(char_bytes)
            info(f"Unassigned character added, {len(self.unassigned_char_bytes)} total")
        return None

    @classmethod
    def array_to_bytes(cls, char_array: np.ndarray) -> bytes:
        """Convert char array to char bytes.

        Arguments:
            char_array: Char array
        Returns:
            char bytes
        """
        return char_array.tobytes()

    @classmethod
    def bytes_to_array(cls, char_bytes: Iterable[bytes]) -> np.ndarray:
        """Convert char bytes to char array.

        Arguments:
            char_bytes: Char bytes
        Returns:
            Char array
        """
        return np.frombuffer(char_bytes, dtype=np.uint8).reshape(cls.char_array_shape)

    @classmethod
    def decode_chars(cls, encoded_chars: Iterable[bytes]) -> list[str]:
        """Decode chars from HDF5 file.

        Arguments:
            encoded_chars: Chars to decode
        Returns:
            Decoded chars
        """
        return [assignment.decode("utf-8") for assignment in encoded_chars]

    @classmethod
    def encode_chars(cls, chars: Iterable[str]) -> list[bytes]:
        """Encode chars for HDF5 file.

        Arguments:
            chars: Chars to encode
        Returns:
            Encoded chars
        """
        return [assignment.encode("utf-8") for assignment in chars]

    @classmethod
    def load_hdf5(cls, infile: Path) -> tuple[dict[bytes, str], list[bytes]]:
        """Load char arrays and assignments from an HDF5 file.

        Arguments:
            infile: Path to HDF5 file
        Returns:
            Assigned and unassigned char bytes
        """
        assigned, assignments = [], []
        unassigned = []

        with h5py.File(infile, "r") as h5_file:
            if "assigned" in h5_file and "assignments" in h5_file:
                assigned = map(cls.array_to_bytes, np.array(h5_file["assigned"]))
                assignments = cls.decode_chars(h5_file["assignments"])

            if "unassigned" in h5_file:
                unassigned = map(cls.array_to_bytes, np.array(h5_file["unassigned"]))

        return dict(zip(assigned, assignments)), list(unassigned)

    @classmethod
    def save_hdf5(
        cls,
        assigned_char_bytes: dict[bytes, str],
        unassigned_char_bytes: list[bytes],
        outfile: Path,
    ) -> None:
        """Save char bytes and assignments to an HDF5 file.

        Arguments:
            assigned_char_bytes: Assigned images
            unassigned_char_bytes: Unassigned images
            outfile: Path to HDF5 outfile
        """
        with h5py.File(outfile, "w") as h5_file:
            if "assigned" in h5_file:
                del h5_file["assigned"]
            if "assignments" in h5_file:
                del h5_file["assignments"]
            if len(assigned_char_bytes) > 0:
                h5_file.create_dataset(
                    f"assigned",
                    data=np.array(
                        list(map(cls.bytes_to_array, assigned_char_bytes.keys()))
                    ),
                    dtype=np.uint8,
                    chunks=True,
                    compression="gzip",
                )
                h5_file.create_dataset(
                    f"assignments",
                    data=cls.encode_chars(assigned_char_bytes.values()),
                    dtype="S4",
                    chunks=True,
                    compression="gzip",
                )

            if "unassigned" in h5_file:
                del h5_file["unassigned"]
            if len(unassigned_char_bytes) > 0:
                h5_file.create_dataset(
                    f"unassigned",
                    data=np.array(list(map(cls.bytes_to_array, unassigned_char_bytes))),
                    dtype=np.uint8,
                    chunks=True,
                    compression="gzip",
                )
