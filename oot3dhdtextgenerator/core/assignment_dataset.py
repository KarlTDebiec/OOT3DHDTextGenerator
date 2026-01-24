#  Copyright 2020-2026 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Assignment project."""

from __future__ import annotations

from collections.abc import Iterable
from logging import debug, info
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from torch import Tensor
from torchvision.datasets import VisionDataset
from torchvision.transforms import Compose, Normalize, ToTensor

from oot3dhdtextgenerator.common.validation import val_input_path


class AssignmentDataset(VisionDataset):
    """Assignment project."""

    multi_char_array_shapes = ((128, 256), (128, 512), (256, 256))
    char_array_shape = (16, 16)

    def __init__(self, input_path: Path) -> None:
        """Initialize."""
        input_path = val_input_path(input_path, must_exist=False)

        transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
        super().__init__(str(input_path.parent), transform=transform)

        assigned_char_bytes: dict[bytes, str] = {}
        unassigned_char_bytes: list[bytes] = []
        if input_path.exists():
            assigned_char_bytes, unassigned_char_bytes = self.load_hdf5(input_path)

        self.assigned_char_bytes = assigned_char_bytes
        """Dictionary whose keys are char bytes and values are char strs"""
        self.unassigned_char_bytes = unassigned_char_bytes
        """List of unassigned char bytes"""

    def __getitem__(self, index: int) -> Tensor:
        """Get unassigned char Tensor at index."""
        char_bytes = self.unassigned_char_bytes[index]
        char_array = self.bytes_to_array(char_bytes)
        char_image = Image.fromarray(char_array)
        char_tensor: Tensor = self.transform(char_image)

        return char_tensor

    def __len__(self) -> int:
        """Number of images in the dataset."""
        return len(self.unassigned_char_bytes)

    def __str__(self) -> str:
        """String representation."""
        return f"<{self.__class__.__name__}>"

    def assign(self, char_array: np.ndarray, char: str | None) -> None:
        """Assign char to char array.

        Arguments:
            char_array: Char array to assign
            char: Char to assign
        """
        char_bytes = self.array_to_bytes(char_array)

        if char is None:
            self.assigned_char_bytes.pop(char_bytes)
            self.unassigned_char_bytes.append(char_bytes)
            info(f"Unassigned {char}")
            return

        if len(char) != 1:
            raise ValueError(f"Character {char} must be a single character or None")

        if char_bytes in self.unassigned_char_bytes:
            self.unassigned_char_bytes.pop(self.unassigned_char_bytes.index(char_bytes))
            info(f"Assigned {char}")
        else:
            info(f"Reassigned {self.assigned_char_bytes[char_bytes]} to {char}")
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
        for x in range(0, multi_char_array.shape[0], self.char_array_shape[0]):
            for y in range(0, multi_char_array.shape[1], self.char_array_shape[1]):
                char_array = multi_char_array[
                    x : x + self.char_array_shape[0], y : y + self.char_array_shape[1]
                ]
                if char_array.sum() == 0:
                    chars.append(" ")
                    continue
                char = self.get_char_for_char_array(char_array)
                if char is None:
                    all_chars_assigned = False
                else:
                    chars.append(char)

        # Return assignments, if all characters are assigned, or None otherwise
        if all_chars_assigned:
            return "".join(chars).rstrip()
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
            debug(f"Assigned character {char} retrieved")
            return char

        if char_bytes not in self.unassigned_char_bytes:
            self.unassigned_char_bytes.append(char_bytes)
            debug(
                f"Unassigned character added, {len(self.unassigned_char_bytes)} total"
            )

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
    def bytes_to_array(cls, char_bytes: bytes) -> np.ndarray:
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
    def load_hdf5(cls, input_path: Path) -> tuple[dict[bytes, str], list[bytes]]:
        """Load char arrays and assignments from an HDF5 file.

        Arguments:
            input_path: Path to HDF5 file
        Returns:
            Assigned and unassigned char bytes
        """
        assigned: Iterable[bytes] = []
        assignments: list[str] = []
        unassigned: Iterable[bytes] = []

        with h5py.File(input_path, "r") as h5_file:
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
        output_path: Path,
    ) -> None:
        """Save char bytes and assignments to an HDF5 file.

        Arguments:
            assigned_char_bytes: Assigned images
            unassigned_char_bytes: Unassigned images
            output_path: Path to HDF5 outfile
        """
        with h5py.File(output_path, "w") as h5_file:
            if "assigned" in h5_file:
                del h5_file["assigned"]
            if "assignments" in h5_file:
                del h5_file["assignments"]
            if len(assigned_char_bytes) > 0:
                sorted_assigned_char_bytes = dict(
                    sorted(assigned_char_bytes.items(), key=lambda item: item[1])
                )
                h5_file.create_dataset(
                    "assigned",
                    data=np.array(
                        list(map(cls.bytes_to_array, sorted_assigned_char_bytes.keys()))
                    ),
                    dtype=np.uint8,
                    chunks=True,
                    compression="gzip",
                )
                h5_file.create_dataset(
                    "assignments",
                    data=cls.encode_chars(sorted_assigned_char_bytes.values()),
                    dtype="S4",
                    chunks=True,
                    compression="gzip",
                )

            if "unassigned" in h5_file:
                del h5_file["unassigned"]
            if len(unassigned_char_bytes) > 0:
                unassigned_arrays = list(map(cls.bytes_to_array, unassigned_char_bytes))
                sorted_unassigned_arrays = sorted(
                    unassigned_arrays, key=lambda x: x.sum()
                )
                h5_file.create_dataset(
                    "unassigned",
                    data=np.array(sorted_unassigned_arrays),
                    dtype=np.uint8,
                    chunks=True,
                    compression="gzip",
                )
        info(f"Saved AssignmentDataset to {output_path}")
