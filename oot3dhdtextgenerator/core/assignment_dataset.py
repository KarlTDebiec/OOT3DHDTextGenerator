#  Copyright 2020-2026 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Assignment project."""

from __future__ import annotations

from csv import DictReader, DictWriter
from logging import debug, info
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.transforms import Compose, Normalize, ToTensor

from oot3dhdtextgenerator.common.validation import val_output_dir_path
from oot3dhdtextgenerator.core.assignment_dataset_helpers import (
    array_to_raw_base64_png,
    raw_base64_png_to_array,
)
from oot3dhdtextgenerator.data import (
    oot3d_assigned_csv_path,
    oot3d_unassigned_csv_path,
)

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor


class AssignmentDataset(VisionDataset):
    """Assignment project."""

    multi_char_array_shapes = ((128, 256), (128, 512), (256, 256))
    char_array_shape = (16, 16)

    def __init__(self, assignment_dir_path: Path | str) -> None:
        """Initialize.

        Arguments:
            assignment_dir_path: path to assignment CSV directory
        """
        self.assignment_dir_path = val_output_dir_path(assignment_dir_path)
        self.assigned_csv_path = self.assignment_dir_path / "assigned.csv"
        self.unassigned_csv_path = self.assignment_dir_path / "unassigned.csv"

        transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
        super().__init__(str(self.assignment_dir_path), transform=transform)

        assigned_char_bytes: dict[bytes, str] = {}
        unassigned_char_bytes: list[bytes] = []
        assigned_csv_exists = self.assigned_csv_path.exists()
        unassigned_csv_exists = self.unassigned_csv_path.exists()
        if assigned_csv_exists or unassigned_csv_exists:
            assigned_char_bytes, unassigned_char_bytes = self.load_csv(
                self.assigned_csv_path,
                self.unassigned_csv_path,
            )

        self.assigned_char_bytes = assigned_char_bytes
        """Dictionary whose keys are char bytes and values are char strs"""
        self.unassigned_char_bytes = unassigned_char_bytes
        """List of unassigned char bytes"""

    def __getitem__(self, index: int) -> Tensor:
        """Get unassigned char Tensor at index."""
        if self.transform is None:
            raise RuntimeError("AssignmentDataset transform is not configured")
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
            char_array: char array to assign
            char: char to assign
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
            multi_char_array: multi-char array whose chars to retrieve
        Returns:
            chars, or None if not all chars are assigned
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
            char_array: char array whose char to retrieve
        Returns:
            char, or None if not assigned
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
            char_array: char array
        Returns:
            char bytes
        """
        return char_array.tobytes()

    @classmethod
    def bytes_to_array(cls, char_bytes: bytes) -> np.ndarray:
        """Convert char bytes to char array.

        Arguments:
            char_bytes: char bytes
        Returns:
            char array
        """
        return np.frombuffer(char_bytes, dtype=np.uint8).reshape(cls.char_array_shape)

    @classmethod
    def load_csv(
        cls,
        assigned_csv_path: Path = oot3d_assigned_csv_path,
        unassigned_csv_path: Path = oot3d_unassigned_csv_path,
    ) -> tuple[dict[bytes, str], list[bytes]]:
        """Load char arrays and assignments from CSV files.

        Arguments:
            assigned_csv_path: path to assigned CSV file
            unassigned_csv_path: path to unassigned CSV file
        Returns:
            assigned and unassigned char bytes
        """
        assigned: dict[bytes, str] = {}
        if assigned_csv_path.exists():
            with assigned_csv_path.open("r", encoding="utf-8", newline="") as infile:
                reader = DictReader(infile)
                fieldnames = set(reader.fieldnames or [])
                required_fieldnames = {"character", "png_base64"}
                missing_fieldnames = required_fieldnames - fieldnames
                if missing_fieldnames:
                    raise ValueError(
                        "Missing required CSV columns in "
                        f"{assigned_csv_path}: {sorted(missing_fieldnames)}"
                    )

                for row_number, row in enumerate(reader, start=2):
                    character = row.get("character")
                    png_base64 = row.get("png_base64")
                    if character is None or png_base64 is None:
                        raise ValueError(
                            "Malformed assigned CSV row "
                            f"{row_number} in {assigned_csv_path}"
                        )
                    char_array = raw_base64_png_to_array(row["png_base64"])
                    if char_array.shape != cls.char_array_shape:
                        raise ValueError(
                            "Invalid array shape "
                            f"{char_array.shape}, expected {cls.char_array_shape}"
                        )
                    char_bytes = cls.array_to_bytes(char_array)
                    assigned[char_bytes] = character

        unassigned: list[bytes] = []
        if unassigned_csv_path.exists():
            with unassigned_csv_path.open("r", encoding="utf-8", newline="") as infile:
                reader = DictReader(infile)
                fieldnames = set(reader.fieldnames or [])
                required_fieldnames = {"png_base64"}
                missing_fieldnames = required_fieldnames - fieldnames
                if missing_fieldnames:
                    raise ValueError(
                        "Missing required CSV columns in "
                        f"{unassigned_csv_path}: {sorted(missing_fieldnames)}"
                    )

                for row_number, row in enumerate(reader, start=2):
                    png_base64 = row.get("png_base64")
                    if png_base64 is None:
                        raise ValueError(
                            "Malformed unassigned CSV row "
                            f"{row_number} in {unassigned_csv_path}"
                        )
                    char_array = raw_base64_png_to_array(row["png_base64"])
                    if char_array.shape != cls.char_array_shape:
                        raise ValueError(
                            "Invalid array shape "
                            f"{char_array.shape}, expected {cls.char_array_shape}"
                        )
                    char_bytes = cls.array_to_bytes(char_array)
                    unassigned.append(char_bytes)

        return assigned, unassigned

    @classmethod
    def save_csv(
        cls,
        assigned_char_bytes: dict[bytes, str],
        unassigned_char_bytes: list[bytes],
        assigned_csv_path: Path = oot3d_assigned_csv_path,
        unassigned_csv_path: Path = oot3d_unassigned_csv_path,
    ) -> None:
        """Save char bytes and assignments to CSV files.

        Arguments:
            assigned_char_bytes: assigned images
            unassigned_char_bytes: unassigned images
            assigned_csv_path: path to assigned CSV output file
            unassigned_csv_path: path to unassigned CSV output file
        """
        assigned_csv_path.parent.mkdir(parents=True, exist_ok=True)
        unassigned_csv_path.parent.mkdir(parents=True, exist_ok=True)

        sorted_assigned_items = sorted(
            assigned_char_bytes.items(), key=lambda item: item[1]
        )
        with assigned_csv_path.open("w", encoding="utf-8", newline="") as outfile:
            writer = DictWriter(outfile, fieldnames=["character", "png_base64"])
            writer.writeheader()
            for char_bytes, char in sorted_assigned_items:
                char_array = cls.bytes_to_array(char_bytes)
                if char_array.shape != cls.char_array_shape:
                    raise ValueError(
                        "Invalid array shape "
                        f"{char_array.shape}, expected {cls.char_array_shape}"
                    )
                writer.writerow(
                    {
                        "character": char,
                        "png_base64": array_to_raw_base64_png(char_array),
                    }
                )

        def _encode_unassigned(char_bytes: bytes) -> str:
            char_array = cls.bytes_to_array(char_bytes)
            if char_array.shape != cls.char_array_shape:
                raise ValueError(
                    "Invalid array shape "
                    f"{char_array.shape}, expected {cls.char_array_shape}"
                )
            return array_to_raw_base64_png(char_array)

        sorted_unassigned_char_bytes = sorted(
            unassigned_char_bytes,
            key=_encode_unassigned,
        )
        with unassigned_csv_path.open("w", encoding="utf-8", newline="") as outfile:
            writer = DictWriter(outfile, fieldnames=["png_base64"])
            writer.writeheader()
            for char_bytes in sorted_unassigned_char_bytes:
                writer.writerow(
                    {
                        "png_base64": _encode_unassigned(char_bytes),
                    }
                )

        info(
            "Saved AssignmentDataset to %s and %s",
            assigned_csv_path,
            unassigned_csv_path,
        )
