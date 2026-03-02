#  Copyright 2020-2026 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Training dataset."""

from __future__ import annotations

from csv import DictReader, DictWriter
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset

from oot3dhdtextgenerator.common.validation import (
    val_input_dir_path,
    val_output_dir_path,
)
from oot3dhdtextgenerator.data import character_to_index

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path


class TrainingDataset(VisionDataset):
    """Training dataset."""

    images_npy_file_name = "images.npy"
    specifications_csv_file_name = "specifications.csv"
    char_array_shape = (16, 16)

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

    @classmethod
    def specification_fieldnames(cls) -> tuple[str, ...]:
        """Ordered specification field names."""
        return tuple(name for name, _ in cls.specification_dtypes)

    def __init__(
        self,
        input_dir_path: Path | str,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        """Initialize.

        Arguments:
            input_dir_path: path to dataset directory with images.npy and
              specifications.csv
            transform: transform to apply to images
            target_transform: transform to apply to targets
        """
        input_dir_path = val_input_dir_path(input_dir_path)
        super().__init__(
            str(input_dir_path),
            transform=transform,
            target_transform=target_transform,
        )
        self.images, self.specifications = self.load_dataset(input_dir_path)

    def __getitem__(self, index: int) -> tuple[object, int]:
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
    def _validate_required_columns(
        cls,
        fieldnames: Sequence[str] | None,
        required_fieldnames: set[str],
        csv_path: Path,
    ) -> None:
        """Validate required CSV column names.

        Arguments:
            fieldnames: CSV field names from DictReader
            required_fieldnames: names that must exist in the CSV
            csv_path: CSV file path for error context
        """
        missing_fieldnames = required_fieldnames - set(fieldnames or [])
        if missing_fieldnames:
            raise ValueError(
                f"Missing required CSV columns in {csv_path}: "
                f"{sorted(missing_fieldnames)}"
            )

    @classmethod
    def _parse_specification_row(
        cls, row: dict[str, str], csv_path: Path, row_number: int
    ) -> tuple[str, str, int, int, int, int, float]:
        """Parse one specification CSV row.

        Arguments:
            row: CSV row dictionary
            csv_path: CSV file path for error context
            row_number: one-based CSV row number for error context
        Returns:
            parsed specification tuple
        """
        character = row.get("character")
        font = row.get("font")
        size = row.get("size")
        x_offset = row.get("x_offset")
        y_offset = row.get("y_offset")
        fill = row.get("fill")
        rotation = row.get("rotation")

        if (
            character is None
            or font is None
            or size is None
            or x_offset is None
            or y_offset is None
            or fill is None
            or rotation is None
        ):
            raise ValueError(f"Malformed specification row {row_number} in {csv_path}")

        if len(character) != 1:
            raise ValueError(
                "Invalid character in specifications CSV at "
                f"{csv_path}:{row_number}: {character!r}"
            )

        try:
            return (
                character,
                font,
                int(size),
                int(x_offset),
                int(y_offset),
                int(fill),
                float(rotation),
            )
        except ValueError as exc:
            raise ValueError(
                "Invalid numeric value in specifications CSV at "
                f"{csv_path}:{row_number}"
            ) from exc

    @classmethod
    def load_dataset(
        cls,
        input_dir_path: Path | str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load images and specifications from dataset directory.

        Arguments:
            input_dir_path: path to dataset directory
        Returns:
            images and specifications
        Raises:
            FileNotFoundError: If required files are missing
            ValueError: If files are malformed
        """
        input_dir_path = val_input_dir_path(input_dir_path)
        images_npy_path = input_dir_path / cls.images_npy_file_name
        specifications_csv_path = input_dir_path / cls.specifications_csv_file_name

        if not images_npy_path.exists() or not images_npy_path.is_file():
            raise FileNotFoundError(f"Input file {images_npy_path} does not exist")
        if (
            not specifications_csv_path.exists()
            or not specifications_csv_path.is_file()
        ):
            raise FileNotFoundError(
                f"Input file {specifications_csv_path} does not exist"
            )

        images = np.load(images_npy_path)
        if images.ndim != 3 or images.shape[1:] != cls.char_array_shape:
            raise ValueError(
                f"Invalid images array shape {images.shape}, "
                f"expected (n, {cls.char_array_shape[0]}, {cls.char_array_shape[1]})"
            )
        if images.dtype != np.uint8:
            raise ValueError(f"Invalid images dtype {images.dtype}, expected uint8")

        with specifications_csv_path.open("r", encoding="utf-8", newline="") as infile:
            reader = DictReader(infile)
            cls._validate_required_columns(
                reader.fieldnames,
                set(cls.specification_fieldnames()),
                specifications_csv_path,
            )
            specification_rows = [
                cls._parse_specification_row(row, specifications_csv_path, row_number)
                for row_number, row in enumerate(reader, start=2)
            ]

        specifications = np.array(specification_rows, dtype=cls.specification_dtypes)
        if len(specifications) != images.shape[0]:
            raise ValueError(
                "Image/specification length mismatch: "
                f"{images.shape[0]} images vs {len(specifications)} rows"
            )

        return images, specifications

    @classmethod
    def save_dataset(
        cls,
        images: np.ndarray,
        specifications: np.ndarray,
        output_dir_path: Path | str,
    ) -> None:
        """Save images and specifications to dataset directory.

        Arguments:
            images: image arrays with shape (n, 16, 16)
            specifications: image specifications
            output_dir_path: output directory path
        """
        output_dir_path = val_output_dir_path(output_dir_path)
        images_npy_path = output_dir_path / cls.images_npy_file_name
        specifications_csv_path = output_dir_path / cls.specifications_csv_file_name

        if images.ndim != 3 or images.shape[1:] != cls.char_array_shape:
            raise ValueError(
                f"Invalid images array shape {images.shape}, "
                f"expected (n, {cls.char_array_shape[0]}, {cls.char_array_shape[1]})"
            )
        if images.dtype != np.uint8:
            raise ValueError(f"Invalid images dtype {images.dtype}, expected uint8")

        if list(specifications.dtype.names or []) != list(
            cls.specification_fieldnames()
        ):
            raise ValueError(
                "Invalid specification columns: "
                f"{list(specifications.dtype.names or [])}, "
                f"expected {list(cls.specification_fieldnames())}"
            )

        if len(specifications) != images.shape[0]:
            raise ValueError(
                "Image/specification length mismatch: "
                f"{images.shape[0]} images vs {len(specifications)} rows"
            )

        np.save(images_npy_path, images)

        with specifications_csv_path.open("w", encoding="utf-8", newline="") as outfile:
            writer = DictWriter(
                outfile, fieldnames=list(cls.specification_fieldnames())
            )
            writer.writeheader()
            for row in specifications:
                writer.writerow(
                    {
                        "character": str(row["character"]),
                        "font": str(row["font"]),
                        "size": int(row["size"]),
                        "x_offset": int(row["x_offset"]),
                        "y_offset": int(row["y_offset"]),
                        "fill": int(row["fill"]),
                        "rotation": float(row["rotation"]),
                    }
                )
