#  Copyright 2020-2026 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Tests for training dataset CSV and NPY persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from oot3dhdtextgenerator.core import TrainingDataset


def _build_specifications() -> np.ndarray:
    """Build deterministic specification rows.

    Returns:
        deterministic specifications array
    """
    return np.array(
        [
            ("你", "font_a.ttf", 14, -1, 0, 255, 0.0),
            ("好", "font_b.ttf", 16, 1, -1, 255, 5.0),
        ],
        dtype=TrainingDataset.specification_dtypes,
    )


def test_save_and_load_dataset_round_trip(tmp_path: Path) -> None:
    """Round trips training dataset through CSV and NPY files.

    Arguments:
        tmp_path: temporary test path fixture
    """
    images = np.arange(2 * 16 * 16, dtype=np.uint8).reshape(2, 16, 16)
    specifications = _build_specifications()
    output_dir_path = tmp_path / "train"

    TrainingDataset.save_dataset(images, specifications, output_dir_path)

    loaded_images, loaded_specifications = TrainingDataset.load_dataset(output_dir_path)

    np.testing.assert_array_equal(loaded_images, images)
    np.testing.assert_array_equal(loaded_specifications, specifications)


def test_load_dataset_rejects_image_specification_length_mismatch(
    tmp_path: Path,
) -> None:
    """Rejects datasets where image count and index row count diverge.

    Arguments:
        tmp_path: temporary test path fixture
    """
    output_dir_path = tmp_path / "train"
    output_dir_path.mkdir(parents=True)
    images = np.arange(16 * 16, dtype=np.uint8).reshape(1, 16, 16)
    np.save(output_dir_path / TrainingDataset.images_npy_file_name, images)
    (output_dir_path / TrainingDataset.specifications_csv_file_name).write_text(
        "character,font,size,x_offset,y_offset,fill,rotation\n"
        "你,font_a.ttf,14,0,0,255,0\n"
        "好,font_b.ttf,14,0,0,255,0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Image/specification length mismatch"):
        TrainingDataset.load_dataset(output_dir_path)


def test_load_dataset_rejects_invalid_character_width(tmp_path: Path) -> None:
    """Rejects specifications with non-single-character values.

    Arguments:
        tmp_path: temporary test path fixture
    """
    output_dir_path = tmp_path / "train"
    output_dir_path.mkdir(parents=True)
    images = np.arange(16 * 16, dtype=np.uint8).reshape(1, 16, 16)
    np.save(output_dir_path / TrainingDataset.images_npy_file_name, images)
    (output_dir_path / TrainingDataset.specifications_csv_file_name).write_text(
        "character,font,size,x_offset,y_offset,fill,rotation\n"
        "你好,font_a.ttf,14,0,0,255,0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid character in specifications CSV"):
        TrainingDataset.load_dataset(output_dir_path)


def test_load_dataset_rejects_invalid_image_shape(tmp_path: Path) -> None:
    """Rejects invalid image tensor shape in images NPY file.

    Arguments:
        tmp_path: temporary test path fixture
    """
    output_dir_path = tmp_path / "train"
    output_dir_path.mkdir(parents=True)
    images = np.arange(16 * 16, dtype=np.uint8).reshape(16, 16)
    np.save(output_dir_path / TrainingDataset.images_npy_file_name, images)
    (output_dir_path / TrainingDataset.specifications_csv_file_name).write_text(
        "character,font,size,x_offset,y_offset,fill,rotation\n"
        "你,font_a.ttf,14,0,0,255,0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid images array shape"):
        TrainingDataset.load_dataset(output_dir_path)
