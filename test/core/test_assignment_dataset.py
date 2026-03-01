#  Copyright 2020-2026 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Tests for assignment dataset CSV persistence."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from oot3dhdtextgenerator.core import AssignmentDataset
from oot3dhdtextgenerator.core.assignment_dataset_helpers import (
    array_to_raw_base64_png,
    raw_base64_png_to_array,
)


def _build_char_array(seed: int) -> np.ndarray:
    """Build a deterministic 16x16 character array.

    Arguments:
        seed: random seed
    Returns:
        deterministic character array
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, AssignmentDataset.char_array_shape, dtype=np.uint8)


def test_csv_bootstrap_empty_dataset(tmp_path: Path) -> None:
    """Bootstraps from missing CSV files and persists empty CSV files.

    Arguments:
        tmp_path: temporary test path fixture
    """
    assignment_dir_path = tmp_path / "oot3d"
    dataset = AssignmentDataset(assignment_dir_path)

    assert dataset.assigned_char_bytes == {}
    assert dataset.unassigned_char_bytes == []
    assert not dataset.assigned_csv_path.exists()
    assert not dataset.unassigned_csv_path.exists()

    dataset.save_csv(
        dataset.assigned_char_bytes,
        dataset.unassigned_char_bytes,
        dataset.assigned_csv_path,
        dataset.unassigned_csv_path,
    )

    with dataset.assigned_csv_path.open("r", encoding="utf-8", newline="") as infile:
        assigned_rows = list(csv.DictReader(infile))
    with dataset.unassigned_csv_path.open("r", encoding="utf-8", newline="") as infile:
        unassigned_rows = list(csv.DictReader(infile))

    assert assigned_rows == []
    assert unassigned_rows == []


def test_csv_round_trip_integrity(tmp_path: Path) -> None:
    """Round trips assigned and unassigned arrays through CSV.

    Arguments:
        tmp_path: temporary test path fixture
    """
    assigned_arrays = [
        _build_char_array(1),
        _build_char_array(2),
    ]
    unassigned_arrays = [
        _build_char_array(3),
        _build_char_array(4),
    ]
    assigned_char_bytes = {
        AssignmentDataset.array_to_bytes(assigned_arrays[0]): "你",
        AssignmentDataset.array_to_bytes(assigned_arrays[1]): "好",
    }
    unassigned_char_bytes = [
        AssignmentDataset.array_to_bytes(unassigned_arrays[0]),
        AssignmentDataset.array_to_bytes(unassigned_arrays[1]),
    ]

    assigned_csv_path = tmp_path / "oot3d" / "assigned.csv"
    unassigned_csv_path = tmp_path / "oot3d" / "unassigned.csv"

    AssignmentDataset.save_csv(
        assigned_char_bytes,
        unassigned_char_bytes,
        assigned_csv_path,
        unassigned_csv_path,
    )

    loaded_assigned, loaded_unassigned = AssignmentDataset.load_csv(
        assigned_csv_path,
        unassigned_csv_path,
    )

    assert loaded_assigned == assigned_char_bytes

    expected_unassigned = sorted(
        unassigned_char_bytes,
        key=lambda char_bytes: array_to_raw_base64_png(
            AssignmentDataset.bytes_to_array(char_bytes)
        ),
    )
    assert loaded_unassigned == expected_unassigned

    encoded = array_to_raw_base64_png(assigned_arrays[0])
    assert not encoded.startswith("data:image/png;base64,")
    decoded = raw_base64_png_to_array(encoded)
    np.testing.assert_array_equal(decoded, assigned_arrays[0])


def test_csv_sorting_behavior(tmp_path: Path) -> None:
    """Persists deterministic ordering for assigned and unassigned CSV rows.

    Arguments:
        tmp_path: temporary test path fixture
    """
    assigned_arrays = [
        _build_char_array(7),
        _build_char_array(8),
        _build_char_array(9),
    ]
    assigned_char_bytes = {
        AssignmentDataset.array_to_bytes(assigned_arrays[0]): "漢",
        AssignmentDataset.array_to_bytes(assigned_arrays[1]): "一",
        AssignmentDataset.array_to_bytes(assigned_arrays[2]): "乙",
    }

    unassigned_arrays = [
        _build_char_array(10),
        _build_char_array(11),
        _build_char_array(12),
    ]
    unassigned_char_bytes = [
        AssignmentDataset.array_to_bytes(array) for array in unassigned_arrays
    ]

    assigned_csv_path = tmp_path / "oot3d" / "assigned.csv"
    unassigned_csv_path = tmp_path / "oot3d" / "unassigned.csv"

    AssignmentDataset.save_csv(
        assigned_char_bytes,
        unassigned_char_bytes,
        assigned_csv_path,
        unassigned_csv_path,
    )

    with assigned_csv_path.open("r", encoding="utf-8", newline="") as infile:
        assigned_rows = list(csv.DictReader(infile))
    with unassigned_csv_path.open("r", encoding="utf-8", newline="") as infile:
        unassigned_rows = list(csv.DictReader(infile))

    assert [row["character"] for row in assigned_rows] == sorted(["漢", "一", "乙"])

    expected_unassigned_base64 = sorted(
        [array_to_raw_base64_png(array) for array in unassigned_arrays]
    )
    assert [row["png_base64"] for row in unassigned_rows] == expected_unassigned_base64


def test_load_csv_rejects_missing_assigned_header(tmp_path: Path) -> None:
    """Rejects assigned CSV files missing required headers.

    Arguments:
        tmp_path: temporary test path fixture
    """
    assigned_csv_path = tmp_path / "oot3d" / "assigned.csv"
    assigned_csv_path.parent.mkdir(parents=True, exist_ok=True)
    assigned_csv_path.write_text("character\n你\n", encoding="utf-8")
    unassigned_csv_path = tmp_path / "oot3d" / "unassigned.csv"
    unassigned_csv_path.write_text("png_base64\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required CSV columns"):
        AssignmentDataset.load_csv(assigned_csv_path, unassigned_csv_path)


def test_load_csv_rejects_missing_unassigned_header(tmp_path: Path) -> None:
    """Rejects unassigned CSV files missing required headers.

    Arguments:
        tmp_path: temporary test path fixture
    """
    assigned_csv_path = tmp_path / "oot3d" / "assigned.csv"
    assigned_csv_path.parent.mkdir(parents=True, exist_ok=True)
    assigned_csv_path.write_text("character,png_base64\n", encoding="utf-8")
    unassigned_csv_path = tmp_path / "oot3d" / "unassigned.csv"
    unassigned_csv_path.write_text("character\n你\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required CSV columns"):
        AssignmentDataset.load_csv(assigned_csv_path, unassigned_csv_path)
