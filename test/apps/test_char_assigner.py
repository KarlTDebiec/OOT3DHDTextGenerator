#  Copyright 2020-2026 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Tests for character assigner application."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from oot3dhdtextgenerator.apps.char_assigner.char_assigner import CharAssigner
from oot3dhdtextgenerator.data import characters as known_characters

if TYPE_CHECKING:
    from oot3dhdtextgenerator.core import AssignmentDataset


def test_get_characters_predictions_use_label_list() -> None:
    """Test predictions are mapped from model labels, not output list length."""

    class FakeDataset:
        """Minimal dataset stub for char assigner tests."""

        unassigned_char_bytes = [bytes(range(16 * 16))]
        assigned_char_bytes: dict[bytes, str] = {}

        @staticmethod
        def bytes_to_array(char_bytes: bytes) -> np.ndarray:
            """Convert bytes to a 16x16 array."""
            return np.frombuffer(char_bytes, dtype=np.uint8).reshape((16, 16))

    score = np.zeros(10, dtype=np.float32)
    score[9] = 1.0
    dataset = cast("AssignmentDataset", FakeDataset())
    characters = CharAssigner.get_characters(
        dataset,
        score[None, :],
        {},
    )

    assert len(characters) == 1
    assert characters[0].predictions is not None
    assert characters[0].predictions[0] == known_characters[9]


def test_get_characters_assigned_rows_are_sorted_and_predicted() -> None:
    """Test assigned rows are sorted by label order and include predictions."""

    class FakeDataset:
        """Minimal dataset stub for char assigner tests."""

        unassigned_char_bytes = [bytes([1] * (16 * 16))]
        assigned_char_bytes: dict[bytes, str] = {
            bytes([3] * (16 * 16)): known_characters[5],
            bytes([2] * (16 * 16)): known_characters[1],
        }

        @staticmethod
        def bytes_to_array(char_bytes: bytes) -> np.ndarray:
            """Convert bytes to a 16x16 array."""
            return np.frombuffer(char_bytes, dtype=np.uint8).reshape((16, 16))

    unassigned_score = np.zeros(10, dtype=np.float32)
    unassigned_score[3] = 1.0
    assigned_score_1 = np.zeros(10, dtype=np.float32)
    assigned_score_1[8] = 1.0
    assigned_score_2 = np.zeros(10, dtype=np.float32)
    assigned_score_2[7] = 1.0

    dataset = cast("AssignmentDataset", FakeDataset())
    characters = CharAssigner.get_characters(
        dataset,
        unassigned_score[None, :],
        {
            bytes([3] * (16 * 16)): assigned_score_1,
            bytes([2] * (16 * 16)): assigned_score_2,
        },
    )

    assert len(characters) == 3
    assert characters[0].assignment is None
    assert characters[0].predictions is not None
    assert characters[0].predictions[0] == known_characters[3]

    # Assigned rows should be sorted by known character order: index 1 before index 5.
    assert characters[1].assignment == known_characters[1]
    assert characters[2].assignment == known_characters[5]
    assert characters[1].predictions is not None
    assert characters[1].predictions[0] == known_characters[7]
    assert characters[2].predictions is not None
    assert characters[2].predictions[0] == known_characters[8]
