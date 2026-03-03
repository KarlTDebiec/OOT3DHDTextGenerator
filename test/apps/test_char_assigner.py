#  Copyright 2020-2026 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Tests for character assigner application."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from oot3dhdtextgenerator.apps.char_assigner.char_assigner import CharAssigner
from oot3dhdtextgenerator.apps.char_assigner.character import Character
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


def test_get_characters_unassigned_rows_are_sorted_by_top_prediction() -> None:
    """Test unassigned rows are sorted by top prediction label order."""

    class FakeDataset:
        """Minimal dataset stub for char assigner tests."""

        unassigned_char_bytes = [bytes([1] * (16 * 16)), bytes([2] * (16 * 16))]
        assigned_char_bytes: dict[bytes, str] = {}

        @staticmethod
        def bytes_to_array(char_bytes: bytes) -> np.ndarray:
            """Convert bytes to a 16x16 array."""
            return np.frombuffer(char_bytes, dtype=np.uint8).reshape((16, 16))

    # First unassigned char predicts label 5, second predicts label 1.
    # Output should be sorted as label 1 then label 5, independent of input order.
    scores = np.zeros((2, 10), dtype=np.float32)
    scores[0, 5] = 1.0
    scores[1, 1] = 1.0

    dataset = cast("AssignmentDataset", FakeDataset())
    characters = CharAssigner.get_characters(dataset, scores, {})

    assert len(characters) == 2
    assert characters[0].assignment is None
    assert characters[1].assignment is None
    assert characters[0].predictions is not None
    assert characters[1].predictions is not None
    assert characters[0].predictions[0] == known_characters[1]
    assert characters[1].predictions[0] == known_characters[5]


def test_filter_characters_conflicts_only_assigned() -> None:
    """Test assigned conflicts-only filter shows duplicate assignments only."""
    characters = [
        Character(0, np.zeros((16, 16), dtype=np.uint8), None, [known_characters[2]]),
        Character(1, np.zeros((16, 16), dtype=np.uint8), known_characters[3], ["x"]),
        Character(2, np.zeros((16, 16), dtype=np.uint8), known_characters[4], ["x"]),
        Character(3, np.zeros((16, 16), dtype=np.uint8), known_characters[3], ["x"]),
    ]

    filtered = CharAssigner.filter_characters(
        characters,
        unassigned_filter="hidden",
        assigned_filter="conflicts_only",
    )

    assert [character.assignment for character in filtered] == [
        known_characters[3],
        known_characters[3],
    ]


def test_filter_characters_unassigned_hidden_assigned_hidden() -> None:
    """Test hidden filters suppress both unassigned and assigned rows."""
    characters = [
        Character(0, np.zeros((16, 16), dtype=np.uint8), None, [known_characters[2]]),
        Character(1, np.zeros((16, 16), dtype=np.uint8), known_characters[3], ["x"]),
    ]

    filtered = CharAssigner.filter_characters(
        characters,
        unassigned_filter="hidden",
        assigned_filter="hidden",
    )

    assert filtered == []


def test_filter_characters_unassigned_top_prediction_available_only() -> None:
    """Test top-prediction-available mode hides unavailable unassigned rows."""
    characters = [
        Character(0, np.zeros((16, 16), dtype=np.uint8), None, [known_characters[2]]),
        Character(1, np.zeros((16, 16), dtype=np.uint8), None, [known_characters[3]]),
        Character(2, np.zeros((16, 16), dtype=np.uint8), known_characters[2], ["x"]),
    ]

    filtered = CharAssigner.filter_characters(
        characters,
        unassigned_filter="top_prediction_available_only",
        assigned_filter="visible",
    )

    # Top prediction known_characters[2] is already assigned, so only [3] remains.
    assert [character.assignment for character in filtered] == [
        None,
        known_characters[2],
    ]
    assert filtered[0].predictions is not None
    assert filtered[0].predictions[0] == known_characters[3]


def test_normalize_prior_weight_percent_clamps_and_defaults() -> None:
    """Test prior weight normalization handles bounds and invalid values."""
    assert CharAssigner.normalize_prior_weight_percent(None) == 0.0
    assert CharAssigner.normalize_prior_weight_percent("abc") == 0.0
    assert CharAssigner.normalize_prior_weight_percent("-5") == 0.0
    assert CharAssigner.normalize_prior_weight_percent("25") == 25.0
    assert CharAssigner.normalize_prior_weight_percent("150") == 100.0


def test_blend_scores_extremes() -> None:
    """Test blend score uses model only at 0 and prior only at 1."""
    score = np.log(np.array([0.2, 0.8], dtype=np.float64))
    priors = np.array([0.9, 0.1], dtype=np.float64)

    model_only = CharAssigner.blend_scores(score, priors, 0.0)
    prior_only = CharAssigner.blend_scores(score, priors, 1.0)

    np.testing.assert_allclose(model_only, np.array([0.2, 0.8], dtype=np.float64))
    np.testing.assert_allclose(prior_only, np.array([0.9, 0.1], dtype=np.float64))


def test_normalize_exclude_assigned_from_predictions() -> None:
    """Test checkbox parsing for exclude-assigned option."""
    assert CharAssigner.normalize_exclude_assigned_from_predictions(None) is False
    assert CharAssigner.normalize_exclude_assigned_from_predictions("0") is False
    assert CharAssigner.normalize_exclude_assigned_from_predictions("off") is False
    assert CharAssigner.normalize_exclude_assigned_from_predictions("1") is True
    assert CharAssigner.normalize_exclude_assigned_from_predictions("on") is True


def test_update_character_predictions_excludes_assigned_labels() -> None:
    """Test assigned labels are excluded from prediction candidates when requested."""

    class FakeAssigner:
        """Minimal object compatible with CharAssigner.update_character_predictions."""

        n_chars = 3
        prior_probabilities = np.array([0.34, 0.33, 0.33], dtype=np.float64)
        characters = [
            Character(
                0,
                np.zeros((16, 16), dtype=np.uint8),
                known_characters[0],
                None,
                np.log(np.array([0.9, 0.05, 0.05], dtype=np.float64)),
            ),
            Character(
                1,
                np.zeros((16, 16), dtype=np.uint8),
                None,
                None,
                np.log(np.array([0.8, 0.15, 0.05], dtype=np.float64)),
            ),
        ]
        blend_scores = staticmethod(CharAssigner.blend_scores)

    assigner = cast("CharAssigner", FakeAssigner())

    CharAssigner.update_character_predictions(
        assigner,
        prior_weight=0.0,
        exclude_assigned_from_predictions=True,
    )
    assert assigner.characters[1].predictions is not None
    assert assigner.characters[1].predictions[0] == known_characters[1]

    CharAssigner.update_character_predictions(
        assigner,
        prior_weight=0.0,
        exclude_assigned_from_predictions=False,
    )
    assert assigner.characters[1].predictions is not None
    assert assigner.characters[1].predictions[0] == known_characters[0]


def test_filter_characters_assigned_top_prediction_mismatch_only() -> None:
    """Test mismatch-only assigned filter keeps only top-prediction mismatches."""
    characters = [
        Character(
            0,
            np.zeros((16, 16), dtype=np.uint8),
            known_characters[1],
            [known_characters[1]],
        ),
        Character(
            1,
            np.zeros((16, 16), dtype=np.uint8),
            known_characters[2],
            [known_characters[3]],
        ),
        Character(
            2,
            np.zeros((16, 16), dtype=np.uint8),
            known_characters[3],
            [known_characters[2]],
        ),
    ]

    filtered = CharAssigner.filter_characters(
        characters,
        unassigned_filter="hidden",
        assigned_filter="top_prediction_mismatch_only",
    )

    assert [character.assignment for character in filtered] == [
        known_characters[2],
        known_characters[3],
    ]
