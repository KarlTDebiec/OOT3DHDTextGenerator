#  Copyright 2020-2026 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Tests for training dataset generator utility methods."""

from __future__ import annotations

import numpy as np
import pytest

from oot3dhdtextgenerator.utilities.training_dataset_generator import (
    TrainingDatasetGenerator,
)


def test_translate_array_no_wrap_translates_in_bounds() -> None:
    """Test in-bounds translation preserves source pixels without wrapping."""
    array = np.zeros((4, 4), dtype=np.uint8)
    array[1, 1] = 255

    translated = TrainingDatasetGenerator._translate_array(
        array, x_offset=1, y_offset=1
    )

    expected = np.zeros((4, 4), dtype=np.uint8)
    expected[2, 2] = 255
    np.testing.assert_array_equal(translated, expected)


def test_translate_array_no_wrap_raises_when_x_offset_off_canvas() -> None:
    """Test translation rejects x offsets that move all pixels off-canvas."""
    array = np.zeros((4, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="Offsets must keep at least one pixel"):
        TrainingDatasetGenerator._translate_array(array, x_offset=4, y_offset=0)


def test_translate_array_no_wrap_raises_when_y_offset_off_canvas() -> None:
    """Test translation rejects y offsets that move all pixels off-canvas."""
    array = np.zeros((4, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="Offsets must keep at least one pixel"):
        TrainingDatasetGenerator._translate_array(array, x_offset=0, y_offset=-4)


def test_generate_character_image_stroke_width_increases_pixel_mass() -> None:
    """Test nonzero stroke width produces a bolder image."""
    fonts = TrainingDatasetGenerator.get_default_font_paths()
    base = TrainingDatasetGenerator.generate_character_image(
        "你",
        font=fonts[0],
        size=16,
        fill=255,
        x_offset=0,
        y_offset=0,
        rotation=0,
        stroke_width=0,
    )
    bold = TrainingDatasetGenerator.generate_character_image(
        "你",
        font=fonts[0],
        size=16,
        fill=255,
        x_offset=0,
        y_offset=0,
        rotation=0,
        stroke_width=1,
    )

    assert int(bold.sum()) >= int(base.sum())
