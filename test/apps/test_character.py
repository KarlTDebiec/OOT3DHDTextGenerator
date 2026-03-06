#  Copyright 2020-2026 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Tests for character view model."""

from __future__ import annotations

import numpy as np

from oot3dhdtextgenerator.apps.char_assigner.character import Character


def test_image_property_with_slots_is_cached() -> None:
    """Test image property works with dataclass slots and caches output."""
    character = Character(id=0, array=np.zeros((16, 16), dtype=np.uint8))
    image_1 = character.image
    image_2 = character.image

    assert image_1.startswith("data:image/png;base64,")
    assert image_1 == image_2
