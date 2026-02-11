#  Copyright 2020-2026 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Data.

This module may import from: common

Hierarchy within module:
* characters (data only, no submodule hierarchy)
"""

from __future__ import annotations

import pandas as pd

from oot3dhdtextgenerator.common import package_root

hanzi_frequency = pd.read_csv(
    f"{package_root}/data/characters.txt",
    sep="\t",
    names=["character", "frequency", "cumulative frequency"],
)

characters = hanzi_frequency["character"].values.tolist()


def character_to_index(character: str) -> int:
    """Get index of a character in the ordered character list."""
    return characters.index(character)


__all__ = [
    "characters",
    "character_to_index",
    "hanzi_frequency",
]
