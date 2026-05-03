#  Copyright 2020-2026 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Data.

This module may import from: common

Hierarchy within module:
* characters (data only, no submodule hierarchy)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

from oot3dhdtextgenerator.common import package_root

oot3d_data_path = package_root / "data" / "oot3d"
oot3d_assigned_csv_path = oot3d_data_path / "assigned.csv"
oot3d_unassigned_csv_path = oot3d_data_path / "unassigned.csv"


@dataclass(frozen=True, slots=True)
class HanziFrequencyEntry:
    """Character metadata row loaded from characters.csv."""

    character: str
    frequency: float
    pinyin: str
    definition: str


def _load_hanzi_frequency() -> list[HanziFrequencyEntry]:
    """Load character metadata from CSV."""
    output: list[HanziFrequencyEntry] = []
    with (package_root / "data" / "characters.csv").open(
        encoding="utf-8", newline=""
    ) as infile:
        reader = csv.reader(infile)
        for row in reader:
            character, frequency, pinyin, definition = row
            output.append(
                HanziFrequencyEntry(
                    character=character,
                    frequency=float(frequency),
                    pinyin=pinyin,
                    definition=definition,
                )
            )
    return output


hanzi_frequency = _load_hanzi_frequency()
characters = [entry.character for entry in hanzi_frequency]


def character_to_index(character: str) -> int:
    """Get index of a character in the ordered character list."""
    return characters.index(character)


__all__ = [
    "characters",
    "character_to_index",
    "HanziFrequencyEntry",
    "hanzi_frequency",
    "oot3d_assigned_csv_path",
    "oot3d_data_path",
    "oot3d_unassigned_csv_path",
]
