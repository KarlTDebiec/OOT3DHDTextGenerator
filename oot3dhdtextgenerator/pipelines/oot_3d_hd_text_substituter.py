#!/usr/bin/env python
#  Copyright 2020-2022 Karl T Debiec
#  All rights reserved. This software may be modified and distributed under
#  the terms of the BSD license. See the LICENSE file for details.
"""Generates hi-res text images for The Legend of Zelda: Ocarina of Time 3D."""
from logging import info
from pathlib import Path
from typing import Union

import h5py
import numpy as np
from pipescaler.core.pipelines import PipeImage

from oot3dhdtextgenerator.common import validate_input_file, validate_int


class OOT3DHDTextSubstituter:
    """Generates hi-res text images for The Legend of Zelda: Ocarina of Time 3D."""

    def __init__(
        self,
        cache_file: Union[Path, str],
        model_file: Union[Path, str],
        font: str,
        size: int,
        offset: tuple[int, int] = (0, 0),
    ):
        self.cache_file = validate_input_file(cache_file)
        self.model_file = validate_input_file(model_file)
        self.font = font
        self.size = validate_int(size, 1)
        self.offset = offset

        self.characters = {}

        self.load_hdf5()

    def __call__(self, pipe_image: PipeImage) -> PipeImage:
        all_characters_assigned = True
        character_arrays = self.get_character_arrays(pipe_image.image)
        characters = []
        for character_array in character_arrays:
            if character_array in self.characters:
                if self.characters[character_array] != "":
                    # Character is known and assigned
                    characters.append(self.characters[character_array])
                else:
                    # Character is known but not assigned
                    all_characters_assigned = False
            else:
                # Character is unknown
                self.characters[character_array] = ""
                all_characters_assigned = False
        if all_characters_assigned:
            substitute_image = self.get_image(characters)
            return PipeImage(substitute_image, parents=pipe_image)
        else:
            raise FileNotFoundError(
                f"{self}: Image {pipe_image.name} contains "
                f"{len(character_arrays) - len(characters)} unknown characters"
            )

    def load_hdf5(self) -> None:
        """Load cache from HDF5 file."""
        info(f"{self}: Loading cache from '{self.cache_file}'")

        with h5py.File(self.cache_file) as cache:

            # Load characters
            if "characters" in cache:
                images = np.array(cache["characters/images"])
                assignments = np.array(cache["characters/assignments"])
                for i, a in zip(images, assignments):
                    self.characters[i.tobytes()] = a.decode("UTF8")

    def save_hdf5(self) -> None:
        """Save cache to HDF5 file."""
        info(f"{self}: Saving cache to '{self.cache_file}'")

        with h5py.File(self.cache_file) as cache:
            if "characters" in cache:
                del cache["characters"]
            cache.create_dataset(
                "characters/images",
                data=np.stack(
                    [np.frombuffer(k, dtype=np.uint8) for k in self.characters.keys()]
                ).reshape((-1, 16, 16)),
                dtype=np.uint8,
                chunks=True,
                compression="gzip",
            )
            cache.create_dataset(
                "characters/assignments",
                data=[a.encode("UTF8") for a in list(self.characters.values())],
                dtype="S4",
                chunks=True,
                compression="gzip",
            )
