#!/usr/bin/env python
#  Copyright 2020-2022 Karl T Debiec
#  All rights reserved. This software may be modified and distributed under
#  the terms of the BSD license. See the LICENSE file for details.
"""Processes text images."""
from logging import info
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from pipescaler.core.image import Processor

from oot3dhdtextgenerator.common import validate_output_file
from oot3dhdtextgenerator.core import AssignmentDataset


class OOT3DHDTextProcessor(Processor):
    """Processes text images."""

    def __init__(
        self,
        assignment_file: Union[Path, str],
        # font: str,
        # size: int,
        # offset: tuple[int, int] = (0, 0),
    ):
        self.assignment_file = validate_output_file(assignment_file, exists_ok=True)
        self.assignment_dataset = AssignmentDataset(self.assignment_file)

        # self.font = font
        # self.size = validate_int(size, 1)
        # self.offset = offset

    def __call__(self, input_image: Image.Image) -> Image.Image:
        characters = self.assignment_dataset[np.array(input_image)[:, :, 3]]
        characters = None
        if characters is None:
            raise FileNotFoundError(f"{self}: Image contains unknown characters")
        return input_image

    def __repr__(self):
        """Representation."""
        return f"{self.__class__.__name__}(assignment_file={self.assignment_file})"

    # def create_image(self, characters: list[str]) -> PipeImage:
    #     hires_image = Image.new("L", (1024, 1024), 0)
    #     draw = ImageDraw.Draw(hires_image)
    #     x = self.offset[0]
    #     y = self.offset[1]
    #     for character in characters:
    #         draw.text((x, y), character, font=self.font, fill=255, align="center")
    #         x += self.size

    def save_assignment_dataset(self):
        """Save assignment dataset to HDF5 file."""
        self.assignment_dataset.save_hdf5(
            self.assignment_dataset.assigned_chars,
            self.assignment_dataset.unassigned_chars,
            self.assignment_file,
        )
        info(f"Saved assignments to {self.assignment_file}")

    @classmethod
    def inputs(cls) -> dict[str, tuple[str, ...]]:
        """Inputs to this operator."""
        return {
            "input": ("RGBA",),
        }

    @classmethod
    def outputs(cls) -> dict[str, tuple[str, ...]]:
        """Outputs of this operator."""
        return {
            "output": ("RGBA",),
        }
