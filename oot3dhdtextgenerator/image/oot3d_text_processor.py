#  Copyright 2020-2025 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Processes text images."""

from __future__ import annotations

from logging import info
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pipescaler.image.core.operators import ImageProcessor

from oot3dhdtextgenerator.common.validation import val_int, val_output_path
from oot3dhdtextgenerator.core import AssignmentDataset


class OOT3DHDTextProcessor(ImageProcessor):
    """Processes text images."""

    def __init__(
        self,
        assignment_path: Path | str,
        font: str = r"C:\Windows\Fonts\simhei.ttf",
        size: int = 48,
        offset: tuple[int, int] = (0, 0),
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.assignment_path = val_output_path(assignment_path, may_exist=True)
        self.assignment_dataset = AssignmentDataset(self.assignment_path)

        self.font = ImageFont.truetype(font, val_int(size, 1))
        self.size = val_int(size, 1)
        self.offset = offset

    def __call__(self, input_image: Image.Image) -> Image.Image:
        array = np.array(input_image)[:, :, 3]
        chars = self.assignment_dataset.get_chars_for_multi_char_array(array)
        if chars is None:
            raise FileNotFoundError(f"{self}: Image contains unknown characters")
        output_image = self.create_image(input_image, chars)

        return output_image

    def __repr__(self) -> str:
        """Representation."""
        return f"{self.__class__.__name__}(assignment_file={self.assignment_path!r})"

    def create_image(self, input_image, characters: list[str]) -> Image.Image:
        """Create image from characters.

        Arguments:
            input_image (Image.Image): Input image
            characters (list[str]): Characters to draw
        Returns:
            Image.Image: Output image
        """
        output_image_alpha = Image.new(
            "L", (input_image.size[0] * 4, input_image.size[1] * 4), 0
        )
        draw = ImageDraw.Draw(output_image_alpha)
        x = self.offset[0]
        y = self.offset[1]
        for i, character in enumerate(characters, 1):
            draw.text((x, y), character, font=self.font, fill=255, align="center")
            if i % 16 == 0:
                y += 64
                x = self.offset[0]
            else:
                x += 64

        output_array = np.zeros(
            (input_image.size[0] * 4, input_image.size[1] * 4, 4), dtype=np.uint8
        )
        output_array[:, :, 3] = np.array(output_image_alpha)
        output_image = Image.fromarray(output_array, mode="RGBA")

        return output_image

    def save_assignment_dataset(self):
        """Save assignment dataset to HDF5 file."""
        self.assignment_dataset.save_hdf5(
            self.assignment_dataset.assigned_char_bytes,
            self.assignment_dataset.unassigned_char_bytes,
            self.assignment_path,
        )
        info(f"Saved assignments to {self.assignment_path}")

    @classmethod
    def help_markdown(cls) -> str:
        """Short description of this tool in markdown, with links."""
        return "Reads OOT 3D text image and draws higher-resolution version."

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
