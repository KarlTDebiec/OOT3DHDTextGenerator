#!/usr/bin/env python
#  Copyright 2020-2022 Karl T Debiec
#  All rights reserved. This software may be modified and distributed under
#  the terms of the BSD license. See the LICENSE file for details.
"""Hanzi character dataset generator."""
from argparse import ArgumentParser
from itertools import product
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from oot3dhdtextgenerator import hanzi_frequency
from oot3dhdtextgenerator.common import CommandLineInterface, set_logging_verbosity


class HanziDatasetGenerator(CommandLineInterface):
    """Hanzi character dataset generator."""

    @classmethod
    def add_arguments_to_argparser(cls, parser: ArgumentParser) -> None:
        """Add arguments to a nascent argument parser.

        Arguments:
            parser: Nascent argument parser
        """
        super().add_arguments_to_argparser(parser)

        parser.add_argument(
            "--n_chars",
            type=cls.int_arg(min_value=10, max_value=9933),
            default=10,
            help="number of characters to include in dataset, starting from the most"
            "common and ending with the least common (default: 10, max: 9933)",
        )
        parser.add_argument(
            "--outfile",
            type=cls.output_file_arg(),
            default="cmn-Hans.h5",
            help="output file",
        )

    @classmethod
    def execute(cls, verbosity: int, **kwargs: Any) -> None:
        """Execute with provided keyword arguments.

        Arguments:
            verbosity: Verbosity level
            **kwargs: Command-line arguments
        """
        set_logging_verbosity(verbosity)
        cls.execute_internal(**kwargs)

    @classmethod
    def execute_internal(cls, n_chars: int, outfile: str) -> None:
        images, labels = cls.create_character_images(n_chars)

        print(f"Must save to {outfile}")

    @classmethod
    def create_character_images(
        cls, n_chars: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        fonts = [
            "C:\Windows\Fonts\simhei.ttf",
        ]
        sizes = [15, 16]
        offsets = [-1, 0, 1]
        fills = [215, 225, 235, 245, 255]
        rotations = [0]
        n_images = (
            n_chars
            * len(fonts)
            * len(sizes)
            * len(fills)
            * len(offsets)
            * len(offsets)
            * len(rotations)
        )
        arrays = np.zeros((n_images, 16, 16), np.uint8)
        labels = np.zeros(n_images, str)
        i = 0
        for char_i in range(n_chars):
            char = hanzi_frequency.loc[0, "character"]
            print(char_i, char)
            for font in fonts:
                for size in sizes:
                    for fill in fills:
                        for offset in product(offsets, offsets):
                            for rotation in rotations:
                                arrays[i] = cls.create_character_image(
                                    char,
                                    font=font,
                                    size=size,
                                    fill=fill,
                                    offset=offset,
                                    rotation=rotation,
                                )
                                labels[i] = char_i
                                i += 1
        return arrays, labels

    @staticmethod
    def create_character_image(
        char: str,
        font: str = "C:\Windows\Fonts\simhei.ttf",
        size: int = 12,
        fill: int = 0,
        offset: tuple[int, int] = (0, 0),
        rotation: int = 0,
    ) -> np.ndarray:
        image = Image.new("L", (16, 16), 0)
        draw = ImageDraw.Draw(image)
        font_type = ImageFont.truetype(font, size)
        width, height = draw.textsize(char, font=font_type)
        xy = ((16 - width) / 2, (16 - height) / 2)
        draw.text(xy, char, font=font_type, fill=fill)
        image = image.rotate(rotation)
        array = np.array(image)
        array = np.roll(array, offset, (0, 1))

        return array


if __name__ == "__main__":
    HanziDatasetGenerator.main()
