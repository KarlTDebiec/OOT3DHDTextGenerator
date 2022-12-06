#!/usr/bin/env python
#  Copyright 2020-2022 Karl T Debiec
#  All rights reserved. This software may be modified and distributed under
#  the terms of the BSD license. See the LICENSE file for details.
"""Hanzi character dataset generator."""
from argparse import ArgumentParser
from itertools import product
from logging import info
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from oot3dhdtextgenerator import hanzi_frequency
from oot3dhdtextgenerator.common import (
    CommandLineInterface,
    int_arg,
    output_file_arg,
    set_logging_verbosity,
    validate_output_file,
)
from oot3dhdtextgenerator.hanzi_dataset import HanziDataset


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
            type=int_arg(min_value=10, max_value=9933),
            default=10,
            help="number of characters to include in dataset, starting from the most "
            "common and ending with the least common (default: %(default)d, max: 9933)",
        )
        parser.add_argument(
            "--outfile",
            type=output_file_arg(),
            default="cmn-Hans.h5",
            help="output file (default: cmn-Hans.h5)",
        )

    @classmethod
    def generate_character_images(
        cls, n_chars: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        characters = hanzi_frequency["character"].values[:n_chars]
        fonts = [
            r"C:\Windows\Fonts\simhei.ttf",
        ]
        sizes = [14, 15, 16]
        offsets = [-2, -1, 0, 1, 2]
        fills = [215, 220, 225, 230, 235, 240, 245, 250, 255]
        rotations = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
        combinations = product(
            characters,
            fonts,
            sizes,
            offsets,
            offsets,
            fills,
            rotations,
        )
        specifications = np.array(
            list(combinations), dtype=HanziDataset.specification_dtypes
        )
        info(f"Generating {len(specifications)} images")
        arrays = np.zeros((len(specifications), 16, 16), np.uint8)
        for i, specification in enumerate(specifications):
            arrays[i] = cls.generate_character_image(
                specification["character"],
                font=specification["font"],
                size=specification["size"],
                fill=specification["fill"],
                offset=(specification["x_offset"], specification["y_offset"]),
                rotation=specification["rotation"],
            )

        return arrays, specifications

    @classmethod
    def main(cls) -> None:
        """Execute from command line."""
        parser = cls.argparser()
        kwargs = vars(parser.parse_args())
        verbosity = kwargs.pop("verbosity", 1)
        set_logging_verbosity(verbosity)
        cls.main_internal(**kwargs)

    @classmethod
    def main_internal(cls, n_chars: int, outfile: Union[Path, str]) -> None:
        images, labels = cls.generate_character_images(n_chars)
        info(f"Generated {images.shape[0]} character images")

        outfile = validate_output_file(outfile)
        HanziDataset.save_hdf5(images, labels, outfile)
        info(f"Saved {images.shape[0]} character images to {outfile}")

    @staticmethod
    def generate_character_image(
        char: str,
        font: str = r"C:\Windows\Fonts\simhei.ttf",
        size: int = 12,
        fill: int = 0,
        offset: tuple[int, int] = (0, 0),
        rotation: int = 0,
    ) -> np.ndarray:
        image = Image.new("L", (16, 16), 0)
        draw = ImageDraw.Draw(image)
        font_type = ImageFont.truetype(font, size)
        _, _, width, height = draw.textbbox((0, 0), char, font=font_type)
        xy = ((16 - width) / 2, (16 - height) / 2)
        draw.text(xy, char, font=font_type, fill=int(fill))
        image = image.rotate(rotation)
        array = np.array(image)
        array = np.roll(array, offset, (0, 1))

        return array


if __name__ == "__main__":
    HanziDatasetGenerator.main()
