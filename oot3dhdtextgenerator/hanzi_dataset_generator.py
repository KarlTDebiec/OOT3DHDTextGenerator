#!/usr/bin/env python
#  Copyright 2020-2022 Karl T Debiec
#  All rights reserved. This software may be modified and distributed under
#  the terms of the BSD license. See the LICENSE file for details.
"""Hanzi character dataset generator."""
from argparse import ArgumentParser
from itertools import product
from logging import info
from pathlib import Path
from random import sample
from typing import Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from oot3dhdtextgenerator import hanzi_frequency
from oot3dhdtextgenerator.common import (
    CommandLineInterface,
    float_arg,
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
            "--test_proportion",
            default=0.1,
            type=float_arg(min_value=0, max_value=1),
            help="proportion of dataset to be used as test set (default: %(default)f)",
        )
        parser.add_argument(
            "--outfile",
            type=output_file_arg(),
            default="cmn-Hans.h5",
            help="output file (default: %(default)s)",
        )

    @classmethod
    def generate_character_images(
        cls, n_chars: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate character images.

        Arguments:
            n_chars: Number of unique characters to include in dataset
        Returns:
            Images and specifications
        """
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
                char=specification["character"],
                font=specification["font"],
                size=specification["size"],
                fill=specification["fill"],
                x_offset=specification["x_offset"],
                y_offset=specification["y_offset"],
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
        """Execute from command line.

        Arguments:
            n_chars: Number of unique characters to include in dataset
            outfile: Output file
        """
        images, specifications = cls.generate_character_images(n_chars)
        info(f"Generated {images.shape[0]} character images")
        (
            train_images,
            train_specifications,
            test_images,
            test_specifications,
        ) = cls.split_train_and_test(images, specifications)
        info(
            f"Split into {train_images.shape[0]} train and "
            f"{test_images.shape[0]} test images"
        )

        outfile = validate_output_file(outfile)
        HanziDataset.save_hdf5(train_images, train_specifications, outfile, "train")
        HanziDataset.save_hdf5(test_images, test_specifications, outfile, "test")
        info(f"Saved {images.shape[0]} character images to {outfile}")

    @staticmethod
    def generate_character_image(
        char: str,
        font: str = r"C:\Windows\Fonts\simhei.ttf",
        size: int = 12,
        fill: int = 0,
        x_offset: int = 0,
        y_offset: int = 0,
        rotation: int = 0,
    ) -> np.ndarray:
        """Generate a character image.

        Arguments:
            char: Hanzi character
            font: Font name
            size: Font size
            fill: Fill color
            x_offset: Horizontal offset
            y_offset: Vertical offset
            rotation: Rotation in degrees
        Returns:
            numpy array of character image
        """
        image = Image.new("L", (16, 16), 0)
        draw = ImageDraw.Draw(image)
        font_type = ImageFont.truetype(font, size)
        _, _, width, height = draw.textbbox((0, 0), char, font=font_type)
        xy = ((16 - width) / 2, (16 - height) / 2)
        draw.text(xy, char, font=font_type, fill=int(fill))
        image = image.rotate(rotation)
        array = np.array(image)
        array = np.roll(array, (x_offset, y_offset), (0, 1))

        return array

    @staticmethod
    def split_train_and_test(
        images: np.ndarray, specifications: np.ndarray, test_proportion: float = 0.1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split images and specifications into training and testing sets.

        Images are divided by character, such that test_proportion of each character
        is included in the test set.

        Arguments:
            images: Images
            specifications: Specifications
            test_proportion: Proportion of images to use for testing
        Returns:
            Train and test images and specifications
        """
        train_index_set = set()
        characters = set(specifications["character"])
        for character in characters:
            indexes = set(np.where(specifications["character"] == character)[0])
            n_test = int(len(indexes) * test_proportion)
            train_index_set |= sample(list(indexes), n_test)
        test_index_set = set(range(images.shape[0])) - train_index_set
        train_indexes = sorted(train_index_set)
        test_indexes = sorted(test_index_set)
        train_images = images[train_indexes]
        train_labels = specifications[train_indexes]
        test_images = images[test_indexes]
        test_labels = specifications[test_indexes]

        return train_images, train_labels, test_images, test_labels


if __name__ == "__main__":
    HanziDatasetGenerator.main()
