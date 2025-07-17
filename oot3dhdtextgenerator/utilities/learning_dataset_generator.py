#  Copyright 2020-2025 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Learning dataset generator."""

from __future__ import annotations

import time
from itertools import product
from logging import info
from pathlib import Path
from random import sample

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pipescaler.core import Utility

from oot3dhdtextgenerator.core import LearningDataset
from oot3dhdtextgenerator.data import hanzi_frequency


class LearningDatasetGenerator(Utility):
    """Learning dataset generator."""

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
        offsets = [-1, 0, 1]
        fills = [255]
        rotations = [-5, 0, 5]
        info(
            f"Generating images of "
            f"{n_chars} character{'s' if n_chars > 1 else ''} using "
            f"{len(fonts)} font{'s' if len(fonts) > 1 else ''}, "
            f"{len(sizes)} size{'s' if len(sizes) > 1 else ''}, "
            f"{len(offsets)} offset{'s' if len(offsets) > 1 else ''}, "
            f"{len(fills)} fill{'s' if len(fills) > 1 else ''}, and "
            f"{len(rotations)} rotation{'s' if len(rotations) > 1 else ''}"
        )
        specifications = np.array(
            list(
                product(
                    characters,
                    fonts,
                    sizes,
                    offsets,
                    offsets,
                    fills,
                    rotations,
                )
            ),
            dtype=LearningDataset.specification_dtypes,
        )
        n_images = len(specifications)
        info(f"Generating {n_images} images total")
        arrays = np.zeros((n_images, 16, 16), np.uint8)
        last_update_time = time.time()
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
            if i % 1000 == 0 and (current_time := time.time()) - last_update_time > 10:
                info(f"{int(float(i + 1) / n_images * 100):3d}% complete")
                last_update_time = current_time
        info("100% complete")

        return arrays, specifications

    @classmethod
    def run(
        cls,
        n_chars: int,
        test_proportion: float,
        train_outfile: Path,
        test_outfile: Path,
    ) -> None:
        """Execute.

        Arguments:
            n_chars: Number of unique characters to include in dataset
            test_proportion: Proportion of dataset to be set aside for testing
            train_outfile: Train output file path
            test_outfile: Test output file path
        """
        images, specifications = cls.generate_character_images(n_chars)
        info(f"Generated {images.shape[0]} character images")
        (
            train_images,
            train_specifications,
            test_images,
            test_specifications,
        ) = cls.split_train_and_test(images, specifications, test_proportion)
        info(
            f"Split into {train_images.shape[0]} train and "
            f"{test_images.shape[0]} test images"
        )

        LearningDataset.save_hdf5(train_images, train_specifications, train_outfile)
        info(f"Saved {train_images.shape[0]} character images to {train_outfile}")
        LearningDataset.save_hdf5(test_images, test_specifications, test_outfile)
        info(f"Saved {test_images.shape[0]} character images to {test_outfile}")

    @staticmethod
    def generate_character_image(
        char: str,
        *,
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
        # info(
        #     f"Generating image of {char} with font {font}, size {size}, fill {fill}, "
        #     f"x offset {x_offset}, y offset {y_offset}, and rotation {rotation}"
        # )
        image = Image.new("L", (16, 16), 0)
        draw = ImageDraw.Draw(image)
        font_type = ImageFont.truetype(font, size)
        _, _, width, height = draw.textbbox((0, 0), char, font=font_type)
        xy = ((16 - width) / 2, (16 - height) / 2)
        draw.text(xy, char, font=font_type, fill=int(fill))
        image = image.rotate(rotation)
        array = np.array(image)
        array = np.roll(array, (x_offset, y_offset), (0, 1))
        # image = Image.fromarray(array)
        # image.show()
        # input()

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
            test_proportion: Proportion of images to set aside for testing
        Returns:
            Train and test images and specifications
        """
        train_index_set = set()
        for character in set(specifications["character"]):
            indexes = set(np.where(specifications["character"] == character)[0])
            n_train = int(len(indexes) * (1.0 - test_proportion))
            train_index_set |= set(sample(list(indexes), n_train))
        test_index_set = set(range(images.shape[0])) - train_index_set

        train_indexes = sorted(train_index_set)
        test_indexes = sorted(test_index_set)

        train_images = images[train_indexes]
        train_specifications = specifications[train_indexes]
        test_images = images[test_indexes]
        test_specifications = specifications[test_indexes]

        return train_images, train_specifications, test_images, test_specifications
