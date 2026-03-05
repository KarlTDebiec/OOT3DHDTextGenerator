#  Copyright 2020-2026 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Training dataset generator."""

from __future__ import annotations

import time
from itertools import product
from logging import info
from pathlib import Path
from platform import system
from random import sample

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pipescaler.core import Utility

from oot3dhdtextgenerator.core import TrainingDataset
from oot3dhdtextgenerator.data import hanzi_frequency


class TrainingDatasetGenerator(Utility):
    """Training dataset generator."""

    @classmethod
    def generate_character_images(
        cls, n_chars: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate character images.

        Arguments:
            n_chars: number of unique characters to include in dataset
        Returns:
            images and specifications
        """
        characters = [entry.character for entry in hanzi_frequency[:n_chars]]
        fonts = cls.get_default_font_paths()
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
            dtype=TrainingDataset.specification_dtypes,
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
        train_output_dir_path: Path,
        test_output_dir_path: Path,
    ) -> None:
        """Execute.

        Arguments:
            n_chars: number of unique characters to include in dataset
            test_proportion: proportion of dataset to be set aside for testing
            train_output_dir_path: train output directory path
            test_output_dir_path: test output directory path
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

        TrainingDataset.save_dataset(
            train_images, train_specifications, train_output_dir_path
        )
        info(
            f"Saved {train_images.shape[0]} character images to {train_output_dir_path}"
        )
        TrainingDataset.save_dataset(
            test_images, test_specifications, test_output_dir_path
        )
        info(f"Saved {test_images.shape[0]} character images to {test_output_dir_path}")

    @staticmethod
    def get_default_font_paths() -> list[str]:
        """Resolve platform-appropriate font paths.

        Returns:
            existing font path strings
        Raises:
            FileNotFoundError: if no candidate font path exists
        """
        system_name = system()
        if system_name == "Windows":
            candidates = [
                r"C:\Windows\Fonts\simhei.ttf",
                r"C:\Windows\Fonts\msyh.ttc",
                r"C:\Windows\Fonts\simsun.ttc",
                r"C:\Windows\Fonts\arial.ttf",
            ]
        elif system_name == "Darwin":
            candidates = [
                "/System/Library/Fonts/PingFang.ttc",
                "/System/Library/Fonts/STHeiti Medium.ttc",
                "/System/Library/Fonts/STHeiti Light.ttc",
                "/System/Library/Fonts/Hiragino Sans GB.ttc",
            ]
        else:
            candidates = [
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]

        existing_candidates = [
            candidate for candidate in candidates if Path(candidate).exists()
        ]

        if existing_candidates:
            return existing_candidates

        raise FileNotFoundError(
            "No default font was found for this platform. "
            "Pass an explicit font path in code."
        )

    @staticmethod
    def generate_character_image(  # noqa: PLR0913
        char: str,
        *,
        font: str | None = None,
        size: int = 12,
        fill: int = 0,
        x_offset: int = 0,
        y_offset: int = 0,
        rotation: int = 0,
    ) -> np.ndarray:
        """Generate a character image.

        Arguments:
            char: hanzi character
            font: font name
            size: font size
            fill: fill color
            x_offset: horizontal offset
            y_offset: vertical offset
            rotation: rotation in degrees
        Returns:
            numpy array of character image
        """
        if font is None:
            font = TrainingDatasetGenerator.get_default_font_paths()[0]
        font_type = ImageFont.truetype(font, size)
        image = Image.new("L", (16, 16), 0)
        draw = ImageDraw.Draw(image)
        _, _, width, height = draw.textbbox((0, 0), char, font=font_type)
        xy = ((16 - width) / 2, (16 - height) / 2)
        draw.text(xy, char, font=font_type, fill=int(fill))
        image = image.rotate(rotation, fillcolor=0)
        array = np.array(image)
        array = TrainingDatasetGenerator._translate_array_no_wrap(
            array, x_offset, y_offset
        )

        return array

    @staticmethod
    def _translate_array_no_wrap(
        array: np.ndarray, x_offset: int, y_offset: int
    ) -> np.ndarray:
        """Translate a 2D array without wraparound.

        Arguments:
            array: source image array
            x_offset: horizontal translation in pixels, positive is right
            y_offset: vertical translation in pixels, positive is down
        Returns:
            translated array with uncovered pixels filled with zero
        """
        height, width = array.shape
        translated = np.zeros_like(array)

        dst_x_start = max(0, x_offset)
        dst_x_end = min(width, width + x_offset)
        dst_y_start = max(0, y_offset)
        dst_y_end = min(height, height + y_offset)

        src_x_start = max(0, -x_offset)
        src_x_end = src_x_start + (dst_x_end - dst_x_start)
        src_y_start = max(0, -y_offset)
        src_y_end = src_y_start + (dst_y_end - dst_y_start)

        if dst_x_start < dst_x_end and dst_y_start < dst_y_end:
            translated[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = array[
                src_y_start:src_y_end,
                src_x_start:src_x_end,
            ]

        return translated

    @staticmethod
    def split_train_and_test(
        images: np.ndarray,
        specifications: np.ndarray,
        test_proportion: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split images and specifications into training and testing sets.

        Images are divided by character, such that test_proportion of each character
        is included in the test set.

        Arguments:
            images: images
            specifications: specifications
            test_proportion: proportion of images to set aside for testing
        Returns:
            train and test images and specifications
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
