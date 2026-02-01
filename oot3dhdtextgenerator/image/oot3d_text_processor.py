#  Copyright 2020-2026 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Processes text images."""

from __future__ import annotations

from hashlib import sha1
from logging import info, warning
from pathlib import Path
from platform import system
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pipescaler.image.core.operators import ImageProcessor

from oot3dhdtextgenerator.common.validation import (
    val_input_path,
    val_int,
    val_output_dir_path,
    val_output_path,
)
from oot3dhdtextgenerator.core import AssignmentDataset


class OOT3DHDTextProcessor(ImageProcessor):
    """Processes text images."""

    _char_size: tuple[int, int] = (16, 16)
    _scale: int = 4

    def __init__(
        self,
        assignment_path: Path | str,
        font: Path | str | None = None,
        size: int = 48,
        offset: tuple[int, int] = (0, 0),
        debug_dir: Path | str | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.assignment_path = val_output_path(assignment_path, exist_ok=True)
        self.assignment_dataset = AssignmentDataset(self.assignment_path)

        font_path = self._resolve_font_path(font)
        self.font = ImageFont.truetype(str(font_path), val_int(size, 1))
        self.size = val_int(size, 1)
        self.offset = offset
        self.debug_dir = None if debug_dir is None else val_output_dir_path(debug_dir)

    def __call__(self, input_image: Image.Image) -> Image.Image:
        array = np.array(input_image)[:, :, 3]
        chars = self.assignment_dataset.get_chars_for_multi_char_array(array)
        if chars is None:
            self._write_unknown_char_debug_artifacts(input_image, array)
            raise FileNotFoundError(f"{self}: Image contains unknown characters")
        output_image = self.create_image(input_image, chars)

        return output_image

    def __repr__(self) -> str:
        """Representation."""
        return f"{self.__class__.__name__}(assignment_file={self.assignment_path!r})"

    def create_image(self, input_image: Image.Image, characters: str) -> Image.Image:
        """Create image from characters.

        Arguments:
            input_image: Input image
            characters: Characters to draw, in row-major order
        Returns:
            Image.Image: Output image
        """
        if input_image.width % self._char_size[0] != 0:
            raise ValueError(
                f"Input width {input_image.width} is not divisible by "
                f"{self._char_size[0]}"
            )

        columns = input_image.width // self._char_size[0]
        cell = self._char_size[0] * self._scale
        output_image_alpha = Image.new(
            "L",
            (input_image.width * self._scale, input_image.height * self._scale),
            0,
        )
        draw = ImageDraw.Draw(output_image_alpha)
        x = self.offset[0]
        y = self.offset[1]
        for i, character in enumerate(characters, 1):
            draw.text((x, y), character, font=self.font, fill=255, align="center")
            if i % columns == 0:
                y += cell
                x = self.offset[0]
            else:
                x += cell

        output_array = np.zeros(
            (output_image_alpha.height, output_image_alpha.width, 4), dtype=np.uint8
        )
        output_array[:, :, 3] = np.array(output_image_alpha)
        output_image = Image.fromarray(output_array, mode="RGBA")

        return output_image

    def _resolve_font_path(self, font: Path | str | None) -> Path:
        """Resolve a platform-appropriate font path.

        Arguments:
            font: Explicit font path, or None to use an OS-specific default
        Returns:
            Path to a font file usable by PIL
        Raises:
            FileNotFoundError: If no suitable font can be found
        """
        if font is not None:
            return val_input_path(font)

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
            # Not a primary target, but try common fonts to keep things usable.
            candidates = [
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]

        for candidate in candidates:
            path = Path(candidate)
            if path.exists():
                return path

        raise FileNotFoundError(
            "No default font was found for this platform. "
            "Pass an explicit `font=` path."
        )

    def _write_unknown_char_debug_artifacts(
        self, input_image: Image.Image, alpha_array: np.ndarray
    ) -> None:
        """Write debug artifacts for an image with unknown characters.

        Arguments:
            input_image: Original input image
            alpha_array: Input alpha channel (H×W) as uint8
        """
        if self.debug_dir is None:
            return

        digest = sha1(alpha_array.tobytes()).hexdigest()[:12]
        out_dir = self.debug_dir / "unknown_text" / digest
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            input_image.save(out_dir / "input.png")
            Image.fromarray(alpha_array, mode="L").save(out_dir / "alpha.png")
        except OSError as exc:
            warning(f"{self}: Failed to write debug input images: {exc}")
            return

        char_h, char_w = self._char_size
        rows = alpha_array.shape[0] // char_h
        cols = alpha_array.shape[1] // char_w

        wrote_any = False
        for row in range(rows):
            for col in range(cols):
                tile = alpha_array[
                    row * char_h : (row + 1) * char_h,
                    col * char_w : (col + 1) * char_w,
                ]
                if tile.sum() == 0:
                    continue

                if self.assignment_dataset.get_char_for_char_array(tile) is not None:
                    continue

                tile_img = Image.fromarray(tile, mode="L").resize(
                    (char_w * self._scale, char_h * self._scale), resample=Image.NEAREST
                )
                tile_img.save(out_dir / f"tile_r{row:02d}_c{col:02d}.png")
                wrote_any = True

        if wrote_any:
            warning(f"{self}: Wrote unknown-character debug tiles to {out_dir}")

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
