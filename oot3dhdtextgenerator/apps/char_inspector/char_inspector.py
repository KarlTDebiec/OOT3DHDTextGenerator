#  Copyright 2020-2026 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Character inspector."""

from __future__ import annotations

from base64 import b64encode
from dataclasses import dataclass, field
from io import BytesIO
from itertools import product
from pathlib import Path

import numpy as np
from flask import Flask
from PIL import Image
from PIL.ImageOps import invert

from oot3dhdtextgenerator.common.validation import val_output_dir_path
from oot3dhdtextgenerator.core import AssignmentDataset
from oot3dhdtextgenerator.data import characters as known_characters
from oot3dhdtextgenerator.data import oot3d_data_path
from oot3dhdtextgenerator.utilities.training_dataset_generator import (
    TrainingDatasetGenerator,
)

from .routes import route


@dataclass(frozen=True, slots=True)
class InspectorFilters:
    """Training-image filter selection."""

    fonts: tuple[str, ...]
    sizes: tuple[int, ...]
    x_offsets: tuple[int, ...]
    y_offsets: tuple[int, ...]
    rotations: tuple[int, ...]
    fills: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TrainingImagePreview:
    """Synthetic training image preview metadata."""

    image: str
    label: str


@dataclass(slots=True)
class CharInspectorRow:
    """Character row for inspector display."""

    id: int
    assignment: str
    array: np.ndarray
    _real_image: str | None = field(default=None, init=False, repr=False)

    @property
    def real_image(self) -> str:
        """Base64 encoded PNG representation of the real assigned image."""
        if self._real_image is not None:
            return self._real_image
        self._real_image = CharInspector.array_to_data_url(self.array)
        return self._real_image


@dataclass(frozen=True, slots=True)
class DisplayRow:
    """Display row and synthetic previews."""

    row: CharInspectorRow
    training_previews: tuple[TrainingImagePreview, ...]


class CharInspector:
    """Character inspector web application."""

    default_page_size = 40
    available_sizes = (14, 15, 16)
    available_offsets = (-1, 0, 1)
    available_rotations = (-5, 0, 5)
    available_fills = (255,)

    def __init__(
        self,
        n_chars: int,
        assignment_dir_path: Path = oot3d_data_path,
    ) -> None:
        """Initialize inspector application.

        Arguments:
            n_chars: number of characters included in active label space
            assignment_dir_path: assignment CSV directory
        """
        self.n_chars = n_chars
        self.assignment_dir_path = val_output_dir_path(assignment_dir_path)
        self.dataset = AssignmentDataset(self.assignment_dir_path)
        self.available_fonts = tuple(TrainingDatasetGenerator.get_default_font_paths())
        self.available_font_labels = tuple(
            (font, Path(font).name) for font in self.available_fonts
        )
        self.rows = self.get_rows()
        self.synthetic_image_cache: dict[
            tuple[str, str, int, int, int, int, int], str
        ] = {}
        self.app = Flask(__name__)
        route(self)

    def run(self, **kwargs) -> None:
        """Run the Flask application."""
        self.app.run(**kwargs)

    def get_rows(self) -> list[CharInspectorRow]:
        """Get sorted assigned-character rows within active label space."""
        character_indexes = {
            character: index for index, character in enumerate(known_characters)
        }
        unsorted_rows: list[tuple[str, np.ndarray]] = []
        for char_bytes, assignment in self.dataset.assigned_char_bytes.items():
            assignment_index = character_indexes.get(assignment)
            if assignment_index is None or assignment_index >= self.n_chars:
                continue
            array = self.dataset.bytes_to_array(char_bytes)
            unsorted_rows.append((assignment, array))
        sorted_rows = sorted(
            unsorted_rows,
            key=lambda row: (
                character_indexes.get(row[0], len(known_characters)),
                row[1].tobytes(),
            ),
        )
        return [
            CharInspectorRow(i, assignment, array)
            for i, (assignment, array) in enumerate(sorted_rows)
        ]

    @staticmethod
    def array_to_data_url(array: np.ndarray) -> str:
        """Convert character image array to base64 PNG data URL."""
        image = Image.fromarray(array)
        image = invert(image)
        image_io = BytesIO()
        image.save(image_io, format="PNG")
        b64_image = b64encode(image_io.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64_image}"

    def normalize_filters(  # noqa: PLR0913
        self,
        fonts: list[str],
        sizes: list[str],
        x_offsets: list[str],
        y_offsets: list[str],
        rotations: list[str],
        fills: list[str],
    ) -> InspectorFilters:
        """Normalize filter selections to supported option values."""
        normalized_fonts = self.normalize_selected_strings(fonts, self.available_fonts)
        normalized_sizes = self.normalize_selected_ints(sizes, self.available_sizes)
        normalized_x_offsets = self.normalize_selected_ints(
            x_offsets, self.available_offsets
        )
        normalized_y_offsets = self.normalize_selected_ints(
            y_offsets, self.available_offsets
        )
        normalized_rotations = self.normalize_selected_ints(
            rotations, self.available_rotations
        )
        normalized_fills = self.normalize_selected_ints(fills, self.available_fills)
        return InspectorFilters(
            fonts=normalized_fonts,
            sizes=normalized_sizes,
            x_offsets=normalized_x_offsets,
            y_offsets=normalized_y_offsets,
            rotations=normalized_rotations,
            fills=normalized_fills,
        )

    @staticmethod
    def normalize_selected_strings(
        selected_values: list[str], available_values: tuple[str, ...]
    ) -> tuple[str, ...]:
        """Normalize selected string values to non-empty supported subset."""
        selected_value_set = {
            selected_value
            for selected_value in selected_values
            if selected_value in available_values
        }
        if selected_value_set:
            return tuple(
                value for value in available_values if value in selected_value_set
            )
        return available_values

    @staticmethod
    def normalize_selected_ints(
        selected_values: list[str], available_values: tuple[int, ...]
    ) -> tuple[int, ...]:
        """Normalize selected int values to non-empty supported subset."""
        parsed_values: set[int] = set()
        for selected_value in selected_values:
            try:
                parsed_value = int(selected_value)
            except ValueError:
                continue
            if parsed_value in available_values:
                parsed_values.add(parsed_value)
        if parsed_values:
            return tuple(value for value in available_values if value in parsed_values)
        return available_values

    def get_display_rows_page(
        self,
        filters: InspectorFilters,
        *,
        offset: int,
        limit: int,
    ) -> tuple[list[DisplayRow], int, bool]:
        """Get paged display rows and pagination metadata."""
        total_count = len(self.rows)
        page_start = min(max(0, offset), total_count)
        page_end = min(total_count, page_start + max(1, limit))
        page_rows = self.rows[page_start:page_end]
        display_rows = [
            DisplayRow(
                row=row,
                training_previews=self.get_training_previews(row.assignment, filters),
            )
            for row in page_rows
        ]
        has_more = page_end < total_count
        return display_rows, total_count, has_more

    def get_training_previews(
        self, assignment: str, filters: InspectorFilters
    ) -> tuple[TrainingImagePreview, ...]:
        """Get synthetic previews for assignment and filter selection."""
        previews: list[TrainingImagePreview] = []
        for (
            font,
            size,
            x_offset,
            y_offset,
            rotation,
            fill,
        ) in product(
            filters.fonts,
            filters.sizes,
            filters.x_offsets,
            filters.y_offsets,
            filters.rotations,
            filters.fills,
        ):
            cache_key = (
                assignment,
                font,
                size,
                x_offset,
                y_offset,
                rotation,
                fill,
            )
            image = self.synthetic_image_cache.get(cache_key)
            if image is None:
                synthetic_array = TrainingDatasetGenerator.generate_character_image(
                    assignment,
                    font=font,
                    size=size,
                    fill=fill,
                    x_offset=x_offset,
                    y_offset=y_offset,
                    rotation=rotation,
                )
                image = self.array_to_data_url(synthetic_array)
                self.synthetic_image_cache[cache_key] = image
            label = (
                f"{Path(font).stem} | sz {size} | x {x_offset:+d} | y {y_offset:+d}"
                f" | rot {rotation:+d}"
            )
            previews.append(TrainingImagePreview(image=image, label=label))
        return tuple(previews)
