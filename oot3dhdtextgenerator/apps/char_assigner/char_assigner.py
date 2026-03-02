#  Copyright 2020-2026 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Character assigner."""

from __future__ import annotations

from logging import warning
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from flask import Flask
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor

from oot3dhdtextgenerator.common.validation import val_output_dir_path
from oot3dhdtextgenerator.core import AssignmentDataset, Model
from oot3dhdtextgenerator.data import characters as known_characters
from oot3dhdtextgenerator.data import oot3d_data_path

from .character import Character
from .routes import route

if TYPE_CHECKING:
    from pathlib import Path


class CharAssigner:
    """Character assigner."""

    def __init__(
        self,
        n_chars: int,
        model_input_path: Path,
        assignment_dir_path: Path = oot3d_data_path,
        *,
        cuda_enabled: bool = True,
        mps_enabled: bool = True,
    ) -> None:
        """Run character assigner.

        Arguments:
            n_chars: number of characters included in model
            model_input_path: model pth file
            assignment_dir_path: assignment CSV directory
            cuda_enabled: whether to use CUDA
            mps_enabled: whether to use macOS GPU
        """
        # Determine which device to use
        cuda_enabled = torch.cuda.is_available() and cuda_enabled
        if cuda_enabled:
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and mps_enabled:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        checkpoint = torch.load(model_input_path, map_location=device)
        state_dict: dict[str, Any]
        checkpoint_n_chars: int | None = None
        normalization_mean = 0.1307
        normalization_std = 0.3081
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            normalization = checkpoint.get("normalization", {})
            if isinstance(normalization, dict):
                normalization_mean = float(
                    normalization.get("mean", normalization_mean)
                )
                normalization_std = float(normalization.get("std", normalization_std))
            checkpoint_n_chars = checkpoint.get("n_chars")
        else:
            state_dict = checkpoint

        if checkpoint_n_chars is not None and checkpoint_n_chars != n_chars:
            warning(
                (
                    "Model checkpoint n_chars (%d) differs from CLI n_chars (%d); "
                    "using checkpoint value"
                ),
                checkpoint_n_chars,
                n_chars,
            )
            n_chars = int(checkpoint_n_chars)

        # Load assignment data
        self.assignment_dir_path = val_output_dir_path(assignment_dir_path)
        self.dataset = AssignmentDataset(self.assignment_dir_path)
        self.dataset.transform = Compose(
            [ToTensor(), Normalize((normalization_mean,), (normalization_std,))]
        )

        unassigned_char_bytes = list(self.dataset.unassigned_char_bytes)
        assigned_char_bytes = list(self.dataset.assigned_char_bytes)
        if not unassigned_char_bytes and not assigned_char_bytes:
            unassigned_scores = np.empty((0, n_chars), dtype=np.float32)
            assigned_scores_by_bytes: dict[bytes, np.ndarray] = {}
        else:
            # Load model
            model = Model(n_chars)
            model.load_state_dict(state_dict)
            model.eval()
            model = model.to(device)

            all_char_bytes = unassigned_char_bytes + assigned_char_bytes
            data = self.get_tensors(self.dataset, all_char_bytes).to(device)

            # Get predictions
            scores = model(data)  # pylint: disable=not-callable
            scores = scores.detach().cpu().numpy()
            unassigned_scores = scores[: len(unassigned_char_bytes)]
            assigned_scores_by_bytes = {
                char_bytes: score
                for char_bytes, score in zip(
                    assigned_char_bytes, scores[len(unassigned_char_bytes) :]
                )
            }

        # Prepare characters for frontend
        self.characters = self.get_characters(
            self.dataset,
            unassigned_scores,
            assigned_scores_by_bytes,
        )

        self.app = Flask(__name__)

        route(self)

    def run(self, **kwargs: Any) -> None:
        """Run the Flask application."""
        self.app.run(**kwargs)

    @staticmethod
    def get_tensors(
        dataset: AssignmentDataset, char_bytes_list: list[bytes]
    ) -> torch.Tensor:
        """Get transformed character tensors for a list of character bytes.

        Arguments:
            dataset: assignment dataset
            char_bytes_list: character byte arrays to transform
        Returns:
            stacked transformed tensors
        """
        if dataset.transform is None:
            raise RuntimeError("AssignmentDataset transform is not configured")

        tensors = []
        for char_bytes in char_bytes_list:
            char_array = dataset.bytes_to_array(char_bytes)
            char_image = Image.fromarray(char_array)
            char_tensor = dataset.transform(char_image)
            if not isinstance(char_tensor, torch.Tensor):
                raise TypeError(
                    f"Expected Tensor image transform, got {type(char_tensor)}"
                )
            tensors.append(char_tensor)

        if not tensors:
            return torch.empty((0, 1, *dataset.char_array_shape), dtype=torch.float32)
        return torch.stack(tensors)

    @staticmethod
    def get_characters(
        dataset: AssignmentDataset,
        unassigned_scores: np.ndarray,
        assigned_scores_by_bytes: dict[bytes, np.ndarray],
    ) -> list[Character]:
        """Get characters.

        Arguments:
            dataset: assignment dataset
            unassigned_scores: scores for unassigned characters in dataset order
            assigned_scores_by_bytes: scores for assigned characters by byte key
        Returns:
            list of characters
        """
        characters = []
        label_count = (
            int(unassigned_scores.shape[1]) if unassigned_scores.ndim == 2 else 0
        )
        if assigned_scores_by_bytes:
            label_count = max(
                label_count, len(next(iter(assigned_scores_by_bytes.values())))
            )
        prediction_labels = np.array(known_characters[:label_count], dtype=object)
        i = 0
        unassigned_items = list(zip(dataset.unassigned_char_bytes, unassigned_scores))
        unassigned_items = sorted(
            enumerate(unassigned_items),
            key=lambda item: (
                int(np.argmax(item[1][1]))
                if item[1][1].size > 0
                else len(known_characters),
                item[0],
            ),
        )
        for _, (char_bytes, score) in unassigned_items:
            char_array = dataset.bytes_to_array(char_bytes)
            prediction_indexes = list(np.argsort(score))[::-1]
            predictions = prediction_labels[prediction_indexes].tolist()[:10]
            characters.append(Character(i, char_array, None, predictions))
            i += 1
        character_indexes = {
            character: index for index, character in enumerate(known_characters)
        }
        assigned_items = sorted(
            dataset.assigned_char_bytes.items(),
            key=lambda item: character_indexes.get(item[1], len(known_characters)),
        )
        for char_bytes, assignment in assigned_items:
            char_array = dataset.bytes_to_array(char_bytes)
            score = assigned_scores_by_bytes.get(char_bytes)
            predictions: list[str] | None = None
            if score is not None:
                prediction_indexes = list(np.argsort(score))[::-1]
                predictions = prediction_labels[prediction_indexes].tolist()[:10]
            characters.append(Character(i, char_array, assignment, predictions))
            i += 1

        return characters
