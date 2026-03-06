#  Copyright 2020-2026 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Character assigner."""

from __future__ import annotations

from collections import Counter
from logging import warning
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flask import Flask
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor

from oot3dhdtextgenerator.common.validation import val_output_dir_path
from oot3dhdtextgenerator.core import AssignmentDataset, Model
from oot3dhdtextgenerator.data import characters as known_characters
from oot3dhdtextgenerator.data import hanzi_frequency, oot3d_data_path

from .character import Character
from .routes import route


class CharAssigner:
    """Character assigner."""

    unassigned_filter_values = {
        "visible",
        "top_prediction_available_only",
        "hidden",
    }
    assigned_filter_values = {
        "visible",
        "conflicts_only",
        "top_prediction_mismatch_only",
        "hidden",
    }

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
        self.n_chars = n_chars
        self.prior_probabilities = self.get_prior_probabilities(n_chars)
        self.characters = self.get_characters(
            self.dataset,
            unassigned_scores,
            assigned_scores_by_bytes,
        )
        self.update_character_predictions(
            prior_weight=0.0, exclude_assigned_from_predictions=False
        )

        self.app = Flask(__name__)

        route(self)

    def run(self, **kwargs: Any) -> None:
        """Run the Flask application."""
        self.app.run(**kwargs)

    @staticmethod
    def normalize_filters(
        unassigned_filter: str | None, assigned_filter: str | None
    ) -> tuple[str, str]:
        """Normalize filter values to supported options.

        Arguments:
            unassigned_filter: unassigned visibility filter
            assigned_filter: assigned visibility filter
        Returns:
            normalized unassigned and assigned filters
        """
        normalized_unassigned_filter = (
            unassigned_filter
            if unassigned_filter in CharAssigner.unassigned_filter_values
            else "visible"
        )
        normalized_assigned_filter = (
            assigned_filter
            if assigned_filter in CharAssigner.assigned_filter_values
            else "visible"
        )
        return normalized_unassigned_filter, normalized_assigned_filter

    @staticmethod
    def normalize_prior_weight_percent(prior_weight_percent: str | None) -> float:
        """Normalize prior-weight percentage value.

        Arguments:
            prior_weight_percent: prior weight in percent units [0, 100]
        Returns:
            normalized prior weight in percent units [0, 100]
        """
        try:
            if prior_weight_percent is None:
                return 0.0
            value = float(prior_weight_percent)
        except ValueError:
            return 0.0
        return min(100.0, max(0.0, value))

    @staticmethod
    def normalize_exclude_assigned_from_predictions(
        exclude_assigned_from_predictions: str | None,
    ) -> bool:
        """Normalize exclude-assigned checkbox value.

        Arguments:
            exclude_assigned_from_predictions: checkbox value from request
        Returns:
            whether to exclude currently assigned characters from predictions
        """
        return exclude_assigned_from_predictions in {"1", "true", "on", "yes"}

    @staticmethod
    def get_prior_probabilities(n_chars: int) -> np.ndarray:
        """Get normalized character prior probabilities for active labels.

        Arguments:
            n_chars: number of characters in active model label space
        Returns:
            normalized prior probabilities
        """
        priors = np.array(
            [entry.frequency for entry in hanzi_frequency[:n_chars]],
            dtype=np.float64,
        )
        if priors.size == 0:
            return priors
        priors = np.maximum(priors, 0.0)
        total = priors.sum()
        if total <= 0:
            return np.full(priors.shape, 1.0 / len(priors), dtype=np.float64)
        return priors / total

    @staticmethod
    def blend_scores(
        model_score: np.ndarray, prior_probabilities: np.ndarray, prior_weight: float
    ) -> np.ndarray:
        """Blend model and prior scores using a linear combination.

        Arguments:
            model_score: raw model score vector (log probabilities)
            prior_probabilities: normalized prior probabilities
            prior_weight: blend weight in [0.0, 1.0]
        Returns:
            blended score vector in probability space
        """
        if model_score.size == 0:
            return model_score
        model_probabilities = np.exp(model_score.astype(np.float64))
        return (
            1.0 - prior_weight
        ) * model_probabilities + prior_weight * prior_probabilities[
            : model_probabilities.shape[0]
        ]

    def update_character_predictions(
        self, prior_weight: float, *, exclude_assigned_from_predictions: bool
    ) -> None:
        """Update predictions for all characters at a given blend weight.

        Arguments:
            prior_weight: blend weight in [0.0, 1.0]
            exclude_assigned_from_predictions: whether to exclude currently assigned
              characters from prediction candidates
        """
        prediction_labels = np.array(known_characters[: self.n_chars], dtype=object)
        assigned_label_indexes = set()
        if exclude_assigned_from_predictions:
            character_indexes = {
                character: index for index, character in enumerate(known_characters)
            }
            for character in self.characters:
                if character.assignment is None:
                    continue
                assignment_index = character_indexes.get(character.assignment)
                if assignment_index is None or assignment_index >= self.n_chars:
                    continue
                assigned_label_indexes.add(assignment_index)
        for character in self.characters:
            if character.score is None:
                character.predictions = None
                continue
            blended = self.blend_scores(
                character.score, self.prior_probabilities, prior_weight
            )
            if assigned_label_indexes:
                blended = blended.copy()
                for assigned_label_index in assigned_label_indexes:
                    blended[assigned_label_index] = float("-inf")
            prediction_indexes = list(np.argsort(blended))[::-1]
            character.predictions = prediction_labels[prediction_indexes].tolist()[:10]

    @staticmethod
    def sort_characters(characters: list[Character]) -> list[Character]:
        """Sort unassigned and assigned characters for display.

        Arguments:
            characters: unsorted characters
        Returns:
            sorted characters
        """
        character_indexes = {
            character: index for index, character in enumerate(known_characters)
        }

        unassigned_characters = [char for char in characters if char.assignment is None]
        unassigned_characters = sorted(
            unassigned_characters,
            key=lambda char: (
                character_indexes.get(char.predictions[0], len(known_characters))
                if char.predictions
                else len(known_characters),
                char.id,
            ),
        )

        assigned_characters = [
            char for char in characters if char.assignment is not None
        ]
        assigned_characters = sorted(
            assigned_characters,
            key=lambda char: (
                character_indexes.get(char.assignment or "", len(known_characters)),
                char.id,
            ),
        )
        return unassigned_characters + assigned_characters

    @staticmethod
    def filter_characters(
        characters: list[Character],
        *,
        unassigned_filter: str,
        assigned_filter: str,
    ) -> list[Character]:
        """Filter characters for display.

        Arguments:
            characters: characters to filter
            unassigned_filter: unassigned visibility filter
            assigned_filter: assigned visibility filter
        Returns:
            filtered characters
        """
        sorted_characters = CharAssigner.sort_characters(characters)
        unassigned_characters = [
            char for char in sorted_characters if char.assignment is None
        ]
        assigned_characters = [
            char for char in sorted_characters if char.assignment is not None
        ]

        if unassigned_filter == "hidden":
            unassigned_characters = []
        elif unassigned_filter == "top_prediction_available_only":
            assigned_characters_set = {
                char.assignment
                for char in sorted_characters
                if char.assignment is not None
            }
            unassigned_characters = [
                char
                for char in unassigned_characters
                if char.predictions is not None
                and len(char.predictions) > 0
                and char.predictions[0] not in assigned_characters_set
            ]

        if assigned_filter == "hidden":
            assigned_characters = []
        elif assigned_filter == "conflicts_only":
            assigned_counts = Counter(
                char.assignment
                for char in assigned_characters
                if char.assignment is not None
            )
            assigned_characters = [
                char
                for char in assigned_characters
                if char.assignment is not None
                and assigned_counts.get(char.assignment, 0) > 1
            ]
        elif assigned_filter == "top_prediction_mismatch_only":
            character_indexes = {
                character: index for index, character in enumerate(known_characters)
            }
            assigned_characters = [
                char
                for char in assigned_characters
                if char.assignment is not None
                and char.predictions is not None
                and len(char.predictions) > 0
                and char.score is not None
                and character_indexes.get(char.assignment, len(known_characters))
                < int(char.score.shape[0])
                and char.predictions[0] != char.assignment
            ]

        return unassigned_characters + assigned_characters

    def get_display_characters(
        self,
        unassigned_filter: str | None,
        assigned_filter: str | None,
        prior_weight_percent: str | None,
        exclude_assigned_from_predictions: str | None,
    ) -> tuple[list[Character], str, str, float, bool]:
        """Get display characters and normalized filter values.

        Arguments:
            unassigned_filter: unassigned visibility filter
            assigned_filter: assigned visibility filter
            prior_weight_percent: prior weight in percent units [0, 100]
            exclude_assigned_from_predictions: checkbox value from request
        Returns:
            display characters and normalized filter values
        """
        (
            normalized_unassigned_filter,
            normalized_assigned_filter,
        ) = self.normalize_filters(unassigned_filter, assigned_filter)
        normalized_prior_weight_percent = self.normalize_prior_weight_percent(
            prior_weight_percent
        )
        normalized_exclude_assigned_from_predictions = (
            self.normalize_exclude_assigned_from_predictions(
                exclude_assigned_from_predictions
            )
        )
        self.update_character_predictions(
            normalized_prior_weight_percent / 100.0,
            exclude_assigned_from_predictions=normalized_exclude_assigned_from_predictions,
        )
        display_characters = self.filter_characters(
            self.characters,
            unassigned_filter=normalized_unassigned_filter,
            assigned_filter=normalized_assigned_filter,
        )
        return (
            display_characters,
            normalized_unassigned_filter,
            normalized_assigned_filter,
            normalized_prior_weight_percent,
            normalized_exclude_assigned_from_predictions,
        )

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
            prediction_labels = np.array(
                known_characters[: int(score.shape[0])], dtype=object
            )
            predictions = prediction_labels[prediction_indexes].tolist()[:10]
            characters.append(Character(i, char_array, None, predictions, score))
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
                prediction_labels = np.array(
                    known_characters[: int(score.shape[0])], dtype=object
                )
                predictions = prediction_labels[prediction_indexes].tolist()[:10]
            characters.append(Character(i, char_array, assignment, predictions, score))
            i += 1

        return characters
