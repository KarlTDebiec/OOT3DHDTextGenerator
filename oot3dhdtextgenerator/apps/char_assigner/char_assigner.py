#  Copyright 2020-2026 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Character assigner."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from flask import Flask
from torch.utils.data import DataLoader

from oot3dhdtextgenerator.common.validation import val_input_path
from oot3dhdtextgenerator.core import AssignmentDataset, Model

from .character import Character
from .routes import route

if TYPE_CHECKING:
    from pathlib import Path


class CharAssigner:
    """Character assigner."""

    def __init__(
        self,
        n_chars: int,
        assignment_path: Path,
        model_input_path: Path,
        *,
        cuda_enabled: bool = True,
        mps_enabled: bool = True,
    ) -> None:
        """Run character assigner.

        Arguments:
            n_chars: Number of characters included in model
            assignment_path: Assignment HDF5 file
            model_input_path: Model pth file
            cuda_enabled: Whether to use CUDA
            mps_enabled: Whether to use macOS GPU
        """
        # Determine which device to use
        cuda_enabled = torch.cuda.is_available() and cuda_enabled
        if cuda_enabled:
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and mps_enabled:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        # Load assignment data
        self.assignment_path = val_input_path(assignment_path)
        self.dataset = AssignmentDataset(self.assignment_path)
        loader_kwargs = {"batch_size": len(self.dataset), "shuffle": False}
        if cuda_enabled:
            loader_kwargs.update({"num_workers": 1, "pin_memory": True})
        data = list(DataLoader(self.dataset, **loader_kwargs))[0]
        data = data.to(device)

        # Load model
        model = Model(n_chars)
        model.load_state_dict(torch.load(model_input_path))
        model.eval()
        model = model.to(device)

        # Get predictions
        scores = model(data)  # pylint: disable=not-callable
        scores = scores.detach().cpu().numpy()

        # Prepare characters for frontend
        self.characters = self.get_characters(self.dataset, scores)

        self.app = Flask(__name__)

        route(self)

    def run(self, **kwargs: Any) -> None:
        self.app.run(**kwargs)

    @staticmethod
    def get_characters(
        dataset: AssignmentDataset, scores: np.ndarray
    ) -> list[Character]:
        """Get characters.

        Arguments:
            dataset: Assignment dataset
            scores: Scores
        Returns:
            List of characters
        """
        characters = []
        i = 0
        for char_bytes, score in zip(dataset.unassigned_char_bytes, scores):
            char_array = dataset.bytes_to_array(char_bytes)
            predictions = np.array(characters)[list(np.argsort(score))[::-1]]
            characters.append(Character(i, char_array, None, predictions[:10]))
            i += 1
        for char_bytes, assignment in dataset.assigned_char_bytes.items():
            char_array = dataset.bytes_to_array(char_bytes)
            characters.append(Character(i, char_array, assignment))
            i += 1

        return characters
