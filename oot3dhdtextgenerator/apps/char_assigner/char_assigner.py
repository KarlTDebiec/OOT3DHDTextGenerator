#  Copyright 2020-2025 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Character assigner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from flask import Flask
from torch.utils.data import DataLoader

from oot3dhdtextgenerator.apps.char_assigner.character import Character
from oot3dhdtextgenerator.apps.char_assigner.routes import route
from oot3dhdtextgenerator.common.validation import validate_input_file
from oot3dhdtextgenerator.core import AssignmentDataset, Model


class CharAssigner:
    """Assign characters using a trained model."""

    def __init__(
        self,
        n_chars: int,
        assignment_file: Path,
        model_infile: Path,
        *,
        cuda_enabled: bool = True,
        mps_enabled: bool = True,
    ) -> None:
        """Run character assigner.

        Arguments:
            n_chars: Number of characters included in model
            assignment_file: Assignment HDF5 file
            model_infile: Model pth file
            cuda_enabled: Whether to use CUDA
            mps_enabled: Whether to use macOS GPU
        """
        # Determine which device to use
        self.n_chars = n_chars
        self.assignment_file = validate_input_file(assignment_file)
        self.model_infile = model_infile
        self.cuda_enabled = torch.cuda.is_available() and cuda_enabled
        self.mps_enabled = mps_enabled

        cuda_enabled = self.cuda_enabled
        if cuda_enabled:
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and mps_enabled:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        # Load assignment data
        self.dataset = AssignmentDataset(self.assignment_file)
        loader_kwargs = {"batch_size": len(self.dataset), "shuffle": False}
        if cuda_enabled:
            loader_kwargs.update({"num_workers": 1, "pin_memory": True})
        data = list(DataLoader(self.dataset, **loader_kwargs))[0]
        data = data.to(device)

        # Load model
        model = Model(n_chars)
        model.load_state_dict(torch.load(model_infile))
        model.eval()
        model = model.to(device)

        # Get predictions
        scores = model(data)  # pylint: disable=not-callable
        scores = scores.detach().cpu().numpy()

        # Prepare characters for frontend
        self.characters = self.get_characters(self.dataset, scores)

        self.app = Flask(__name__)

        route(self)

    def __repr__(self) -> str:
        """Representation."""
        return (
            f"{self.__class__.__name__}("
            f"{self.n_chars!r}, "
            f"Path({self.assignment_file!r}), "
            f"Path({self.model_infile!r}), "
            f"cuda_enabled={self.cuda_enabled!r}, "
            f"mps_enabled={self.mps_enabled!r})"
        )

    def run(self, **kwargs: Any) -> None:
        """Run the Flask application."""
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
            pred_chars = [
                c.assignment
                for c in np.array(characters)[list(np.argsort(score))[::-1]]
            ]
            pred_chars = [p for p in pred_chars if p is not None][:10]
            characters.append(Character(i, char_array, None, pred_chars))
            i += 1
        for char_bytes, assignment in dataset.assigned_char_bytes.items():
            char_array = dataset.bytes_to_array(char_bytes)
            characters.append(Character(i, char_array, assignment))
            i += 1

        return characters
