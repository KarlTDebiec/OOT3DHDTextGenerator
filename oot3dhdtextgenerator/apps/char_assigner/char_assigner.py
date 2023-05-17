#  Copyright 2020-2023 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Character assigner."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from flask import Flask
from torch.utils.data import DataLoader

from oot3dhdtextgenerator.apps.char_assigner.character import Character
from oot3dhdtextgenerator.apps.char_assigner.routes import route
from oot3dhdtextgenerator.common import validate_input_file
from oot3dhdtextgenerator.core import AssignmentDataset


class CharAssigner:
    def __init__(
        self,
        n_chars: int,
        assignment_file: Path,
        model_infile: Path,
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
        cuda_enabled = torch.cuda.is_available() and cuda_enabled
        mps_enabled = torch.backends.mps.is_available() and mps_enabled
        if cuda_enabled:
            device = torch.device("cuda")
        elif mps_enabled:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        # Load assignment data
        self.assignment_file = validate_input_file(assignment_file)
        dataset = AssignmentDataset(self.assignment_file)

        characters = []
        i = 0
        for char_bytes in dataset.unassigned_char_bytes:
            char_array = dataset.bytes_to_array(char_bytes)
            characters.append(Character(i, char_array, None))
            i += 1
        for char_bytes, assignment in dataset.assigned_char_bytes.items():
            char_array = dataset.bytes_to_array(char_bytes)
            characters.append(Character(i, char_array, assignment))
            i += 1
        self.characters = characters

        loader_kwargs = dict(batch_size=len(dataset))
        if cuda_enabled:
            loader_kwargs.update(dict(num_workers=1, pin_memory=True, shuffle=True))
        loader = DataLoader(dataset, **loader_kwargs)
        data = list(loader)[0]
        data = data.to(device)
        self.dataset = dataset

        self.app = Flask(__name__)

        route(self)

    def run(self, **kwargs: Any) -> None:
        self.app.run(**kwargs)
