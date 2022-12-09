#!/usr/bin/env python
#  Copyright 2020-2022 Karl T Debiec
#  All rights reserved. This software may be modified and distributed under
#  the terms of the BSD license. See the LICENSE file for details.
"""Character assigner."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor

from oot3dhdtextgenerator.core import AssignmentDataset, Model

# TODO: Load dataset in form supported by model
# TODO: Make and display model predictions
# TODO: Create a web GUI


class CharAssigner:
    """Character assigner"""

    @classmethod
    def run(
        cls,
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
        dataset = AssignmentDataset(assignment_file)

        # Configure model
        cuda_enabled = torch.cuda.is_available() and cuda_enabled
        mps_enabled = torch.backends.mps.is_available() and mps_enabled
        if cuda_enabled:
            device = torch.device("cuda")
        elif mps_enabled:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        model = Model(10)
        state_dict = torch.load(model_infile)
        model.load_state_dict(state_dict)
        model.eval()
        model = model.to(device)
        yat = dataset.decode_images(dataset.unassigned_chars)
        transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
        eee = transform(yat[0])

        # Assign chars
        for char_bytes in dataset.unassigned_chars:
            char_array = np.frombuffer(char_bytes, dtype=np.uint8).reshape((16, 16))
            char_image = Image.fromarray(char_array)
            char_image.show()
            char = input("Character: ")
            if char != "":
                dataset.assign(char_bytes, char)
        pass
