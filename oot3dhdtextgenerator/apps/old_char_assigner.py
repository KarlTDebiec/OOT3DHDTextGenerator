#  Copyright 2020-2023 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Character assigner."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from oot3dhdtextgenerator.core import AssignmentDataset


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
        dataset = AssignmentDataset(assignment_file)
        loader_kwargs = dict(batch_size=len(dataset))
        if cuda_enabled:
            loader_kwargs.update(dict(num_workers=1, pin_memory=True, shuffle=True))
        loader = DataLoader(dataset, **loader_kwargs)
        data = list(loader)[0]
        data = data.to(device)

        # Load model
        # model = Model(n_chars)
        # state_dict = torch.load(model_infile)
        # model.load_state_dict(state_dict)
        # model.eval()
        # model = model.to(device)

        # Get predictions
        # scores = model(data)
        # scores = scores.detach().cpu().numpy()
        # for image, score in zip(images, scores):
        #     Image.fromarray(image).show()
        #     print(score)
        #     print(characters[np.argmin(score)])
        #     char = input("Character: ")
        #     if char != "":
        #         project.assign(image, char)
        cls.assign(assignment_dataset=dataset)
