#  Copyright 2020-2026 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Core code.

This module may import from: common, data

Hierarchy within module:
* assignment_dataset / training_dataset
* model
"""

from .assignment_dataset import AssignmentDataset
from .model import Model
from .training_dataset import TrainingDataset

__all__ = [
    "AssignmentDataset",
    "TrainingDataset",
    "Model",
]
