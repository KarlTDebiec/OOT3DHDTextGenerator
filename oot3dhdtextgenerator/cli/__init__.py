#  Copyright 2020-2025 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Command-line interfaces."""

from __future__ import annotations

from oot3dhdtextgenerator.cli.char_assigner_cli import CharAssignerCli
from oot3dhdtextgenerator.cli.learning_dataset_generator_cli import (
    LearningDatasetGeneratorCli,
)
from oot3dhdtextgenerator.cli.model_trainer_cli import ModelTrainerCli

__all__ = [
    "CharAssignerCli",
    "LearningDatasetGeneratorCli",
    "ModelTrainerCli",
]
