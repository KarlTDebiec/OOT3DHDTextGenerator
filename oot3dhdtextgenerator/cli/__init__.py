#!/usr/bin/env python
#  Copyright 2020-2022 Karl T Debiec
#  All rights reserved. This software may be modified and distributed under
#  the terms of the BSD license. See the LICENSE file for details.
"""Command line interfaces."""
from oot3dhdtextgenerator.cli.learning_dataset_generator_cli import (
    LearningDatasetGeneratorCli,
)
from oot3dhdtextgenerator.cli.model_trainer_cli import ModelTrainerCli

__all__ = [
    "LearningDatasetGeneratorCli",
    "ModelTrainerCli",
]