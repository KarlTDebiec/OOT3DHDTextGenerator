#!/usr/bin/env python
#  Copyright 2020-2023 Karl T Debiec. All rights reserved. This software may be modified
#  and distributed under the terms of the BSD license. See the LICENSE file for details.
"""Optical character recognition model trainer command-line interface."""
from __future__ import annotations

from argparse import ArgumentParser
from typing import Any, Type

from pipescaler.core.cli import UtilityCli

from oot3dhdtextgenerator.common import (
    get_arg_groups_by_name,
    input_file_arg,
    output_file_arg,
)
from oot3dhdtextgenerator.utilities import ModelTrainer


class ModelTrainerCli(UtilityCli):
    """Optical character recognition model trainer command-line interface."""

    @classmethod
    def add_arguments_to_argparser(cls, parser: ArgumentParser) -> None:
        """Add arguments to a nascent argument parser.

        Arguments:
            parser: Nascent argument parser
        """
        super().add_arguments_to_argparser(parser)

        arg_groups = get_arg_groups_by_name(
            parser,
            "input arguments",
            "operation arguments",
            "output arguments",
            optional_arguments_name="additional arguments",
        )
        arg_groups["input arguments"].add_argument(
            "--train-infile",
            type=input_file_arg(),
            default="train_100.h5",
            help="train data input file (default: %(default)s)",
        )
        arg_groups["input arguments"].add_argument(
            "--test-infile",
            type=input_file_arg(),
            default="test_100.h5",
            help="test data input file (default: %(default)s)",
        )

        # Operation arguments
        arg_groups["operation arguments"].add_argument(
            "--batch-size",
            type=int,
            default=100,
            help="batch size for training (default: %(default)d)",
        )
        arg_groups["operation arguments"].add_argument(
            "--test-batch-size",
            type=int,
            default=1000,
            help="batch size for testing (default: %(default)d)",
        )
        arg_groups["operation arguments"].add_argument(
            "--epochs",
            type=int,
            default=1,
            help="number of epochs to train (default: %(default)d)",
        )
        arg_groups["operation arguments"].add_argument(
            "--lr",
            type=float,
            default=1.0,
            help="learning rate (default: %(default)f)",
        )
        arg_groups["operation arguments"].add_argument(
            "--gamma",
            type=float,
            default=0.7,
            help="learning rate step gamma (default: %(default)f)",
        )
        arg_groups["operation arguments"].add_argument(
            "--seed",
            type=int,
            default=1,
            help="random seed (default: %(default)d)",
        )
        arg_groups["operation arguments"].add_argument(
            "--disable-cuda",
            dest="cuda_enabled",
            action="store_false",
            default=True,
            help="disable CUDA",
        )
        arg_groups["operation arguments"].add_argument(
            "--disable-mps",
            dest="mps_enabled",
            action="store_false",
            default=True,
            help="disable macOS GPU",
        )
        arg_groups["operation arguments"].add_argument(
            "--dry-run",
            action="store_true",
            default=False,
            help="check a single pass",
        )

        # Output arguments
        arg_groups["output arguments"].add_argument(
            "--log-interval",
            type=int,
            default=10,
            help="training status logging interval (default: %(default)d)",
        )
        arg_groups["output arguments"].add_argument(
            "--model-outfile",
            type=output_file_arg(),
            default="model.pth",
            help="model output file (default: %(default)s)",
        )

    @classmethod
    def main_internal(cls, **kwargs: Any) -> None:
        """Execute with provided keyword arguments."""
        utility_cls = cls.utility()
        utility_cls.run(**kwargs)

    @classmethod
    def utility(cls) -> Type[ModelTrainer]:
        """Type of utility wrapped by command-line interface."""
        return ModelTrainer


if __name__ == "__main__":
    ModelTrainerCli.main()
